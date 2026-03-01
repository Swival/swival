"""Public library API for swival: Session class and Result dataclass."""

import copy
from dataclasses import dataclass
from pathlib import Path

from .report import ConfigError, ReportCollector
from .snapshot import SnapshotState
from .thinking import ThinkingState
from .todo import TodoState
from .tracker import FileAccessTracker


def _resolve_dir_list(dirs: list, label: str) -> list[Path]:
    """Resolve a list of directory strings to absolute Paths with validation."""
    result = []
    for d in dirs:
        p = Path(d).expanduser().resolve()
        if not p.is_dir():
            raise ConfigError(f"{label} path is not a directory: {d}")
        if p == Path(p.anchor):
            raise ConfigError(f"{label} cannot be the filesystem root: {d}")
        result.append(p)
    return result


@dataclass
class Result:
    """Result of a session run or ask call."""

    answer: str | None
    exhausted: bool
    messages: list[dict]
    report: dict | None


class Session:
    """Programmatic interface to the swival agent loop.

    Stores configuration as plain attributes. Call .run() for single-shot
    questions or .ask() for multi-turn conversations.
    """

    def __init__(
        self,
        *,
        base_dir: str = ".",
        provider: str = "lmstudio",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_turns: int = 100,
        max_output_tokens: int = 32768,
        max_context_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float = 1.0,
        seed: int | None = None,
        allowed_commands: list[str] | None = None,
        yolo: bool = False,
        verbose: bool = False,
        system_prompt: str | None = None,
        no_system_prompt: bool = False,
        no_instructions: bool = False,
        no_skills: bool = False,
        skills_dir: list[str] | None = None,
        allowed_dirs: list[str] | None = None,
        allowed_dirs_ro: list[str] | None = None,
        sandbox: str = "builtin",
        sandbox_session: str | None = None,
        read_guard: bool = True,
        history: bool = True,
        config_dir: "Path | None" = None,
        proactive_summaries: bool = False,
        mcp_servers: dict | None = None,
        extra_body: dict | None = None,
    ):
        self.base_dir = base_dir
        self.config_dir = config_dir
        self.proactive_summaries = proactive_summaries
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.max_context_tokens = max_context_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.allowed_commands = allowed_commands
        self.yolo = yolo
        self.verbose = verbose
        self.system_prompt = system_prompt
        self.no_system_prompt = no_system_prompt
        self.no_instructions = no_instructions
        self.no_skills = no_skills
        self.skills_dir = skills_dir or []
        self.allowed_dirs = allowed_dirs or []
        self.allowed_dirs_ro = allowed_dirs_ro or []
        self.sandbox = sandbox
        self.sandbox_session = sandbox_session
        self.read_guard = read_guard
        self.history = history
        self.mcp_servers = mcp_servers
        self.extra_body = extra_body

        # Setup state (cached after first _setup())
        self._setup_done = False
        self._model_id: str | None = None
        self._api_base: str | None = None
        self._resolved_key: str | None = None
        self._context_length: int | None = None
        self._llm_kwargs: dict = {}
        self._resolved_commands: dict[str, str] = {}
        self._skills_catalog: dict = {}
        self._tools: list = []
        self._system_content: str | None = None
        self._instructions_loaded: list[str] = []
        self._allowed_dir_paths: list[Path] = []
        self._allowed_dir_ro_paths: list[Path] = []

        # MCP manager (created in _setup if mcp_servers is non-empty)
        self._mcp_manager = None

        # Per-conversation state (for ask() mode)
        self._conv_state: dict | None = None

    def _setup(self) -> None:
        """Perform one-time setup: resolve provider, commands, tools, system prompt."""
        if self._setup_done:
            return

        if self.sandbox == "agentfs":
            from .sandbox_agentfs import check_sandbox_available

            check_sandbox_available()

        from .agent import (
            resolve_provider,
            resolve_commands,
            build_tools,
            build_system_prompt,
            cleanup_old_cmd_outputs,
        )
        from .skills import discover_skills

        # Resolve provider
        (
            self._model_id,
            self._api_base,
            self._resolved_key,
            self._context_length,
            self._llm_kwargs,
        ) = resolve_provider(
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            max_context_tokens=self.max_context_tokens,
            verbose=self.verbose,
        )
        if self.extra_body is not None:
            self._llm_kwargs["extra_body"] = self.extra_body

        # Resolve --add-dir and --add-dir-ro paths
        self._allowed_dir_paths = _resolve_dir_list(self.allowed_dirs, "allowed_dirs")
        self._allowed_dir_ro_paths = _resolve_dir_list(
            self.allowed_dirs_ro, "allowed_dirs_ro"
        )

        # Resolve commands
        self._resolved_commands = resolve_commands(
            self.allowed_commands, self.yolo, self.base_dir
        )

        # Discover skills
        self._skills_catalog = {}
        if not self.no_skills:
            self._skills_catalog = discover_skills(
                self.base_dir, self.skills_dir, self.verbose
            )

        # Build tools
        self._tools = build_tools(
            self._resolved_commands, self._skills_catalog, self.yolo
        )

        # Initialize MCP servers
        if self.mcp_servers:
            from .mcp_client import McpManager

            self._mcp_manager = McpManager(self.mcp_servers, verbose=self.verbose)
            self._mcp_manager.start()
            mcp_tools = self._mcp_manager.list_tools()
            if mcp_tools:
                self._tools.extend(mcp_tools)

            from .agent import enforce_mcp_token_budget

            self._tools = enforce_mcp_token_budget(
                self._tools,
                self._mcp_manager,
                self._context_length,
                verbose=self.verbose,
            )

        # Build system prompt
        mcp_tool_info = self._mcp_manager.get_tool_info() if self._mcp_manager else None
        self._system_content, self._instructions_loaded = build_system_prompt(
            base_dir=self.base_dir,
            system_prompt=self.system_prompt,
            no_system_prompt=self.no_system_prompt,
            no_instructions=self.no_instructions,
            skills_catalog=self._skills_catalog,
            yolo=self.yolo,
            resolved_commands=self._resolved_commands,
            verbose=self.verbose,
            config_dir=self.config_dir,
            mcp_tool_info=mcp_tool_info,
        )

        # Clean up stale cmd_output files
        cleanup_old_cmd_outputs(self.base_dir)

        if self.verbose:
            from . import fmt

            fmt.init()

        self._setup_done = True

    def _make_initial_messages(self) -> list[dict]:
        """Create the initial messages list with system prompt if configured."""
        messages: list[dict] = []
        if self._system_content is not None:
            messages.append({"role": "system", "content": self._system_content})
        return messages

    def _make_per_run_state(self) -> dict:
        """Create fresh per-run state: thinking, tracker, skill roots, messages."""
        from .agent import CompactionState

        state = {
            "thinking_state": ThinkingState(verbose=self.verbose),
            "todo_state": TodoState(notes_dir=self.base_dir, verbose=self.verbose),
            "snapshot_state": SnapshotState(verbose=self.verbose),
            "file_tracker": FileAccessTracker() if self.read_guard else None,
            "skill_read_roots": list(self._allowed_dir_ro_paths),
            "messages": self._make_initial_messages(),
            "compaction_state": CompactionState() if self.proactive_summaries else None,
        }
        return state

    def _build_loop_kwargs(self, state: dict) -> dict:
        """Build kwargs for run_agent_loop() from setup + per-run state."""
        kwargs = dict(
            api_base=self._api_base,
            model_id=self._model_id,
            max_turns=self.max_turns,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            context_length=self._context_length,
            base_dir=self.base_dir,
            thinking_state=state["thinking_state"],
            todo_state=state["todo_state"],
            snapshot_state=state["snapshot_state"],
            resolved_commands=self._resolved_commands,
            skills_catalog=self._skills_catalog,
            skill_read_roots=state["skill_read_roots"],
            extra_write_roots=self._allowed_dir_paths,
            yolo=self.yolo,
            verbose=self.verbose,
            llm_kwargs=self._llm_kwargs,
            file_tracker=state["file_tracker"],
        )
        if state.get("compaction_state") is not None:
            kwargs["compaction_state"] = state["compaction_state"]
        if self._mcp_manager is not None:
            kwargs["mcp_manager"] = self._mcp_manager
        return kwargs

    def run(self, question: str, *, report: bool = False) -> Result:
        """Single-shot: run a question with fresh state. Each call is independent."""
        self._setup()

        from .agent import run_agent_loop, append_history

        state = self._make_per_run_state()
        messages = state["messages"]
        messages.append({"role": "user", "content": question})

        collector = ReportCollector() if report else None
        loop_kwargs = self._build_loop_kwargs(state)

        answer, exhausted = run_agent_loop(
            messages, self._tools, **loop_kwargs, report=collector
        )

        if self.history and answer:
            append_history(self.base_dir, question, answer, diagnostics=self.verbose)

        report_dict = None
        if collector:
            report_dict = collector.build_report(
                task=question,
                model=self._model_id or "unknown",
                provider=self.provider,
                settings={
                    "max_turns": self.max_turns,
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "seed": self.seed,
                    "yolo": self.yolo,
                },
                outcome="exhausted" if exhausted else "success",
                answer=answer,
                exit_code=2 if exhausted else 0,
                turns=collector.max_turn_seen,
                sandbox_mode=self.sandbox,
                sandbox_session=self.sandbox_session,
            )

        return Result(
            answer=answer,
            exhausted=exhausted,
            messages=copy.deepcopy(messages),
            report=report_dict,
        )

    def ask(self, question: str) -> Result:
        """Conversational: share context across questions (like the REPL)."""
        self._setup()

        from .agent import run_agent_loop, append_history

        if self._conv_state is None:
            self._conv_state = self._make_per_run_state()

        state = self._conv_state
        messages = state["messages"]
        messages.append({"role": "user", "content": question})

        loop_kwargs = self._build_loop_kwargs(state)

        answer, exhausted = run_agent_loop(messages, self._tools, **loop_kwargs)

        if self.history and answer:
            append_history(self.base_dir, question, answer, diagnostics=self.verbose)

        return Result(
            answer=answer,
            exhausted=exhausted,
            messages=copy.deepcopy(messages),
            report=None,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._mcp_manager is not None:
            self._mcp_manager.close()

    def reset(self) -> None:
        """Clear conversation state without invalidating setup. Next ask() starts fresh."""
        self._conv_state = None
