"""Public library API for swival: Session class and Result dataclass."""

import copy
from dataclasses import dataclass
from pathlib import Path

from .agent import _InteractionPolicy, _apply_interaction_policy
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
        sandbox_strict_read: bool = False,
        sandbox_auto_session: bool = True,
        read_guard: bool = True,
        history: bool = True,
        memory: bool = True,
        memory_full: bool = False,
        config_dir: "Path | None" = None,
        proactive_summaries: bool = False,
        mcp_servers: dict | None = None,
        a2a_servers: dict | None = None,
        extra_body: dict | None = None,
        reasoning_effort: str | None = None,
        continue_here: bool = True,
        sanitize_thinking: bool | None = None,
        cache: bool = False,
        cache_dir: str | None = None,
        scratch_dir: str | None = None,
        retries: int = 5,
        encrypt_secrets: bool = False,
        encrypt_secrets_key: str | None = None,
        encrypt_secrets_tweak: str | None = None,
        encrypt_secrets_patterns: list | None = None,
    ):
        self.base_dir = base_dir
        self.scratch_dir = scratch_dir
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
        if retries < 1:
            raise ValueError("retries must be >= 1")
        self.retries = retries
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
        self.sandbox_strict_read = sandbox_strict_read
        self.sandbox_auto_session = sandbox_auto_session
        self.read_guard = read_guard
        self.history = history
        self.memory = memory
        self.memory_full = memory_full
        self.mcp_servers = mcp_servers
        self.a2a_servers = a2a_servers
        self.extra_body = extra_body
        self.reasoning_effort = reasoning_effort
        self.sanitize_thinking = sanitize_thinking
        self.continue_here = continue_here
        self.cache = cache
        self.cache_dir = cache_dir
        self.encrypt_secrets = encrypt_secrets
        self.encrypt_secrets_key = encrypt_secrets_key
        self.encrypt_secrets_tweak = encrypt_secrets_tweak
        self.encrypt_secrets_patterns = encrypt_secrets_patterns

        # Streaming hooks (set externally, e.g. by A2A server)
        self.event_callback = None
        self.cancel_flag = None

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

        # Cache handle (created in _setup if cache is enabled)
        self._llm_cache = None

        # MCP manager (created in _setup if mcp_servers is non-empty)
        self._mcp_manager = None

        # A2A manager (created in _setup if a2a_servers is non-empty)
        self._a2a_manager = None

        # Secret shield (created in _setup if encrypt_secrets is enabled)
        self._secret_shield = None

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
            _filter_command_tool_schemas,
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
        if self.reasoning_effort is not None:
            self._llm_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.sanitize_thinking is not None:
            self._llm_kwargs["sanitize_thinking"] = self.sanitize_thinking
        self._llm_kwargs["max_retries"] = self.retries

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
            # Auto-grant read access to external skill directories so the LLM
            # can read supporting files (references/, scripts/, etc.) without
            # requiring an explicit --add-dir-ro.
            for skill in self._skills_catalog.values():
                if not skill.is_local and skill.path not in self._allowed_dir_ro_paths:
                    self._allowed_dir_ro_paths.append(skill.path)

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

        # Initialize A2A agents
        if self.a2a_servers:
            from .a2a_client import A2aManager

            self._a2a_manager = A2aManager(self.a2a_servers, verbose=self.verbose)
            self._a2a_manager.start()
            a2a_tools = self._a2a_manager.list_tools()
            if a2a_tools:
                self._tools.extend(a2a_tools)

        # Initialize secret encryption shield
        if self.encrypt_secrets:
            from .secrets import SecretShield

            self._secret_shield = SecretShield.from_config(
                key_hex=self.encrypt_secrets_key,
                tweak_str=self.encrypt_secrets_tweak,
                extra_patterns=self.encrypt_secrets_patterns,
            )

        # Open cache
        if self.cache:
            from .cache import open_cache

            self._llm_cache = open_cache(self.base_dir, self.cache_dir)

        # Build system prompt (without memory — memory is injected per-call
        # in run()/ask() so it can be keyed from the user's question).
        mcp_tool_info = self._mcp_manager.get_tool_info() if self._mcp_manager else None
        a2a_tool_info = self._a2a_manager.get_tool_info() if self._a2a_manager else None
        # Build list of tool schemas exposable to command provider (MCP/A2A/skills).
        _command_tool_schemas = (
            _filter_command_tool_schemas(self._tools) or None
            if self.provider == "command"
            else None
        )

        self._system_content, self._instructions_loaded = build_system_prompt(
            base_dir=self.base_dir,
            system_prompt=self.system_prompt,
            no_system_prompt=self.no_system_prompt,
            no_instructions=self.no_instructions,
            no_memory=True,
            skills_catalog=self._skills_catalog,
            yolo=self.yolo,
            resolved_commands=self._resolved_commands,
            verbose=self.verbose,
            config_dir=self.config_dir,
            mcp_tool_info=mcp_tool_info,
            a2a_tool_info=a2a_tool_info,
            no_continue=not self.continue_here,
            provider=self.provider,
            command_tool_schemas=_command_tool_schemas,
        )

        # Clean up stale cmd_output files
        cleanup_old_cmd_outputs(self.base_dir)

        if self.verbose:
            from . import fmt

            fmt.init()

        self._setup_done = True

    def _system_with_memory(
        self,
        question: str,
        report: "ReportCollector | None" = None,
        policy: "_InteractionPolicy" = "autonomous",
    ) -> str | None:
        """Return system content with memory and interaction policy applied."""
        if self._system_content is None:
            return None

        result = self._system_content

        # Inject memory (skipped for custom prompts and when memory is disabled)
        if self.memory and not self.system_prompt:
            from .agent import load_memory

            memory_text = load_memory(
                self.base_dir,
                verbose=self.verbose,
                memory_full=self.memory_full,
                user_query=question,
                report=report,
            )
            if memory_text:
                result = result + "\n\n" + memory_text

        # Substitute interaction-policy placeholders
        result = _apply_interaction_policy(result, policy)

        return result

    def _make_initial_messages(
        self,
        system_content: str | None = None,
    ) -> list[dict]:
        """Create the initial messages list with system prompt if configured."""
        content = system_content if system_content is not None else self._system_content
        messages: list[dict] = []
        if content is not None:
            messages.append({"role": "system", "content": content})
        return messages

    def _make_per_run_state(self, system_content: str | None = None) -> dict:
        """Create fresh per-run state: thinking, tracker, skill roots, messages."""
        from .agent import CompactionState

        state = {
            "thinking_state": ThinkingState(verbose=self.verbose),
            "todo_state": TodoState(
                notes_dir=self.base_dir, verbose=self.verbose, todo_dir=self.scratch_dir
            ),
            "snapshot_state": SnapshotState(verbose=self.verbose),
            "file_tracker": FileAccessTracker() if self.read_guard else None,
            "skill_read_roots": list(self._allowed_dir_ro_paths),
            "messages": self._make_initial_messages(system_content),
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
            scratch_dir=self.scratch_dir,
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
            continue_here=self.continue_here,
            cache=self._llm_cache,
        )
        if state.get("compaction_state") is not None:
            kwargs["compaction_state"] = state["compaction_state"]
        if self._mcp_manager is not None:
            kwargs["mcp_manager"] = self._mcp_manager
        if self._a2a_manager is not None:
            kwargs["a2a_manager"] = self._a2a_manager
        if self._secret_shield is not None:
            kwargs["secret_shield"] = self._secret_shield
        if self.event_callback is not None:
            kwargs["event_callback"] = self.event_callback
        if self.cancel_flag is not None:
            kwargs["cancel_flag"] = self.cancel_flag
        return kwargs

    def run(self, question: str, *, report: bool = False) -> Result:
        """Single-shot: run a question with fresh state. Each call is independent."""
        self._setup()

        from .agent import run_agent_loop, append_history

        collector = ReportCollector() if report else None
        system_content = self._system_with_memory(question, collector)
        state = self._make_per_run_state(system_content=system_content)
        messages = state["messages"]
        messages.append({"role": "user", "content": question})
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
                sandbox_strict_read=self.sandbox_strict_read,
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
            system_content = self._system_with_memory(question, policy="interactive")
            self._conv_state = self._make_per_run_state(system_content=system_content)

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
        if self._llm_cache is not None:
            self._llm_cache.close()
            self._llm_cache = None
        if self._mcp_manager is not None:
            self._mcp_manager.close()
        if self._a2a_manager is not None:
            self._a2a_manager.close()
        if self._secret_shield is not None:
            self._secret_shield.destroy()
            self._secret_shield = None

    def reset(self) -> None:
        """Clear conversation state without invalidating setup. Next ask() starts fresh."""
        self._conv_state = None
