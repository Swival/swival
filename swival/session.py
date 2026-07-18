"""Public library API for swival: Session class and Result dataclass."""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import tools
from .agent import _InteractionPolicy, _apply_interaction_policy
from .config import _UNSET, NETWORK_MODES, first_remote_integration
from .goal import GoalState
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
        user_agent: str | None = None,
        base_url: str | None = None,
        max_turns: int = 100,
        max_output_tokens: int = 32768,
        max_context_tokens: int | None = None,
        max_output_lines: int = 2000,
        max_output_kb: int = 50,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        files: str | object = _UNSET,
        commands: str | list[str] | None | object = _UNSET,
        yolo: bool = False,
        verbose: bool = False,
        system_prompt: str | None = None,
        no_system_prompt: bool = False,
        no_instructions: bool = False,
        no_skills: bool = False,
        skills_dir: list[str] | None = None,
        allowed_dirs: list[str] | None = None,
        allowed_dirs_ro: list[str] | None = None,
        network: str = "full",
        sandbox: str = "builtin",
        sandbox_session: str | None = None,
        sandbox_strict_read: bool = False,
        sandbox_auto_session: bool = True,
        nono_profile: str | None = None,
        nono_rollback: bool = False,
        nono_block_net: bool = False,
        nono_allow_domain: list[str] | None = None,
        nono_network_profile: str | None = None,
        nono_credential: list[str] | None = None,
        nono_audit_integrity: bool = False,
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
        show_thinking: bool | None = None,
        prompt_cache: bool = True,
        cache: bool = False,
        cache_dir: str | None = None,
        scratch_dir: str | None = None,
        retries: int = 5,
        encrypt_secrets: bool = False,
        encrypt_secrets_key: str | None = None,
        encrypt_secrets_tweak: str | None = None,
        encrypt_secrets_patterns: list | None = None,
        llm_filter: str | None = None,
        trace_dir: str | None = None,
        subagents: bool = False,
        lifecycle_command: str | None = None,
        lifecycle_timeout: int = 300,
        lifecycle_fail_closed: bool = False,
        lifecycle_enabled: bool = True,
        command_middleware: str | None = None,
        aws_profile: str | None = None,
        project: str | None = None,
        location: str | None = None,
        approved_buckets: set[str] | None = None,
        metaskills: str = "local",
        storm_breaker: bool = True,
        flatten_mcp_schemas: bool = True,
    ):
        self.base_dir = base_dir
        self.scratch_dir = scratch_dir
        self.config_dir = config_dir
        self.proactive_summaries = proactive_summaries
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.user_agent = user_agent
        self.base_url = base_url
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.max_context_tokens = max_context_tokens
        self.max_output_lines = max_output_lines
        self.max_output_kb = max_output_kb
        # Tool-output caps are process-global (shared by all tool modules)
        tools.set_output_caps(max_output_lines, max_output_kb)
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        if retries < 1:
            raise ValueError("retries must be >= 1")
        self.retries = retries
        # Resolve _UNSET defaults; yolo only upgrades when the caller didn't
        # pass an explicit value (same semantics as the CLI path).
        if files is _UNSET:
            files = "all" if yolo else "some"
        if commands is _UNSET:
            commands = "all"
        self.commands = commands
        self.files = files
        self.verbose = verbose
        self.system_prompt = system_prompt
        self.no_system_prompt = no_system_prompt
        self.no_instructions = no_instructions
        self.no_skills = no_skills
        self.skills_dir = skills_dir or []
        self.allowed_dirs = allowed_dirs or []
        self.allowed_dirs_ro = allowed_dirs_ro or []
        if network not in NETWORK_MODES:
            raise ConfigError(
                f"'network' must be one of {NETWORK_MODES!r}, got {network!r}"
            )
        self.network = network
        self._net_jail: list[str] | None = None
        self.sandbox = sandbox
        self.sandbox_session = sandbox_session
        self.sandbox_strict_read = sandbox_strict_read
        self.sandbox_auto_session = sandbox_auto_session
        # nono knobs are stored for report metadata and validation only.
        # Session does not bootstrap nono — the CLI re-exec path does that,
        # or the user wraps their process with `nono run` externally.
        self.nono_profile = nono_profile
        self.nono_rollback = nono_rollback
        self.nono_block_net = nono_block_net
        self.nono_allow_domain = nono_allow_domain or []
        self.nono_network_profile = nono_network_profile
        self.nono_credential = nono_credential or []
        self.nono_audit_integrity = nono_audit_integrity
        self.read_guard = read_guard
        self.history = history
        self.memory = memory
        self.memory_full = memory_full
        self.mcp_servers = mcp_servers
        self.a2a_servers = a2a_servers
        self.extra_body = extra_body
        self.reasoning_effort = reasoning_effort
        self.sanitize_thinking = sanitize_thinking
        self.show_thinking = show_thinking
        self.prompt_cache = prompt_cache
        self.continue_here = continue_here
        self.cache = cache
        self.cache_dir = cache_dir
        self.encrypt_secrets = encrypt_secrets
        self.encrypt_secrets_key = encrypt_secrets_key
        self.encrypt_secrets_tweak = encrypt_secrets_tweak
        self.encrypt_secrets_patterns = encrypt_secrets_patterns
        self.llm_filter = llm_filter
        self.trace_dir = trace_dir
        self._trace_session_id: str | None = None
        self.subagents = subagents
        self.lifecycle_command = lifecycle_command
        self.lifecycle_timeout = lifecycle_timeout
        self.lifecycle_fail_closed = lifecycle_fail_closed
        self.lifecycle_enabled = lifecycle_enabled
        self.command_middleware = command_middleware
        self.aws_profile = aws_profile
        self.project = project
        self.location = location
        self.approved_buckets = approved_buckets
        self.metaskills = metaskills
        self.storm_breaker = storm_breaker
        self.flatten_mcp_schemas = flatten_mcp_schemas
        self._mcp_flattened_schemas: dict[str, object] = {}

        # Streaming / cancellation hooks (set externally, e.g. by A2A server).
        # event_callback receives (kind, data) where kind is one of the
        # EVENT_* constants from a2a_types (text_chunk, tool_start,
        # tool_finish, tool_error, status_update) and data is a dict
        # with event-specific keys.
        self.event_callback: Callable[[str, dict], None] | None = None
        # cancel_flag: set the event to request graceful cancellation of the
        # running agent loop.
        self.cancel_flag: threading.Event | None = None

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

        # Lifecycle hook state
        self._lifecycle_git_meta: dict | None = None
        self._lifecycle_startup_result: dict | None = None
        self._lifecycle_exit_ran: bool = False

        # Per-conversation state (for ask() mode)
        self._conv_state: dict | None = None

    def _setup(self) -> None:
        """Perform one-time setup: resolve provider, commands, tools, system prompt."""
        if self._setup_done:
            return

        if self.verbose:
            from . import fmt

            fmt.init()

        # Network policy. Session never bootstraps a sandbox, so "none" is
        # only honest when the process already runs inside nono with the
        # network provably blocked (the CLI re-exec or an external
        # `nono run --block-net` wrapper — the capability file records the
        # policy, and a wrapper without --block-net fails closed here).
        # "provider-only" is enforced in-library: agent subprocesses are
        # wrapped at dispatch time, so it works from any Python process with
        # nono installed. Checked before sandbox availability so
        # configuration contradictions surface regardless of the host
        # environment.
        if self.network == "none":
            from .sandbox_nono import is_inside_nono, verify_net_blocked

            if self.provider != "command":
                raise ConfigError(
                    'network="none" requires the command provider; provider '
                    f"{self.provider!r} needs network access. For hosted or "
                    'local HTTP providers use network="provider-only".'
                )
            if not is_inside_nono():
                raise ConfigError(
                    'network="none" requires an OS-level sandbox that Session '
                    "cannot bootstrap. Use the CLI (swival --network none) for "
                    "automatic re-exec, or wrap your process with "
                    "`nono run --block-net` externally."
                )
            verify_net_blocked('network="none"')
        elif self.network == "provider-only":
            if self.sandbox == "nono":
                raise ConfigError(
                    'network="provider-only" is incompatible with '
                    'sandbox="nono"; agent commands would need to nest nono '
                    "inside nono"
                )
            from .sandbox_nono import check_wrapper_available

            check_wrapper_available()

        if self.network != "full":
            remote = first_remote_integration(self.mcp_servers, self.a2a_servers)
            if remote is not None:
                kind, name = remote
                label = "URL-backed MCP server" if kind == "mcp" else "A2A server"
                raise ConfigError(
                    f'network="{self.network}" is incompatible with '
                    f"{label} {name!r}; remove it"
                )

        if self.sandbox == "agentfs":
            from .sandbox_agentfs import check_sandbox_available

            check_sandbox_available()

        if self.sandbox == "nono":
            from .sandbox_nono import check_sandbox_available, verify_net_blocked

            check_sandbox_available()
            # nono_block_net is a claim about the external wrapper that the
            # report repeats as sandbox.network = "offline"; don't take it
            # on faith.
            if self.nono_block_net:
                verify_net_blocked("nono_block_net=True")

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
            aws_profile=self.aws_profile,
            project=self.project,
            location=self.location,
        )
        if self.user_agent is not None:
            self._llm_kwargs["user_agent"] = self.user_agent
        if self.extra_body is not None:
            self._llm_kwargs["extra_body"] = self.extra_body
        if self.reasoning_effort is not None:
            self._llm_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.sanitize_thinking is not None:
            self._llm_kwargs["sanitize_thinking"] = self.sanitize_thinking
        if self.show_thinking is not None:
            self._llm_kwargs["show_thinking"] = self.show_thinking
        if not self.prompt_cache:
            self._llm_kwargs["prompt_cache"] = False
        self._llm_kwargs["max_retries"] = self.retries

        # Resolve --add-dir and --add-dir-ro paths
        self._allowed_dir_paths = _resolve_dir_list(self.allowed_dirs, "allowed_dirs")
        self._allowed_dir_ro_paths = _resolve_dir_list(
            self.allowed_dirs_ro, "allowed_dirs_ro"
        )

        # Resolve commands
        from .command_policy import CommandPolicy

        cmds = self.commands
        if cmds is None or cmds == "all":
            self._resolved_commands = {}
            self._commands_unrestricted = True
            self._command_policy = CommandPolicy("full")
        elif cmds == "none":
            self._resolved_commands = {}
            self._commands_unrestricted = False
            self._command_policy = CommandPolicy("none")
        elif cmds == "ask":
            self._resolved_commands = {}
            self._commands_unrestricted = True
            self._command_policy = CommandPolicy(
                "ask", approved_buckets=self.approved_buckets or set()
            )
        elif isinstance(cmds, list):
            self._resolved_commands = resolve_commands(cmds, self.base_dir)
            self._commands_unrestricted = False
            self._command_policy = CommandPolicy(
                "allowlist", allowed_basenames=set(self._resolved_commands)
            )
        else:
            raise ConfigError(
                f"'commands' must be 'all', 'none', 'ask', or a list of command names, "
                f"got {cmds!r}"
            )

        self._shell_allowed = self._command_policy.shell_allowed

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

        # Build the provider-only jail only after skill discovery: its read
        # grants must include external skill directories, or wrapped
        # subprocesses cannot read skill files the file tools allow.
        if self.network == "provider-only":
            from .sandbox_nono import build_block_net_wrapper

            self._net_jail = build_block_net_wrapper(
                base_dir=self.base_dir,
                add_dirs=list(self._allowed_dir_paths),
                read_dirs=list(self._allowed_dir_ro_paths),
            )

        # Resolve metaskills
        self._metaskills_policy = self.metaskills if self.metaskills != "off" else "off"
        self._metaskill_names: list[str] = []
        if not self.no_skills and self._metaskills_policy != "off":
            from .metaskills import get_executable_metaskills

            self._metaskill_names = get_executable_metaskills(
                self._skills_catalog, self._metaskills_policy
            )

        # Build tools
        self._tools = build_tools(
            self._resolved_commands,
            self._skills_catalog,
            commands_unrestricted=self._commands_unrestricted,
            shell_allowed=self._shell_allowed,
            subagents=self.subagents,
            metaskill_names=self._metaskill_names,
            network=self.network,
        )

        # Initialize MCP servers
        if self.mcp_servers:
            from .mcp_client import McpManager

            self._mcp_manager = McpManager(
                self.mcp_servers,
                verbose=self.verbose,
                flatten_schemas=self.flatten_mcp_schemas,
                net_jail=self._net_jail,
            )
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

        # --- Lifecycle startup hook ---
        if self.lifecycle_command and self.lifecycle_enabled:
            from .agent import _validate_external_command
            from .lifecycle import run_lifecycle_hook, _git_metadata

            _validate_external_command(self.lifecycle_command, "lifecycle_command")
            self._lifecycle_git_meta = _git_metadata(self.base_dir)
            self._lifecycle_startup_result = run_lifecycle_hook(
                self.lifecycle_command,
                "startup",
                self.base_dir,
                timeout=self.lifecycle_timeout,
                fail_closed=self.lifecycle_fail_closed,
                provider=self.provider,
                model=self._model_id,
                git_meta=self._lifecycle_git_meta,
                verbose=self.verbose,
            )

        # Validate command middleware executable at startup
        if self.command_middleware:
            from .agent import _validate_external_command

            _validate_external_command(self.command_middleware, "command_middleware")

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
            verbose=self.verbose,
            config_dir=self.config_dir,
            mcp_tool_info=mcp_tool_info,
            a2a_tool_info=a2a_tool_info,
            no_continue=not self.continue_here,
            provider=self.provider,
            command_tool_schemas=_command_tool_schemas,
            files_mode=self.files,
            metaskill_names=self._metaskill_names,
            subagents=self.subagents,
        )

        # Clean up stale cmd_output files
        cleanup_old_cmd_outputs(self.base_dir)

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

        return {
            "thinking_state": ThinkingState(verbose=self.verbose),
            "todo_state": TodoState(verbose=self.verbose),
            "snapshot_state": SnapshotState(verbose=self.verbose),
            "goal_state": GoalState(verbose=self.verbose),
            "file_tracker": FileAccessTracker() if self.read_guard else None,
            "skill_read_roots": list(self._allowed_dir_ro_paths),
            "messages": self._make_initial_messages(system_content),
            "compaction_state": CompactionState() if self.proactive_summaries else None,
            "resolved_system_content": system_content,
            # Shared turn budget so a /extend issued via parse_commands persists
            # across subsequent ask() calls, mirroring the REPL.
            "turn_state": {"max_turns": self.max_turns, "turns_used": 0},
        }

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
            goal_state=state["goal_state"],
            resolved_commands=self._resolved_commands,
            skills_catalog=self._skills_catalog,
            skill_read_roots=state["skill_read_roots"],
            extra_write_roots=self._allowed_dir_paths,
            files_mode=self.files,
            commands_unrestricted=self._commands_unrestricted,
            shell_allowed=self._shell_allowed,
            verbose=self.verbose,
            llm_kwargs=self._llm_kwargs,
            file_tracker=state["file_tracker"],
            continue_here=self.continue_here,
            cache=self._llm_cache,
            command_policy=self._command_policy,
            metaskills_policy=self._metaskills_policy,
            enabled_metaskills=set(self._metaskill_names or []),
            storm_breaker_enabled=self.storm_breaker,
            session=self,
            network_mode=self.network,
            net_jail=self._net_jail,
        )
        if state.get("compaction_state") is not None:
            kwargs["compaction_state"] = state["compaction_state"]
        if self._mcp_manager is not None:
            kwargs["mcp_manager"] = self._mcp_manager
        if self._a2a_manager is not None:
            kwargs["a2a_manager"] = self._a2a_manager
        if self.llm_filter is not None:
            kwargs["llm_filter"] = self.llm_filter
        if self.command_middleware is not None:
            kwargs["command_middleware"] = self.command_middleware
        if self._secret_shield is not None:
            kwargs["secret_shield"] = self._secret_shield
        if self.event_callback is not None:
            kwargs["event_callback"] = self.event_callback
        if self.cancel_flag is not None:
            kwargs["cancel_flag"] = self.cancel_flag
        if self.subagents:
            from .subagent import SubagentManager, SA_TEMPLATE_EXCLUDE
            from .a2a_types import EVENT_STATUS_UPDATE
            from . import fmt

            sa_template = {
                k: v for k, v in kwargs.items() if k not in SA_TEMPLATE_EXCLUDE
            }
            event_cb = self.event_callback
            notify = (
                (lambda msg: event_cb(EVENT_STATUS_UPDATE, {"text": msg}))
                if event_cb is not None
                else fmt.info
            )
            kwargs["subagent_manager"] = SubagentManager(
                loop_kwargs_template=sa_template,
                tools=self._tools,
                resolved_system_content=state.get("resolved_system_content"),
                parent_cancel_flag=self.cancel_flag,
                verbose=self.verbose,
                notify_user=notify,
                proactive_summaries=self.proactive_summaries,
            )
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

        answer = None
        exhausted = False
        outcome = "error"
        exit_code = 1
        _subagent_mgr = loop_kwargs.get("subagent_manager")
        try:
            answer, exhausted = run_agent_loop(
                messages, self._tools, **loop_kwargs, report=collector
            )
            outcome = "exhausted" if exhausted else "success"
            exit_code = 2 if exhausted else 0

            if self.history and answer:
                append_history(
                    self.base_dir, question, answer, diagnostics=self.verbose
                )
        finally:
            if _subagent_mgr is not None:
                _subagent_mgr.shutdown()
            self._run_lifecycle_exit(outcome=outcome, exit_code=exit_code)
            if self.trace_dir and messages:
                self._write_trace(messages, question)

        report_dict = None
        if collector:
            nono_version = None
            if self.sandbox == "nono":
                from .sandbox_nono import get_nono_version

                nono_version = get_nono_version()
            _gs = state.get("goal_state")
            goal_stats = None
            if _gs is not None:
                payload = _gs.to_report_dict()
                if payload is not None or _gs.created_count > 0:
                    goal_stats = {
                        "created_count": _gs.created_count,
                        "completed_count": _gs.completed_count,
                    }
                    if payload is not None:
                        goal_stats["current"] = payload
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
                    "files": self.files,
                },
                outcome=outcome,
                answer=answer,
                exit_code=exit_code,
                turns=collector.max_turn_seen,
                goal_stats=goal_stats,
                sandbox_mode=self.sandbox,
                sandbox_session=self.sandbox_session,
                sandbox_strict_read=self.sandbox_strict_read,
                nono_version=nono_version,
                nono_profile=self.nono_profile,
                nono_rollback=self.nono_rollback,
                nono_block_net=self.nono_block_net,
                network_mode=self.network,
            )

        return Result(
            answer=answer,
            exhausted=exhausted,
            messages=copy.deepcopy(messages),
            report=report_dict,
        )

    def _make_input_context(self, state: dict) -> "object":
        """Build an InputContext over the conversation state for command dispatch.

        Used by ``ask(parse_commands=True)`` so slash/bang commands run through
        the same executor the REPL and one-shot paths use. The shared
        ``turn_state`` (created in ``_make_per_run_state``) carries the turn
        budget, so the per-call ``max_turns`` copy is dropped from the loop
        kwargs, matching how the REPL wires ``execute_input``.
        """
        from .input_dispatch import InputContext

        loop_kwargs = self._build_loop_kwargs(state)
        # The agent-turn command path supplies max_turns from the shared
        # turn_state, so drop the per-call copy to avoid a duplicate kwarg.
        loop_kwargs.pop("max_turns", None)
        turn_state = state["turn_state"]
        loop_kwargs["turn_state"] = turn_state
        return InputContext(
            messages=state["messages"],
            tools=self._tools,
            base_dir=self.base_dir,
            turn_state=turn_state,
            interactive=False,
            thinking_state=state["thinking_state"],
            todo_state=state["todo_state"],
            snapshot_state=state["snapshot_state"],
            goal_state=state["goal_state"],
            file_tracker=state["file_tracker"],
            no_history=not self.history,
            continue_here=self.continue_here,
            verbose=self.verbose,
            loop_kwargs=loop_kwargs,
            mcp_manager=self._mcp_manager,
            a2a_manager=self._a2a_manager,
            subagent_manager=loop_kwargs.get("subagent_manager"),
            extra_write_roots=self._allowed_dir_paths,
            skill_read_roots=state["skill_read_roots"],
            skills_catalog=self._skills_catalog,
            trace_dir=self.trace_dir,
        )

    @staticmethod
    @contextmanager
    def _transcript_rollback(messages: list) -> "Iterator[None]":
        """Restore *messages* to its entry state if the body raises.

        Keeps a session usable after a failed turn by reverting in-place
        mutations (appends, compaction, system-prompt truncation). The shallow
        per-message copy mirrors the message objects, not their nested content.
        """
        snapshot = [copy.copy(m) for m in messages]
        try:
            yield
        except BaseException:
            messages[:] = snapshot
            raise

    def _run_command(self, parsed: "object", state: dict) -> Result:
        """Dispatch a parsed slash/bang command and adapt the StepResult."""
        from .agent import execute_input

        ctx = self._make_input_context(state)
        messages = state["messages"]
        # Match ask()'s failure contract: propagate (e.g. AgentError to ACP) but
        # leave the transcript usable.
        with self._transcript_rollback(messages):
            try:
                step = execute_input(parsed, ctx, mode="repl")
            finally:
                if ctx.subagent_manager is not None:
                    ctx.subagent_manager.shutdown()

        # Agent-turn commands stream their answer through event_callback while
        # the loop runs. Info and state-change commands produce text only here,
        # so surface it to streaming clients (e.g. ACP) as a final chunk.
        if (
            step.text is not None
            and step.kind in ("info", "state_change")
            and self.event_callback is not None
        ):
            from .a2a_types import EVENT_TEXT_CHUNK

            self.event_callback(EVENT_TEXT_CHUNK, {"text": step.text})

        if self.trace_dir and messages:
            if self._trace_session_id is None:
                import uuid

                self._trace_session_id = str(uuid.uuid4())
            self._write_trace(messages, parsed.raw, session_id=self._trace_session_id)

        return Result(
            answer=step.text,
            exhausted=step.exhausted,
            messages=copy.deepcopy(messages),
            report=None,
        )

    def ask(self, question: str, *, parse_commands: bool = False) -> Result:
        """Conversational: share context across questions (like the REPL).

        On success, the assistant's reply is appended to the shared message
        history so subsequent calls build on prior context.

        When ``parse_commands`` is true, an input that begins with a slash
        command (``/help``), a custom bang command (``!name``), or the quick
        shell prefix (``!! cmd``) is routed through the shared command executor
        instead of being sent to the model verbatim. This is what lets editors
        speaking ACP use the same commands as the REPL. Plain prompts are
        unaffected and follow the normal agent-loop path below.

        On failure (any exception from the agent loop), the message list is
        rolled back to its exact state before this call — including reverting
        any in-place mutations from compaction or system-prompt truncation —
        so the session remains usable.  State objects (thinking, todo,
        snapshots, file tracker) are **not** rolled back — partial progress
        from the failed turn (e.g. files already read, thinking notes) is
        intentionally preserved.

        Raises:
            AgentError: on LLM, tool, or infrastructure failures.
            ContextOverflowError: (subclass of AgentError) when the context
                window is exhausted even after all compaction strategies.
            LifecycleError: (subclass of AgentError) when a fail-closed
                startup hook fails during the first call's setup.
        """
        self._setup()

        from .agent import run_agent_loop, append_history

        if self._conv_state is None:
            system_content = self._system_with_memory(question, policy="interactive")
            self._conv_state = self._make_per_run_state(system_content=system_content)

        state = self._conv_state

        if parse_commands:
            from .input_dispatch import parse_input_line

            parsed = parse_input_line(question)
            if parsed.is_command or parsed.is_custom_command:
                return self._run_command(parsed, state)

        messages = state["messages"]
        with self._transcript_rollback(messages):
            messages.append({"role": "user", "content": question})

            loop_kwargs = self._build_loop_kwargs(state)
            _subagent_mgr = loop_kwargs.get("subagent_manager")
            # Honor a turn budget extended via a /extend command (kept in the
            # shared turn_state). Set after _build_loop_kwargs so subagents,
            # whose template is captured inside it, keep their own budget.
            turn_state = state["turn_state"]
            loop_kwargs["max_turns"] = turn_state["max_turns"]
            loop_kwargs["turn_state"] = turn_state

            try:
                answer, exhausted = run_agent_loop(messages, self._tools, **loop_kwargs)
            finally:
                if _subagent_mgr is not None:
                    _subagent_mgr.shutdown()

        if self.history and answer:
            append_history(self.base_dir, question, answer, diagnostics=self.verbose)

        if self.trace_dir and messages:
            if self._trace_session_id is None:
                import uuid

                self._trace_session_id = str(uuid.uuid4())
            self._write_trace(messages, question, session_id=self._trace_session_id)

        return Result(
            answer=answer,
            exhausted=exhausted,
            messages=copy.deepcopy(messages),
            report=None,
        )

    def _write_trace(
        self,
        messages: list,
        task: str | None = None,
        *,
        session_id: str | None = None,
    ) -> None:
        from .traces import write_trace_to_dir

        write_trace_to_dir(
            messages,
            trace_dir=self.trace_dir,
            base_dir=self.base_dir,
            model=self._model_id or "unknown",
            task=task,
            session_id=session_id,
            verbose=self.verbose,
            secret_shield=self._secret_shield,
        )

    def _run_lifecycle_exit(
        self,
        *,
        outcome: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Run the lifecycle exit hook if configured and not already run."""
        if self._lifecycle_exit_ran:
            return
        if not (self.lifecycle_command and self.lifecycle_enabled and self._setup_done):
            return

        self._lifecycle_exit_ran = True

        from .lifecycle import run_lifecycle_hook

        run_lifecycle_hook(
            self.lifecycle_command,
            "exit",
            self.base_dir,
            timeout=self.lifecycle_timeout,
            fail_closed=self.lifecycle_fail_closed,
            provider=self.provider,
            model=self._model_id,
            git_meta=self._lifecycle_git_meta,
            outcome=outcome,
            exit_code=exit_code,
            verbose=self.verbose,
        )

    def close(self, *, outcome: str | None = None, exit_code: int | None = None):
        """Explicitly close the session and run the exit hook.

        Call this after the last ask() call when not using a context manager.
        Passing *outcome* and *exit_code* makes them available to the hook via
        SWIVAL_OUTCOME and SWIVAL_EXIT_CODE. Idempotent — safe to call after
        run() already ran the exit hook.

        Raises LifecycleError if the exit hook fails and lifecycle_fail_closed
        is True. Resources are always cleaned up regardless.
        """
        try:
            self._run_lifecycle_exit(outcome=outcome, exit_code=exit_code)
        finally:
            self._cleanup()

    def _cleanup(self):
        """Release resources (cache, MCP, A2A, secrets)."""
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always clean up resources first
        _hook_err = None
        try:
            self._run_lifecycle_exit()
        except Exception as e:
            _hook_err = e
        self._cleanup()

        # Propagate fail-closed exit hook errors when no other exception is active
        if _hook_err is not None and self.lifecycle_fail_closed and exc_type is None:
            raise _hook_err

    def reset(self) -> None:
        """Clear conversation state without invalidating setup. Next ask() starts fresh."""
        self._conv_state = None
