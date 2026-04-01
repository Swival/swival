"""JSON report generation for benchmarking evaluation."""

import json
from datetime import datetime, timezone


class AgentError(Exception):
    """Raised by the agent loop or setup helpers for reportable runtime failures."""


class ConfigError(AgentError):
    """Raised for invalid configuration (missing model, bad API key, etc.)."""


class ContextOverflowError(AgentError):
    """Raised when the LLM call fails due to context window overflow."""


class LifecycleError(AgentError):
    """Raised when a lifecycle hook fails in fail-closed mode."""


class ReportCollector:
    """Accumulates events during an agent run for JSON report output."""

    def __init__(self):
        self.events: list[dict] = []
        self.tool_stats: dict[str, dict[str, int]] = {}
        self.compactions = 0
        self.turn_drops = 0
        self.guardrail_interventions = 0
        self.truncated_responses = 0
        self.llm_calls = 0
        self.total_llm_time = 0.0
        self.total_cached_tokens = 0
        self.total_cache_write_tokens = 0
        self.total_tool_time = 0.0
        self.max_turn_seen = 0
        self.skills_used: list[str] = []
        self.memory_stats: dict | None = None
        self.lifecycle_events: list[dict] = []
        self._last_report: dict | None = None
        self.security_stats: dict[str, int] = {
            "command_policy_blocks": 0,
            "command_policy_approvals": 0,
            "untrusted_inputs": 0,
        }

    def record_llm_call(
        self,
        turn: int,
        duration: float,
        token_est: int,
        finish_reason: str,
        *,
        is_retry: bool = False,
        retry_reason: str | None = None,
        provider_retries: int = 0,
        cached_tokens: int = 0,
        cache_write_tokens: int = 0,
    ):
        self.llm_calls += 1
        self.total_llm_time += duration
        self.total_cached_tokens += cached_tokens
        self.total_cache_write_tokens += cache_write_tokens
        if turn > self.max_turn_seen:
            self.max_turn_seen = turn
        event = {
            "turn": turn,
            "type": "llm_call",
            "duration_s": round(duration, 3),
            "prompt_tokens_est": token_est,
            "finish_reason": finish_reason,
            "is_retry": is_retry,
        }
        if retry_reason is not None:
            event["retry_reason"] = retry_reason
        if provider_retries:
            event["provider_retries"] = provider_retries
        self.events.append(event)

    def record_tool_call(
        self,
        turn: int,
        name: str,
        arguments: dict | None,
        succeeded: bool,
        duration: float,
        result_length: int,
        error: str | None = None,
        repairs: list[dict] | None = None,
    ):
        self.total_tool_time += duration
        if name == "use_skill" and succeeded and arguments:
            skill_name = arguments.get("name")
            if skill_name and skill_name not in self.skills_used:
                self.skills_used.append(skill_name)
        stats = self.tool_stats.setdefault(name, {"succeeded": 0, "failed": 0})
        stats["succeeded" if succeeded else "failed"] += 1
        if repairs:
            stats["repairs"] = stats.get("repairs", 0) + len(repairs)
        event: dict = {
            "turn": turn,
            "type": "tool_call",
            "name": name,
            "arguments": arguments,
            "succeeded": succeeded,
            "duration_s": round(duration, 3),
            "result_length": result_length,
        }
        if error is not None:
            event["error"] = error
        if repairs:
            event["repairs"] = repairs
        self.events.append(event)

    def record_compaction(
        self, turn: int, strategy: str, tokens_before: int, tokens_after: int
    ):
        if strategy == "drop_middle_turns":
            self.turn_drops += 1
        else:
            self.compactions += 1
        self.events.append(
            {
                "turn": turn,
                "type": "compaction",
                "strategy": strategy,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
            }
        )

    def record_guardrail(self, turn: int, tool: str, level: str):
        self.guardrail_interventions += 1
        self.events.append(
            {"turn": turn, "type": "guardrail", "tool": tool, "level": level}
        )

    def record_truncated_response(self, turn: int):
        self.truncated_responses += 1
        self.events.append({"turn": turn, "type": "truncated_response"})

    def record_memory(
        self,
        *,
        total_entries: int,
        bootstrap_entries: int,
        retrievable_entries: int,
        bootstrap_tokens: int,
        retrieval_tokens: int,
        retrieved_ids: list[str],
        mode: str,
    ):
        self.memory_stats = {
            "total_entries": total_entries,
            "bootstrap_entries": bootstrap_entries,
            "retrievable_entries": retrievable_entries,
            "bootstrap_tokens": bootstrap_tokens,
            "retrieval_tokens": retrieval_tokens,
            "retrieved_ids": retrieved_ids,
            "mode": mode,
        }

    def record_lifecycle(self, hook_result: dict):
        event = {
            "type": "lifecycle",
            "event": hook_result.get("event"),
            "exit_code": hook_result.get("exit_code"),
            "duration_s": round(hook_result.get("duration", 0), 3),
        }
        if hook_result.get("error"):
            event["error"] = hook_result["error"]
        self.lifecycle_events.append(event)
        self.events.append(event)

    def record_review(
        self,
        review_round: int,
        exit_code: int,
        feedback: str,
        *,
        stderr: str = "",
    ):
        event: dict = {
            "type": "review",
            "round": review_round,
            "exit_code": exit_code,
            "feedback": feedback,
        }
        if stderr:
            event["stderr"] = stderr
        self.events.append(event)

    def record_command_policy(self, bucket: str, decision: str):
        """Record a command policy decision."""
        if decision in ("deny", "block"):
            self.security_stats["command_policy_blocks"] += 1
        else:
            self.security_stats["command_policy_approvals"] += 1
        self.events.append(
            {"type": "command_policy", "bucket": bucket, "decision": decision}
        )

    def record_untrusted_input(self, source: str, origin: str = ""):
        """Record ingestion of untrusted external content."""
        self.security_stats["untrusted_inputs"] += 1
        self.events.append(
            {"type": "untrusted_input", "source": source, "origin": origin}
        )

    def build_report(
        self,
        *,
        task: str,
        model: str,
        provider: str,
        settings: dict,
        outcome: str,
        answer: str | None,
        exit_code: int,
        turns: int,
        error_message: str | None = None,
        review_rounds: int = 0,
        todo_stats: dict | None = None,
        snapshot_stats: dict | None = None,
        sandbox_mode: str = "builtin",
        sandbox_session: str | None = None,
        sandbox_strict_read: bool = False,
        agentfs_version: str | None = None,
        diff_hint: str | None = None,
    ) -> dict:
        tool_calls_succeeded = sum(s["succeeded"] for s in self.tool_stats.values())
        tool_calls_failed = sum(s["failed"] for s in self.tool_stats.values())

        result: dict = {
            "outcome": outcome,
            "answer": answer,
            "exit_code": exit_code,
        }
        if error_message is not None:
            result["error_message"] = error_message

        sandbox: dict = {"mode": sandbox_mode}
        if sandbox_session is not None:
            sandbox["session"] = sandbox_session
        if sandbox_mode == "agentfs":
            sandbox["strict_read"] = sandbox_strict_read
            if agentfs_version is not None:
                sandbox["agentfs_version"] = agentfs_version
            if diff_hint is not None:
                sandbox["diff_hint"] = diff_hint

        return {
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "model": model,
            "provider": provider,
            "settings": settings,
            "sandbox": sandbox,
            "result": result,
            "stats": {
                "turns": turns,
                "tool_calls_total": tool_calls_succeeded + tool_calls_failed,
                "tool_calls_succeeded": tool_calls_succeeded,
                "tool_calls_failed": tool_calls_failed,
                "tool_calls_by_name": dict(self.tool_stats),
                "compactions": self.compactions,
                "turn_drops": self.turn_drops,
                "guardrail_interventions": self.guardrail_interventions,
                "truncated_responses": self.truncated_responses,
                "llm_calls": self.llm_calls,
                "total_llm_time_s": round(self.total_llm_time, 3),
                **(
                    {
                        "prompt_cache": {
                            "cached_tokens": self.total_cached_tokens,
                            "cache_write_tokens": self.total_cache_write_tokens,
                        }
                    }
                    if self.total_cached_tokens or self.total_cache_write_tokens
                    else {}
                ),
                "total_tool_time_s": round(self.total_tool_time, 3),
                "skills_used": list(self.skills_used),
                "review_rounds": review_rounds,
                **({"todo": todo_stats} if todo_stats else {}),
                **({"snapshot": snapshot_stats} if snapshot_stats else {}),
                **({"memory": self.memory_stats} if self.memory_stats else {}),
                **(
                    {"lifecycle": self.lifecycle_events}
                    if self.lifecycle_events
                    else {}
                ),
                **(
                    {"security": dict(self.security_stats)}
                    if any(self.security_stats.values())
                    else {}
                ),
            },
            "timeline": self.events,
        }

    def write(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._last_report, f, indent=2)
            f.write("\n")

    def finalize(
        self,
        *,
        task: str,
        model: str,
        provider: str,
        settings: dict,
        outcome: str,
        answer: str | None,
        exit_code: int,
        turns: int,
        error_message: str | None = None,
        review_rounds: int = 0,
        todo_stats: dict | None = None,
        snapshot_stats: dict | None = None,
        sandbox_mode: str = "builtin",
        sandbox_session: str | None = None,
        sandbox_strict_read: bool = False,
        agentfs_version: str | None = None,
        diff_hint: str | None = None,
    ) -> dict:
        """Build the report and write it to disk in one step."""
        self._last_report = self.build_report(
            task=task,
            model=model,
            provider=provider,
            settings=settings,
            outcome=outcome,
            answer=answer,
            exit_code=exit_code,
            turns=turns,
            error_message=error_message,
            review_rounds=review_rounds,
            todo_stats=todo_stats,
            snapshot_stats=snapshot_stats,
            sandbox_mode=sandbox_mode,
            sandbox_session=sandbox_session,
            sandbox_strict_read=sandbox_strict_read,
            agentfs_version=agentfs_version,
            diff_hint=diff_hint,
        )
        return self._last_report
