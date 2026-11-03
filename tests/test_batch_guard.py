import json

import pytest

from swival import agent
from swival._msg import _msg_content, _msg_role
from swival.report import ReportCollector
from swival.snapshot import SnapshotState
from swival.tracker import FileAccessTracker
from tests.conftest import ScriptedLLM, build_loop_kwargs, make_message, make_tool_call


def call(name, call_id, **arguments):
    return make_tool_call(name, json.dumps(arguments), call_id)


@pytest.fixture(params=["native", "command"])
def run_batch(request, tmp_path, monkeypatch):
    report = ReportCollector()
    snapshot = SnapshotState()

    def run(batches):
        messages = [{"role": "user", "content": "Apply the changes."}]
        tracker = FileAccessTracker()
        if request.param == "native":
            llm = ScriptedLLM(
                [(make_message(tool_calls=batch), "tool_calls") for batch in batches]
            )
            monkeypatch.setattr(agent, "call_llm", llm)
            kwargs = build_loop_kwargs(tmp_path, max_turns=len(batches) + 2)
            kwargs.update(
                file_tracker=tracker,
                commands_unrestricted=True,
                shell_allowed=True,
                report=report,
                snapshot_state=snapshot,
            )
            agent.run_agent_loop(messages, [], **kwargs)
        else:
            responses = iter(batches)

            def command_round(*args, **kwargs):
                messages[:] = args[4]
                batch = next(responses, None)
                if batch is None:
                    return "Done."
                return "\n".join(
                    f'<swival:call id="{tc.id}" name="{tc.function.name}">'
                    f"{tc.function.arguments}</swival:call>"
                    for tc in batch
                )

            monkeypatch.setattr(agent, "_run_command_round", command_round)
            _, _, run.activity = agent._call_command_with_tools(
                "fake-command",
                messages,
                {
                    "base_dir": str(tmp_path),
                    "thinking_state": None,
                    "verbose": False,
                    "file_tracker": tracker,
                    "commands_unrestricted": True,
                    "shell_allowed": True,
                },
                outer_turn=1,
                outer_turn_offset=0,
                report=report,
                snapshot_state=snapshot,
                verbose=False,
                _emit=lambda *args: None,
            )
        return {
            message["tool_call_id"]: _msg_content(message)
            for message in messages
            if _msg_role(message) == "tool"
        }

    run.mode = request.param
    run.report = report
    run.snapshot = snapshot
    return run


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("write_file", {"content": "changed"}),
        ("edit_file", {"old_string": "original", "new_string": "changed"}),
        ("delete_file", {}),
        ("create_file", {"content": "changed"}),
    ],
)
def test_failed_mutation_skips_later_calls_except_reads(
    run_batch, tmp_path, name, arguments
):
    (tmp_path / "existing.txt").write_text("original")
    results = run_batch(
        [
            [
                call(name, "failed", file_path="existing.txt", **arguments),
                call("run_shell_command", "shell", command="touch executed"),
                call("read_file", "read", file_path="existing.txt"),
                call("write_file", "write", file_path="new.txt", content="new"),
            ]
        ]
    )
    assert set(results) == {"failed", "shell", "read", "write"}
    assert results["failed"].startswith("error:")
    assert results["shell"].startswith("error: skipped")
    assert results["write"].startswith("error: skipped")
    assert "original" in results["read"]
    assert not (tmp_path / "executed").exists()
    assert not (tmp_path / "new.txt").exists()


def test_failed_read_skips_nothing(run_batch, tmp_path):
    results = run_batch(
        [
            [
                call("read_file", "read", file_path="missing.txt"),
                call("run_shell_command", "shell", command="touch executed"),
            ]
        ]
    )
    assert results["read"].startswith("error:")
    assert (tmp_path / "executed").exists()


def test_next_response_can_recover(run_batch, tmp_path):
    (tmp_path / "existing.txt").write_text("original")
    results = run_batch(
        [
            [
                call(
                    "write_file", "failed", file_path="existing.txt", content="changed"
                ),
                call("run_shell_command", "skipped", command="touch executed"),
            ],
            [call("read_file", "read", file_path="existing.txt")],
            [
                call(
                    "write_file", "retry", file_path="existing.txt", content="changed"
                ),
                call("run_shell_command", "shell", command="touch executed"),
            ],
        ]
    )
    assert results["skipped"].startswith("error: skipped")
    assert not results["retry"].startswith("error:")
    assert not results["shell"].startswith("error:")
    assert (tmp_path / "existing.txt").read_text() == "changed"
    assert (tmp_path / "executed").exists()


def test_skipped_calls_are_not_counted_as_failures(run_batch, tmp_path):
    (tmp_path / "existing.txt").write_text("original")
    paths = ["one.txt", "two.txt", "three.txt"]
    results = run_batch(
        [
            [
                call(
                    "write_file", "failed", file_path="existing.txt", content="changed"
                ),
                *[
                    call("write_file", f"skipped-{i}", file_path=path, content=path)
                    for i, path in enumerate(paths)
                ],
                call("run_shell_command", "skipped-shell", command="touch executed"),
            ],
            [
                call("write_file", f"retry-{i}", file_path=path, content=path)
                for i, path in enumerate(paths)
            ],
        ]
    )
    for i in range(len(paths)):
        assert results[f"skipped-{i}"].startswith("error: skipped")
        assert not results[f"retry-{i}"].startswith("error:")

    report = run_batch.report
    assert report.guardrail_interventions == 0
    assert report.tool_stats == {"write_file": {"succeeded": 3, "failed": 1}}
    assert report.skipped_calls == 4
    assert {
        "turn": 1,
        "type": "skipped_call",
        "name": "run_shell_command",
        "after": "write_file",
    } in report.events
    assert run_batch.snapshot.dirty_tools == {"write_file"}
    if run_batch.mode == "command":
        assert run_batch.activity == [
            {"name": "write_file", "succeeded": ok} for ok in (False, True, True, True)
        ]
