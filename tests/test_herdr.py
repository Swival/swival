import subprocess
from unittest.mock import Mock

import pytest

from swival import agent, command_policy, herdr


@pytest.fixture
def reporter(monkeypatch):
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_BIN_PATH", "/herdr/bin/herdr")
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p2")
    run = Mock()
    register = Mock()
    monkeypatch.setattr(herdr.subprocess, "run", run)
    monkeypatch.setattr(herdr.atexit, "register", register)
    activity = herdr.HerdrActivity()
    monkeypatch.setattr(herdr, "activity", activity)
    monkeypatch.setattr(agent, "herdr_activity", activity)
    return activity, run, register


def states(run):
    return [
        "release" if call.args[0][2] == "release-agent" else call.args[0][-1]
        for call in run.call_args_list
    ]


@pytest.mark.parametrize("missing", ["HERDR_ENV", "HERDR_BIN_PATH", "HERDR_PANE_ID"])
def test_missing_environment_is_inert(reporter, monkeypatch, missing):
    activity, run, _ = reporter
    monkeypatch.delenv(missing)
    with activity.turn():
        pass
    activity.release()
    run.assert_not_called()


def test_protocol_and_exit_release(reporter, monkeypatch):
    activity, run, register = reporter
    monkeypatch.setattr(herdr.time, "time_ns", lambda: 1000)
    with activity.turn():
        with activity.turn():
            pass
        assert states(run) == ["working"]
    register.assert_called_once_with(activity.release)
    register.call_args.args[0]()
    activity.release()
    assert states(run) == ["working", "idle", "release"]
    for seq, call in enumerate(run.call_args_list, 1000):
        argv = call.args[0]
        assert argv[:9] == [
            "/herdr/bin/herdr",
            "pane",
            "release-agent" if seq == 1002 else "report-agent",
            "w1:p2",
            "--source",
            "swival",
            "--agent",
            "swival",
            "--seq",
        ]
        assert argv[9] == str(seq)
        assert call.kwargs["timeout"] == 0.5
        assert call.kwargs["stdout"] == subprocess.DEVNULL
        assert call.kwargs["stderr"] == subprocess.DEVNULL
        assert call.kwargs["stdin"] == subprocess.DEVNULL
        assert call.kwargs["check"] is False


@pytest.mark.parametrize(
    "failure", [FileNotFoundError(), subprocess.TimeoutExpired("herdr", 0.5)]
)
def test_connection_failure_does_not_interrupt_turn(reporter, failure):
    activity, run, _ = reporter
    run.side_effect = failure
    with activity.turn():
        pass
    activity.release()
    assert run.call_count == 3


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_agent_loop_resets_activity_on_failure(reporter, monkeypatch, error):
    _, run, _ = reporter
    monkeypatch.setattr(agent, "_run_agent_loop", Mock(side_effect=error))
    with pytest.raises(error):
        agent.run_agent_loop([], [])
    assert states(run) == ["working", "idle"]


def test_subagent_loop_does_not_report(reporter, monkeypatch):
    _, run, _ = reporter
    monkeypatch.setattr(agent, "_run_agent_loop", Mock(return_value=("done", False)))
    assert agent.run_agent_loop([], [], is_subagent=True) == ("done", False)
    run.assert_not_called()


def test_interrupt_during_initial_report_resets_turn(reporter):
    activity, run, _ = reporter
    run.side_effect = [KeyboardInterrupt(), None]
    with pytest.raises(KeyboardInterrupt):
        with activity.turn():
            pytest.fail("The turn should not begin")
    assert states(run) == ["working", "idle"]


@pytest.mark.parametrize("answer", ["y", KeyboardInterrupt()])
def test_approval_restores_working_state(reporter, monkeypatch, answer):
    activity, run, _ = reporter

    def respond():
        assert states(run)[-1] == "blocked"
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr("builtins.input", respond)
    with activity.turn():
        result = command_policy.prompt_approval("git push")
        assert result == ("deny" if isinstance(answer, BaseException) else "allow")
    assert states(run) == ["working", "blocked", "working", "idle"]
