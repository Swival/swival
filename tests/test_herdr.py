import io
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console
from rich.file_proxy import FileProxy

from swival import agent, command_policy, herdr
from swival._env import child_env
from swival.repl_ui import _StreamProxy

# conftest stubs this out, so keep a handle on the real one.
from swival.herdr import _status_terminal

WORKING = "\x1b]7501;state=working:app=swival\x1b\\"
IDLE = "\x1b]7501;state=idle:app=swival\x1b\\"
BLOCKED = "\x1b]7501;state=blocked:app=swival:kind=permission\x1b\\"
CLEAR = "\x1b]7501;state=clear\x1b\\"
RESTORE_TITLE = "\x1b]0;\x07\x1b[23;0t"
TTY = "/dev/ttys001"


@pytest.fixture(autouse=True)
def register(monkeypatch):
    register = Mock()
    monkeypatch.setattr(herdr.atexit, "register", register)
    return register


@pytest.fixture
def reporter(monkeypatch, register):
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_BIN_PATH", "/herdr/bin/herdr")
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p2")
    run = Mock()
    monkeypatch.setattr(herdr.subprocess, "run", run)
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


@pytest.mark.parametrize(
    "owner, reports",
    [("1@w1:p2", False), ("1@w1:p3", True), (f"{os.getpid()}@w1:p2", True)],
)
def test_inherited_owner_marker(reporter, monkeypatch, owner, reports):
    activity, run, _ = reporter
    monkeypatch.setenv(herdr.OWNER_ENV, owner)
    with activity.turn():
        pass
    activity.release()
    assert states(run) == (["working", "idle", "release"] if reports else [])


@pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell")
def test_nested_process_leaves_pane_to_parent(tmp_path, monkeypatch):
    log = tmp_path / "herdr.log"
    fake = tmp_path / "herdr"
    fake.write_text(f'#!/bin/sh\necho "$2 ${{11}}" >> "{log}"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_BIN_PATH", str(fake))
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p2")
    activity = herdr.HerdrActivity()
    nested = "from swival.herdr import activity\nwith activity.turn():\n    pass\n"
    with activity.turn():
        subprocess.run([sys.executable, "-c", nested], env=child_env(), check=True)
        assert log.read_text().split("\n") == ["report-agent working", ""]
    activity.release()
    assert log.read_text().split("\n") == [
        "report-agent working",
        "report-agent idle",
        "release-agent ",
        "",
    ]


@pytest.fixture
def make_terminal(monkeypatch, tmp_path):
    """Point Swival at a file that stands in for a terminal, and return its path."""
    fds = []

    def make(name=TTY):
        path = tmp_path / f"tty{len(fds)}"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT)
        fds.append(fd)
        monkeypatch.setattr(herdr, "_status_terminal", lambda: (fd, name))
        return path

    yield make
    for fd in fds:
        os.close(fd)


@pytest.fixture
def terminal(make_terminal):
    return make_terminal()


def test_terminal_reports_without_herdr_pane(terminal, monkeypatch, register):
    run = Mock()
    monkeypatch.setattr(herdr.subprocess, "run", run)
    monkeypatch.setattr("builtins.input", lambda: "y")
    activity = herdr.HerdrActivity()
    monkeypatch.setattr(herdr, "activity", activity)
    with activity.turn():
        assert command_policy.prompt_approval("git push") == "allow"
    register.assert_called_once_with(activity.release)
    activity.release()
    activity.release()
    run.assert_not_called()
    assert terminal.read_text() == WORKING + BLOCKED + WORKING + IDLE + CLEAR


def test_pane_and_terminal_report_together(reporter, terminal):
    activity, run, _ = reporter
    with activity.turn():
        pass
    activity.release()
    assert states(run) == ["working", "idle", "release"]
    assert terminal.read_text() == WORKING + IDLE + CLEAR


def test_terminal_write_failure_does_not_interrupt_turn(monkeypatch, tmp_path):
    (tmp_path / "tty").touch()
    fd = os.open(tmp_path / "tty", os.O_RDONLY)
    monkeypatch.setattr(herdr, "_status_terminal", lambda: (fd, TTY))
    activity = herdr.HerdrActivity()
    with activity.turn(title="Fix the build"):
        pass
    activity.release()
    os.close(fd)


@pytest.mark.parametrize(
    "owner, reports", [(f"1@{TTY}", False), ("1@/dev/ttys002", True)]
)
def test_inherited_terminal_owner_marker(terminal, monkeypatch, owner, reports):
    monkeypatch.setenv(herdr.TERMINAL_OWNER_ENV, owner)
    activity = herdr.HerdrActivity()
    with activity.turn():
        pass
    activity.release()
    assert terminal.read_text() == (WORKING + IDLE + CLEAR if reports else "")


@pytest.fixture
def fds():
    """A pseudo-terminal and a pipe, the two kinds of file stderr can be."""
    master, tty = os.openpty()
    read, pipe = os.pipe()
    yield tty, pipe
    for fd in (master, tty, read, pipe):
        os.close(fd)


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pseudo-terminal")
@pytest.mark.parametrize(
    "make, found",
    [
        (lambda tty, pipe: tty, True),
        (lambda tty, pipe: FileProxy(Console(file=io.StringIO()), tty), True),
        (lambda tty, pipe: _StreamProxy(Mock(), "stderr", tty), True),
        (lambda tty, pipe: pipe, False),
        (lambda tty, pipe: io.StringIO(), False),
        (lambda tty, pipe: None, False),
    ],
    ids=["tty", "rich", "repl", "pipe", "text", "none"],
)
def test_status_terminal_looks_behind_stderr_wrappers(monkeypatch, fds, make, found):
    tty, pipe = (SimpleNamespace(fileno=lambda fd=fd: fd) for fd in fds)
    monkeypatch.setattr(sys, "stderr", make(tty, pipe))
    monkeypatch.delenv("VTE_VERSION", raising=False)
    assert _status_terminal() == ((fds[0], os.ttyname(fds[0])) if found else None)


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pseudo-terminal")
@pytest.mark.parametrize(
    "platform, vte, found",
    [
        ("darwin", None, True),
        ("linux", "7600", True),
        ("linux", "5302", False),
        ("win32", None, False),
    ],
)
def test_status_terminal_skips_terminals_that_show_garbage(
    monkeypatch, fds, platform, vte, found
):
    monkeypatch.setattr(sys, "stderr", SimpleNamespace(fileno=lambda: fds[0]))
    monkeypatch.setattr(sys, "platform", platform)
    if vte is None:
        monkeypatch.delenv("VTE_VERSION", raising=False)
    else:
        monkeypatch.setenv("VTE_VERSION", vte)
    assert _status_terminal() == ((fds[0], os.ttyname(fds[0])) if found else None)


@pytest.mark.parametrize(
    "text, title",
    [
        ("Fix the build\nThen run the tests", "Fix the build"),
        ("\n \n Second\t  line ", "Second line"),
        ("a\x1b]0;evil\x07b", "a]0;evilb"),
        ("Pro‮j​ect", "Project"),
        ("x" * 60, "x" * 49 + "…"),
        ("\x1b\x07", ""),
        ('<hc from="t1">Explain the build</hc>', "Explain the build"),
        ("<T> generics fail", "<T> generics fail"),
    ],
)
def test_title_text(text, title):
    assert herdr._title_text(text) == title


def test_turn_titles_terminal_and_restores_it(terminal):
    activity = herdr.HerdrActivity()
    with activity.turn(title="Fix the build"):
        pass
    with activity.turn(title="Run the tests"):
        pass
    activity.release()
    assert terminal.read_text() == (
        WORKING
        + "\x1b[22;0t\x1b]0;Fix the build\x07"
        + IDLE
        + WORKING
        + "\x1b]0;Run the tests\x07"
        + IDLE
        + CLEAR
        + RESTORE_TITLE
    )


def test_agent_loop_titles_terminal_with_real_prompt(terminal, monkeypatch):
    activity = herdr.HerdrActivity()
    monkeypatch.setattr(agent, "herdr_activity", activity)
    monkeypatch.setattr(agent, "_run_agent_loop", Mock(return_value=("done", False)))
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Fix the build\nDetails"},
        {"role": "user", "content": "nudge", "_swival_synthetic": True},
    ]
    agent.run_agent_loop(messages, [])
    assert "\x1b]0;Fix the build\x07" in terminal.read_text()


def test_release_cleans_the_terminal_that_was_used(terminal, make_terminal):
    activity = herdr.HerdrActivity()
    with activity.turn(title="Fix the build"):
        pass
    other = make_terminal("/dev/ttys002")
    activity.release()
    assert terminal.read_text().endswith(CLEAR + RESTORE_TITLE)
    assert other.read_text() == ""
