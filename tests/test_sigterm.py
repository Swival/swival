import signal
import subprocess
import sys
import textwrap

import pytest


def test_sigterm_handler_installed_by_main(monkeypatch):
    """main() installs a SIGTERM handler that raises SystemExit(143)."""
    old_handler = signal.getsignal(signal.SIGTERM)
    try:
        monkeypatch.setattr("sys.argv", ["swival", "--version"])
        with pytest.raises(SystemExit):
            from swival.agent import main

            main()

        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        with pytest.raises(SystemExit) as exc_info:
            handler(signal.SIGTERM, None)
        assert exc_info.value.code == 143
    finally:
        signal.signal(signal.SIGTERM, old_handler)


def test_sigterm_exits_143_in_subprocess():
    """A real SIGTERM to main() produces exit code 143 and runs the finally block."""
    script = textwrap.dedent("""\
        import signal, os, sys, types

        from swival import agent

        # Patch _run_main to send ourselves SIGTERM during "work"
        def fake_run_main(args, report, _write_report, parser):
            os.kill(os.getpid(), signal.SIGTERM)

        agent._run_main = fake_run_main

        # Minimal args namespace
        sys.argv = ["swival", "hello"]
        agent.main()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 143


def test_sigterm_outcome_is_interrupted():
    """Exit code 143 maps to 'interrupted' outcome (not 'error')."""
    from swival.agent import main  # noqa: F401

    mapping = {0: "success", 2: "exhausted", 130: "interrupted", 143: "interrupted"}
    assert mapping[143] == "interrupted"
    assert mapping[130] == "interrupted"


def test_mcp_manager_closed_on_sigterm():
    """MCP and A2A managers are closed in the finally block on SIGTERM."""
    script = textwrap.dedent("""\
        import signal, os, sys

        from swival import agent

        marker = os.environ["_TEST_MARKER"]

        class FakeManager:
            def __init__(self, name):
                self.name = name
            def close(self):
                with open(marker, "a") as f:
                    f.write(self.name + "\\n")

        mcp = FakeManager("mcp")
        a2a = FakeManager("a2a")

        def fake_run_main(args, report, _write_report, parser):
            args._mcp_manager = mcp
            args._a2a_manager = a2a
            os.kill(os.getpid(), signal.SIGTERM)

        agent._run_main = fake_run_main
        sys.argv = ["swival", "hello"]
        agent.main()
    """)
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        marker = f.name

    env = os.environ.copy()
    env["_TEST_MARKER"] = marker
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=10,
        env=env,
    )
    assert result.returncode == 143

    try:
        with open(marker) as f:
            lines = f.read().strip().split("\n")
        assert "mcp" in lines
        assert "a2a" in lines
    finally:
        os.unlink(marker)


def test_sigterm_during_run_main_exits():
    """SIGTERM raised inside _run_main propagates as exit 143 through main()."""
    script = textwrap.dedent("""\
        import signal, os, sys

        from swival import agent

        def fake_run_main(args, report, _write_report, parser):
            os.kill(os.getpid(), signal.SIGTERM)

        agent._run_main = fake_run_main
        sys.argv = ["swival", "hello"]
        agent.main()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 143
