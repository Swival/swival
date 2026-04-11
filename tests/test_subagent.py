"""Tests for subagent.py — parallel subagent support."""

import threading
import time

from swival.subagent import (
    SPAWN_SUBAGENT_TOOL,
    CHECK_SUBAGENTS_TOOL,
    SubagentHandle,
    SubagentManager,
    _CompositeCancelFlag,
    _build_subagent_system,
    _subagent_thread_fn,
)
from swival.todo import TodoState


class TestCompositeCancelFlag:
    def test_neither_set(self):
        parent = threading.Event()
        own = threading.Event()
        flag = _CompositeCancelFlag(parent, own)
        assert not flag.is_set()

    def test_own_set(self):
        parent = threading.Event()
        own = threading.Event()
        flag = _CompositeCancelFlag(parent, own)
        own.set()
        assert flag.is_set()

    def test_parent_set(self):
        parent = threading.Event()
        own = threading.Event()
        flag = _CompositeCancelFlag(parent, own)
        parent.set()
        assert flag.is_set()

    def test_no_parent(self):
        own = threading.Event()
        flag = _CompositeCancelFlag(None, own)
        assert not flag.is_set()
        own.set()
        assert flag.is_set()

    def test_set_only_affects_own(self):
        parent = threading.Event()
        own = threading.Event()
        flag = _CompositeCancelFlag(parent, own)
        flag.set()
        assert own.is_set()
        assert not parent.is_set()

    def test_wait_returns_on_set(self):
        own = threading.Event()
        flag = _CompositeCancelFlag(None, own)
        own.set()
        assert flag.wait(timeout=0.1) is True

    def test_wait_timeout(self):
        own = threading.Event()
        flag = _CompositeCancelFlag(None, own)
        result = flag.wait(timeout=0.05)
        assert result is False or not flag.is_set()


class TestBuildSubagentSystem:
    def test_preamble_only(self):
        result = _build_subagent_system(None, None)
        assert "subagent" in result.lower()
        assert "autonomously" in result.lower()

    def test_with_parent_system(self):
        result = _build_subagent_system("You are a Python expert.", None)
        assert "Python expert" in result

    def test_with_system_hint(self):
        result = _build_subagent_system(None, "Focus on security.")
        assert "Focus on security" in result

    def test_ordering(self):
        result = _build_subagent_system("PARENT", "HINT")
        # preamble first, then hint, then parent
        assert result.index("HINT") < result.index("PARENT")


class TestSubagentHandle:
    def test_defaults(self):
        h = SubagentHandle(id="sub_1", task="test task")
        assert h.id == "sub_1"
        assert h.task == "test task"
        assert h.thread is None
        assert h.result is None
        assert h.error is None
        assert h.exhausted is False
        assert not h.done.is_set()
        assert not h.cancel_flag.is_set()


class TestSubagentManager:
    def _make_manager(self, **kwargs):
        defaults = dict(
            loop_kwargs_template={"base_dir": "/tmp/test"},
            tools=[
                SPAWN_SUBAGENT_TOOL,
                CHECK_SUBAGENTS_TOOL,
                {"function": {"name": "read_file"}, "type": "function"},
            ],
            resolved_system_content="test system prompt",
            parent_cancel_flag=None,
            verbose=False,
            notify_user=None,
        )
        defaults.update(kwargs)
        return SubagentManager(**defaults)

    def test_tools_filtered(self):
        mgr = self._make_manager()
        tool_names = [t["function"]["name"] for t in mgr._tools]
        assert "spawn_subagent" not in tool_names
        assert "check_subagents" not in tool_names
        assert "read_file" in tool_names

    def test_poll_empty(self):
        mgr = self._make_manager()
        assert mgr.poll() == "No subagents."

    def test_collect_unknown(self):
        mgr = self._make_manager()
        result = mgr.collect("sub_99")
        assert result.startswith("error:")

    def test_cancel_unknown(self):
        mgr = self._make_manager()
        result = mgr.cancel("sub_99")
        assert result.startswith("error:")

    def test_spawn_and_poll(self, monkeypatch):
        """Spawn a subagent with a mock thread fn that completes immediately."""
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args, **kwargs):
            handle.result = "done"
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        result = mgr.spawn(task="test task")
        assert "sub_1" in result
        assert "ready" in result.lower()
        # Wait for thread to finish
        time.sleep(0.1)
        poll = mgr.poll()
        assert "sub_1" in poll
        assert "done" in poll.lower()

    def test_spawn_and_collect(self, monkeypatch):
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args, **kwargs):
            handle.result = "the answer"
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="test task")
        result = mgr.collect("sub_1", timeout=5)
        assert result == "the answer"

    def test_spawn_and_cancel(self, monkeypatch):
        mgr = self._make_manager()
        barrier = threading.Event()

        def mock_thread_fn(handle, *args, **kwargs):
            barrier.wait(timeout=5)
            if handle.cancel_flag.is_set():
                handle.error = "error: cancelled"
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="long task")
        result = mgr.cancel("sub_1")
        assert "Cancellation" in result
        barrier.set()
        time.sleep(0.1)
        poll = mgr.poll()
        assert "failed" in poll.lower()

    def test_max_concurrent_aborts_on_cancel(self, monkeypatch):
        parent_flag = threading.Event()
        mgr = self._make_manager(parent_cancel_flag=parent_flag)
        barrier = threading.Event()

        def mock_thread_fn(handle, *args):
            barrier.wait(timeout=10)
            handle.done.set()
            args[-1].release()  # slot is last positional arg

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        for i in range(4):
            result = mgr.spawn(task=f"task {i}")
            assert "ready" in result.lower()

        # Set cancel flag so the wait aborts immediately instead of waiting 60s.
        parent_flag.set()
        result = mgr.spawn(task="one too many")
        assert not result.startswith("error:")
        assert "background agents" in result.lower()
        assert "try" in result.lower()
        barrier.set()
        mgr.shutdown(timeout=5)

    def test_shutdown(self, monkeypatch):
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args, **kwargs):
            while not handle.cancel_flag.is_set():
                time.sleep(0.01)
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="task 1")
        mgr.spawn(task="task 2")
        mgr.shutdown(timeout=5)
        # All threads should have exited
        for h in mgr._handles.values():
            assert h.done.is_set()
            assert not h.thread.is_alive()

    def test_fresh_copy(self):
        mgr = self._make_manager()
        copy = mgr.fresh_copy()
        assert copy._template == mgr._template
        assert copy._system_content == mgr._system_content
        assert len(copy._handles) == 0
        assert copy._counter == 0
        assert copy._parent_cancel_flag is not mgr._parent_cancel_flag

    def test_parent_cancel_propagates(self, monkeypatch):
        parent_flag = threading.Event()
        mgr = self._make_manager(parent_cancel_flag=parent_flag)
        seen_cancelled = threading.Event()

        def mock_thread_fn(
            handle,
            template,
            tools,
            task,
            max_turns,
            system_hint,
            system_content,
            composite_cancel,
            slot,
        ):
            while not composite_cancel.is_set():
                time.sleep(0.01)
            seen_cancelled.set()
            handle.done.set()
            slot.release()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="test")
        parent_flag.set()
        assert seen_cancelled.wait(timeout=5)
        mgr.shutdown(timeout=5)

    def test_wait_notifies_user_once(self, monkeypatch):
        """notify_user fires exactly once when all slots are occupied."""
        notifications = []
        mgr = self._make_manager(
            parent_cancel_flag=threading.Event(),
            notify_user=notifications.append,
        )
        barrier = threading.Event()

        def mock_thread_fn(handle, *args):
            barrier.wait(timeout=10)
            handle.done.set()
            args[-1].release()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        for i in range(4):
            mgr.spawn(task=f"task {i}")

        # Abort the wait immediately via cancel flag.
        mgr._parent_cancel_flag.set()
        mgr.spawn(task="overflow")

        assert len(notifications) == 1
        assert "4 background agents" in notifications[0]
        barrier.set()
        mgr.shutdown(timeout=5)

    def test_wait_delayed_success(self, monkeypatch):
        """When a slot frees up during the wait, spawn returns the ready message."""
        mgr = self._make_manager()
        release_first = threading.Event()

        def mock_thread_fn(handle, *args):
            release_first.wait(timeout=10)
            handle.done.set()
            args[-1].release()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        for i in range(4):
            mgr.spawn(task=f"task {i}")

        # Release one slot shortly after the 5th spawn starts waiting.
        def free_one():
            time.sleep(0.1)
            release_first.set()

        threading.Thread(target=free_one, daemon=True).start()

        result = mgr.spawn(task="delayed")
        assert "ready" in result.lower()
        assert "sub_5" in result
        mgr.shutdown(timeout=5)

    def test_semaphore_released_on_completion(self, monkeypatch):
        """Slot is released after a thread finishes so subsequent spawns succeed."""
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args):
            handle.result = "done"
            handle.done.set()
            args[-1].release()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        for i in range(4):
            mgr.spawn(task=f"task {i}")

        for i in range(1, 5):
            mgr.collect(f"sub_{i}", timeout=5)

        result = mgr.spawn(task="after release")
        assert "ready" in result.lower()

    def test_fresh_copy_full_capacity(self, monkeypatch):
        """fresh_copy() starts with all capacity slots available."""
        mgr = self._make_manager()
        barrier = threading.Event()

        def mock_thread_fn(handle, *args):
            barrier.wait(timeout=10)
            handle.done.set()
            args[-1].release()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        for i in range(4):
            mgr.spawn(task=f"task {i}")

        # Original manager is at capacity; fresh copy should have all slots free.
        fresh = mgr.fresh_copy()
        result = fresh.spawn(task="fresh task")
        assert "ready" in result.lower()

        barrier.set()
        mgr.shutdown(timeout=5)
        fresh.shutdown(timeout=5)

    def test_error_captured(self, monkeypatch):
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args, **kwargs):
            try:
                raise RuntimeError("boom")
            except Exception as e:
                handle.error = f"error: subagent crashed: {e}"
            finally:
                handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="failing task")
        result = mgr.collect("sub_1", timeout=5)
        assert "boom" in result


class TestSubagentThreadFn:
    def test_calls_run_agent_loop(self, monkeypatch):
        """Verify the thread fn builds correct state and calls run_agent_loop."""
        captured = {}

        def mock_run_agent_loop(messages, tools, **kwargs):
            captured["messages"] = messages
            captured["tools"] = tools
            captured["kwargs"] = kwargs
            return "the answer", False

        monkeypatch.setattr("swival.agent.run_agent_loop", mock_run_agent_loop)

        handle = SubagentHandle(id="sub_1", task="do something")
        template = {
            "base_dir": "/tmp/test",
            "api_base": "http://localhost",
            "model_id": "test-model",
            "resolved_commands": {},
            "skills_catalog": {},
            "skill_read_roots": [],
            "extra_write_roots": [],
            "files_mode": "some",
            "llm_kwargs": {},
        }
        composite = _CompositeCancelFlag(None, threading.Event())

        _subagent_thread_fn(
            handle,
            template,
            [],
            "do something",
            10,
            None,
            "parent system",
            composite,
            threading.Semaphore(1),
        )

        assert handle.result == "the answer"
        assert handle.exhausted is False
        assert handle.done.is_set()
        assert captured["kwargs"]["verbose"] is False
        assert captured["kwargs"]["continue_here"] is False
        assert captured["kwargs"]["cache"] is None
        assert captured["kwargs"]["cancel_flag"] is composite
        # System message should contain parent system
        sys_msg = captured["messages"][0]
        assert sys_msg["role"] == "system"
        assert "parent system" in sys_msg["content"]
        # User message should be the task
        user_msg = captured["messages"][1]
        assert user_msg["content"] == "do something"

    def test_todo_state_no_persist(self, monkeypatch):
        captured_kwargs = {}

        def mock_run_agent_loop(messages, tools, **kwargs):
            captured_kwargs.update(kwargs)
            return "ok", False

        monkeypatch.setattr("swival.agent.run_agent_loop", mock_run_agent_loop)

        handle = SubagentHandle(id="sub_1", task="test")
        template = {
            "base_dir": "/tmp/test",
            "api_base": "http://localhost",
            "model_id": "test",
            "resolved_commands": {},
            "skills_catalog": {},
            "skill_read_roots": [],
            "extra_write_roots": [],
            "files_mode": "some",
            "llm_kwargs": {},
        }

        _subagent_thread_fn(
            handle,
            template,
            [],
            "test",
            5,
            None,
            None,
            _CompositeCancelFlag(None, threading.Event()),
            threading.Semaphore(1),
        )

        assert isinstance(captured_kwargs["todo_state"], TodoState)

    def test_exception_captured(self, monkeypatch):
        def mock_run_agent_loop(messages, tools, **kwargs):
            raise RuntimeError("test crash")

        monkeypatch.setattr("swival.agent.run_agent_loop", mock_run_agent_loop)

        handle = SubagentHandle(id="sub_1", task="test")
        template = {
            "base_dir": "/tmp/test",
            "api_base": "http://localhost",
            "model_id": "test",
            "resolved_commands": {},
            "skills_catalog": {},
            "skill_read_roots": [],
            "extra_write_roots": [],
            "files_mode": "some",
            "llm_kwargs": {},
        }

        _subagent_thread_fn(
            handle,
            template,
            [],
            "test",
            5,
            None,
            None,
            _CompositeCancelFlag(None, threading.Event()),
            threading.Semaphore(1),
        )

        assert handle.error is not None
        assert "test crash" in handle.error
        assert handle.done.is_set()


class TestCollectEdgeCases:
    def _make_manager(self, **kwargs):
        defaults = dict(
            loop_kwargs_template={"base_dir": "/tmp/test"},
            tools=[{"function": {"name": "read_file"}, "type": "function"}],
            resolved_system_content="test",
            parent_cancel_flag=None,
            verbose=False,
        )
        defaults.update(kwargs)
        return SubagentManager(**defaults)

    def test_collect_cancelled_subagent(self, monkeypatch):
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args, **kwargs):
            handle.cancel_flag.wait(timeout=5)
            # Simulate what _subagent_thread_fn does on cancellation
            handle.result = None
            handle.cancelled = True
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="task")
        mgr.cancel("sub_1")
        result = mgr.collect("sub_1", timeout=5)
        assert "cancelled" in result
        assert result.startswith("error:")

    def test_collect_exhausted_subagent(self, monkeypatch):
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args, **kwargs):
            handle.result = None
            handle.exhausted = True
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="task")
        result = mgr.collect("sub_1", timeout=5)
        assert "exhausted" in result
        assert result.startswith("error:")

    def test_collect_parent_cancelled_subagent(self, monkeypatch):
        """Parent cancellation should also be reported as cancelled."""
        parent_flag = threading.Event()
        mgr = self._make_manager(parent_cancel_flag=parent_flag)

        def mock_run_agent_loop(messages, tools, **kwargs):
            # Simulate run_agent_loop detecting cancellation
            return None, True

        monkeypatch.setattr("swival.agent.run_agent_loop", mock_run_agent_loop)

        # Set parent flag before spawning so it's visible when thread checks
        parent_flag.set()
        mgr.spawn(task="task")
        result = mgr.collect("sub_1", timeout=5)
        assert "cancelled" in result
        assert result.startswith("error:")

    def test_late_cancel_does_not_reclassify(self, monkeypatch):
        """A cancel_all after natural completion should not mark result as cancelled."""
        mgr = self._make_manager()

        def mock_run_agent_loop(messages, tools, **kwargs):
            return "real answer", False

        monkeypatch.setattr("swival.agent.run_agent_loop", mock_run_agent_loop)

        mgr.spawn(task="task")
        # Wait for completion
        mgr.collect("sub_1", timeout=5)
        # Now cancel_all (simulates shutdown after natural completion)
        mgr.cancel_all()
        handle = mgr._handles["sub_1"]
        assert handle.result == "real answer"
        assert not handle.cancelled

    def test_exhausted_no_answer_not_mislabelled(self, monkeypatch):
        """Natural exhaustion with no answer should report exhausted, not cancelled."""
        mgr = self._make_manager()

        def mock_run_agent_loop(messages, tools, **kwargs):
            # Natural max-turn exhaustion, no assistant text produced
            return None, True

        monkeypatch.setattr("swival.agent.run_agent_loop", mock_run_agent_loop)

        mgr.spawn(task="task")
        result = mgr.collect("sub_1", timeout=5)
        assert "exhausted" in result
        assert "cancelled" not in result

    def test_during_loop_cancel_classified_correctly(self, monkeypatch):
        """Cancel during run_agent_loop should be reported as cancelled, not exhausted."""
        mgr = self._make_manager()
        cancel_seen = threading.Event()

        def mock_run_agent_loop(messages, tools, **kwargs):
            cancel_flag = kwargs["cancel_flag"]
            # Wait for the cancel signal, simulating a real loop checking it
            cancel_flag._own.wait(timeout=5)
            cancel_seen.set()
            # Return the same shape as real cancellation in run_agent_loop
            return None, True

        monkeypatch.setattr("swival.agent.run_agent_loop", mock_run_agent_loop)

        mgr.spawn(task="task")
        mgr.cancel("sub_1")
        assert cancel_seen.wait(timeout=5)
        result = mgr.collect("sub_1", timeout=5)
        assert "cancelled" in result
        assert "exhausted" not in result

    def test_collect_successful_with_result(self, monkeypatch):
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args, **kwargs):
            handle.result = "success!"
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="task")
        result = mgr.collect("sub_1", timeout=5)
        assert result == "success!"


class TestFreshCopyLifecycle:
    def _make_manager(self, **kwargs):
        defaults = dict(
            loop_kwargs_template={"base_dir": "/tmp/test"},
            tools=[{"function": {"name": "read_file"}, "type": "function"}],
            resolved_system_content="test",
            parent_cancel_flag=None,
            verbose=False,
        )
        defaults.update(kwargs)
        return SubagentManager(**defaults)

    def test_fresh_copy_after_shutdown_is_clean(self, monkeypatch):
        mgr = self._make_manager()

        def mock_thread_fn(handle, *args, **kwargs):
            while not handle.cancel_flag.is_set():
                time.sleep(0.01)
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        mgr.spawn(task="task 1")
        mgr.shutdown(timeout=5)

        fresh = mgr.fresh_copy()
        assert fresh.poll() == "No subagents."
        assert fresh._counter == 0
        # The fresh manager should be fully functional
        fresh.spawn(task="task 2")
        fresh.collect("sub_1", timeout=5)
        # sub_1 because counter resets
        assert "sub_1" in str(fresh._handles)
        fresh.shutdown(timeout=5)

    def test_outer_finally_shuts_down_fresh_manager(self, monkeypatch):
        """Simulate the REPL pattern: reset creates new manager, outer finally shuts it down."""
        mgr = self._make_manager()
        barrier = threading.Event()

        def mock_thread_fn(handle, *args, **kwargs):
            barrier.wait(timeout=10)
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn)

        # Simulate: spawn on original, then reset (shutdown + fresh_copy)
        mgr.spawn(task="old task")
        mgr.shutdown(timeout=5)
        barrier.set()

        fresh = mgr.fresh_copy()
        barrier2 = threading.Event()

        def mock_thread_fn2(handle, *args, **kwargs):
            barrier2.wait(timeout=10)
            handle.done.set()

        monkeypatch.setattr("swival.subagent._subagent_thread_fn", mock_thread_fn2)

        fresh.spawn(task="new task")
        # The outer finally should shut down `fresh`, not original `mgr`
        fresh.shutdown(timeout=5)
        barrier2.set()
        for h in fresh._handles.values():
            assert h.done.is_set()


class TestToolDefinitions:
    def test_spawn_tool_schema(self):
        assert SPAWN_SUBAGENT_TOOL["function"]["name"] == "spawn_subagent"
        params = SPAWN_SUBAGENT_TOOL["function"]["parameters"]
        assert "task" in params["properties"]
        assert "task" in params["required"]

    def test_check_tool_schema(self):
        assert CHECK_SUBAGENTS_TOOL["function"]["name"] == "check_subagents"
        params = CHECK_SUBAGENTS_TOOL["function"]["parameters"]
        assert "action" in params["properties"]
        assert "poll" in params["properties"]["action"]["enum"]
