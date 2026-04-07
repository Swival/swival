"""Tests for the --oneshot-commands gate in one-shot mode."""

from __future__ import annotations

from swival import agent
from swival.config import _UNSET


class TestParserFlag:
    def test_flag_sets_true(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--oneshot-commands", "task"])
        assert args.oneshot_commands is True

    def test_default_is_unset(self):
        parser = agent.build_parser()
        args = parser.parse_args(["task"])
        assert args.oneshot_commands is _UNSET

    def test_yolo_does_not_imply_oneshot_commands(self):
        parser = agent.build_parser()
        args = parser.parse_args(["--yolo", "task"])
        assert args.oneshot_commands is _UNSET
