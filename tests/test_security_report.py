"""Tests for security reporting in swival.report."""

from swival.report import ReportCollector


class TestSecurityStats:
    def test_initial_stats_zero(self):
        r = ReportCollector()
        assert r.security_stats["command_policy_blocks"] == 0
        assert r.security_stats["command_policy_approvals"] == 0
        assert r.security_stats["untrusted_inputs"] == 0

    def test_record_command_policy_deny(self):
        r = ReportCollector()
        r.record_command_policy("rm", "deny")
        assert r.security_stats["command_policy_blocks"] == 1
        assert r.security_stats["command_policy_approvals"] == 0

    def test_record_command_policy_block(self):
        r = ReportCollector()
        r.record_command_policy("rm", "block")
        assert r.security_stats["command_policy_blocks"] == 1

    def test_record_command_policy_allow(self):
        r = ReportCollector()
        r.record_command_policy("ls", "allow")
        assert r.security_stats["command_policy_approvals"] == 1
        assert r.security_stats["command_policy_blocks"] == 0

    def test_record_command_policy_persist(self):
        r = ReportCollector()
        r.record_command_policy("ls", "persist")
        assert r.security_stats["command_policy_approvals"] == 1

    def test_record_command_policy_once(self):
        r = ReportCollector()
        r.record_command_policy("ls", "once")
        assert r.security_stats["command_policy_approvals"] == 1

    def test_record_command_policy_event(self):
        r = ReportCollector()
        r.record_command_policy("git push", "deny")
        assert len(r.events) == 1
        ev = r.events[0]
        assert ev["type"] == "command_policy"
        assert ev["bucket"] == "git push"
        assert ev["decision"] == "deny"

    def test_record_untrusted_input(self):
        r = ReportCollector()
        r.record_untrusted_input("fetch_url", origin="https://example.com")
        assert r.security_stats["untrusted_inputs"] == 1
        ev = r.events[0]
        assert ev["type"] == "untrusted_input"
        assert ev["source"] == "fetch_url"
        assert ev["origin"] == "https://example.com"

    def test_record_untrusted_input_no_origin(self):
        r = ReportCollector()
        r.record_untrusted_input("mcp__server__tool")
        assert r.security_stats["untrusted_inputs"] == 1
        assert r.events[0]["origin"] == ""

    def test_multiple_events(self):
        r = ReportCollector()
        r.record_command_policy("rm", "deny")
        r.record_command_policy("ls", "allow")
        r.record_untrusted_input("fetch_url")
        assert r.security_stats["command_policy_blocks"] == 1
        assert r.security_stats["command_policy_approvals"] == 1
        assert r.security_stats["untrusted_inputs"] == 1
        assert len(r.events) == 3


class TestSecurityInReport:
    def _build(self, collector):
        return collector.build_report(
            task="test",
            model="test-model",
            provider="test",
            settings={},
            outcome="success",
            answer="done",
            exit_code=0,
            turns=1,
        )

    def test_security_absent_when_all_zero(self):
        r = ReportCollector()
        report = self._build(r)
        assert "security" not in report["stats"]

    def test_security_present_when_nonzero(self):
        r = ReportCollector()
        r.record_command_policy("rm", "deny")
        report = self._build(r)
        assert "security" in report["stats"]
        assert report["stats"]["security"]["command_policy_blocks"] == 1

    def test_security_includes_all_fields(self):
        r = ReportCollector()
        r.record_untrusted_input("fetch_url")
        report = self._build(r)
        sec = report["stats"]["security"]
        assert "command_policy_blocks" in sec
        assert "command_policy_approvals" in sec
        assert "untrusted_inputs" in sec
        assert sec["untrusted_inputs"] == 1
