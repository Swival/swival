"""Tests for swival.cost: session cost accumulation."""

import threading

from swival.cost import CostObservation, CostSnapshot, SessionCost


class TestSessionCost:
    def test_known_values_accumulate_unrounded(self):
        sc = SessionCost()
        sc.record(CostObservation("known", 0.0000001))
        sc.record(CostObservation("known", 0.0000002))
        snap = sc.snapshot()
        assert snap.known_usd == 0.0000001 + 0.0000002
        assert snap.priced_calls == 2
        assert snap.unpriced_calls == 0

    def test_known_zero_counts_as_priced(self):
        sc = SessionCost()
        sc.record(CostObservation("known", 0.0))
        snap = sc.snapshot()
        assert snap.known_usd == 0.0
        assert snap.priced_calls == 1

    def test_unavailable_increments_only_unpriced(self):
        sc = SessionCost()
        sc.record(CostObservation("unavailable"))
        snap = sc.snapshot()
        assert snap.known_usd == 0.0
        assert snap.priced_calls == 0
        assert snap.unpriced_calls == 1

    def test_not_applicable_changes_nothing(self):
        sc = SessionCost()
        sc.record(CostObservation("not_applicable"))
        assert sc.snapshot() == CostSnapshot(0.0, 0, 0)

    def test_concurrent_records_not_lost(self):
        sc = SessionCost()
        per_thread = 200

        def worker():
            for _ in range(per_thread):
                sc.record(CostObservation("known", 0.001))
                sc.record(CostObservation("unavailable"))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = sc.snapshot()
        assert snap.priced_calls == 8 * per_thread
        assert snap.unpriced_calls == 8 * per_thread
        assert abs(snap.known_usd - 8 * per_thread * 0.001) < 1e-9
