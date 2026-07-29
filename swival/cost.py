"""Session cost accounting: a best-effort cumulative USD subtotal.

Costs come from LiteLLM pricing metadata, so the subtotal is an estimate,
never an invoice. Values accumulate unrounded; only the formatter rounds.
"""

import threading
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class CostObservation:
    """Outcome of pricing one LLM call.

    ``known`` carries a finite, nonnegative USD value (a meaningful exact
    zero included). ``unavailable`` is a successful remote call with no
    trustworthy price. ``not_applicable`` is a local, cached, command, or
    subscription-backed result that neither adds dollars nor makes a known
    subtotal partial.
    """

    status: Literal["known", "unavailable", "not_applicable"]
    usd: float | None = None


@dataclass(frozen=True)
class CostSnapshot:
    known_usd: float
    priced_calls: int
    unpriced_calls: int


@dataclass
class SessionCost:
    """Locked accumulator shared across a process session.

    Subagents record from worker threads while the parent renders, so both
    methods hold the lock. Lifetime boundaries (new CLI process, fresh
    ``Session.run()``, ``Session.reset()``) create a fresh instance; there
    is deliberately no ``reset()`` method.
    """

    known_usd: float = 0.0
    priced_calls: int = 0
    unpriced_calls: int = 0
    # Last line actually rendered, letting reconciliation skip duplicates
    # across loop invocations. Owned by the single rendering thread, so it
    # stays outside the lock.
    last_rendered: str | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, observation: CostObservation) -> None:
        with self._lock:
            if observation.status == "known":
                self.known_usd += observation.usd
                self.priced_calls += 1
            elif observation.status == "unavailable":
                self.unpriced_calls += 1

    def snapshot(self) -> CostSnapshot:
        with self._lock:
            return CostSnapshot(
                known_usd=self.known_usd,
                priced_calls=self.priced_calls,
                unpriced_calls=self.unpriced_calls,
            )
