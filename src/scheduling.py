"""Scheduling policies for the fab simulator.

Each policy is a callable that, given the lots waiting at a machine group,
returns the lot to dispatch next. Policies are deliberately small and pure so
they are easy to unit-test and to swap for an RL-based agent later.
"""

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class Lot:
    """A wafer lot travelling through the fab."""

    lot_id: int
    priority: int = 0
    arrival_time: float = 0.0
    due_time: float | None = None
    step_idx: int = 0
    start_time: float | None = None
    completion_time: float | None = None
    # Unknown remaining time is infinity, so SRPT ranks such lots last.
    remaining_time: float = float("inf")

    @property
    def id(self) -> int:
        return self.lot_id


Policy = Callable[..., "Lot | None"]

_REGISTRY: dict[str, Policy] = {}


def register_policy(name: str) -> Callable[[Policy], Policy]:
    """Decorator that registers a policy function under a name."""

    def _decorator(fn: Policy) -> Policy:
        _REGISTRY[name] = fn
        return fn

    return _decorator


def get_policy(name: str) -> Policy:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown scheduling policy: {name!r}") from exc


def available_policies() -> list[str]:
    return sorted(_REGISTRY)


@register_policy("fifo")
def fifo(lots: Sequence[Lot], now: float) -> Lot | None:
    """First-in-first-out by arrival time."""
    if not lots:
        return None
    return min(lots, key=lambda lot: (lot.arrival_time, lot.lot_id))


@register_policy("priority")
def priority(lots: Sequence[Lot], now: float) -> Lot | None:
    """Highest priority first, FIFO to break ties."""
    if not lots:
        return None
    return min(lots, key=lambda lot: (-lot.priority, lot.arrival_time, lot.lot_id))


@register_policy("edd")
def earliest_due_date(lots: Sequence[Lot], now: float) -> Lot | None:
    """Earliest due date first; lots without a deadline go last."""
    if not lots:
        return None
    return min(
        lots,
        key=lambda lot: (
            lot.due_time if lot.due_time is not None else float("inf"),
            lot.arrival_time,
        ),
    )


@register_policy("srpt")
def shortest_remaining_processing_time(lots: Sequence[Lot], now: float) -> Lot | None:
    """Shortest remaining processing time, read from the lot itself.

    The dispatcher calls every policy as policy(lots, now), so the remaining
    time is carried on each Lot and the signatures stay uniform.
    """
    if not lots:
        return None
    return min(lots, key=lambda lot: (lot.remaining_time, lot.arrival_time))

