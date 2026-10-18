"""Unit tests for the scheduling policies.

These functions are pure, so they are the easiest and most valuable
part of the codebase to test. They are also the part an interviewer is most
likely to ask about.
"""

import os

import pytest

from scheduling import (
    Lot,
    available_policies,
    earliest_due_date,
    fifo,
    get_policy,
    priority,
    shortest_remaining_processing_time,
)


def _lot(lot_id, priority=0, arrival=0.0, due=None):
    return Lot(lot_id=lot_id, priority=priority, arrival_time=arrival, due_time=due)


def test_fifo_picks_earliest_arrival():
    lots = [_lot(2, arrival=5.0), _lot(1, arrival=1.0)]
    assert fifo(lots, 10.0).lot_id == 1


def test_fifo_breaks_ties_by_id():
    lots = [_lot(7, arrival=1.0), _lot(3, arrival=1.0)]
    assert fifo(lots, 10.0).lot_id == 3


def test_priority_outranks_arrival():
    lots = [_lot(1, priority=0, arrival=1.0), _lot(2, priority=1, arrival=9.0)]
    assert priority(lots, 10.0).lot_id == 2


def test_priority_breaks_ties_by_arrival():
    lots = [_lot(1, priority=1, arrival=9.0), _lot(2, priority=1, arrival=2.0)]
    assert priority(lots, 10.0).lot_id == 2


def test_edd_picks_earliest_due():
    lots = [_lot(1, due=100.0), _lot(2, due=20.0)]
    assert earliest_due_date(lots, 0.0).lot_id == 2


def test_edd_puts_no_deadline_last():
    lots = [_lot(1, due=None), _lot(2, due=500.0)]
    assert earliest_due_date(lots, 0.0).lot_id == 2


def test_srpt_picks_shortest_remaining():
    lots = [_lot(1), _lot(2)]
    lots[0].remaining_time = 50.0
    lots[1].remaining_time = 2.0
    assert shortest_remaining_processing_time(lots, 0.0).lot_id == 2


def test_srpt_unknown_lot_treated_as_infinity():
    lots = [_lot(1), _lot(2)]
    lots[0].remaining_time = 3.0
    assert lots[1].remaining_time == float("inf")
    assert shortest_remaining_processing_time(lots, 0.0).lot_id == 1



def test_available_policies_contains_expected_names():
    names = set(available_policies())
    assert {"fifo", "priority", "edd", "srpt"}.issubset(names)


def test_unknown_policy_raises_keyerror():
    with pytest.raises(KeyError):
        get_policy("does_not_exist")


def test_policy_module_importable_with_src_on_path():
    assert os.path.exists(os.path.join("src", "scheduling.py"))
