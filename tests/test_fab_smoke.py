"""Smoke tests for the simulation engine.

These are deliberately light: they check that the engine runs to
completion without deadlocking or raising. They do NOT assert that any
policy is better than any other -- no performance claims are made.
"""

import os
from fab import Fab, FabConfig
import simpy
from scheduling import Lot
import statistics


def _run(policy, **kw):
    cfg = FabConfig(
        num_lots=kw.get("num_lots", 20),
        run_length=kw.get("run_length", 500.0),
        policy=policy,
        seed=kw.get("seed", 1),
    )
    return Fab(cfg).run()


def test_engine_completes_for_each_policy():
    for policy in ["fifo", "priority", "edd", "srpt"]:
        result = _run(policy)
        assert result["policy"] == policy
        assert result["throughput"] > 0, "no lot completed the route"
        assert result["avg_cycle_time"] > 0.0


def test_policy_is_actually_consulted():
    """Two different policies on the same seed should not give
    identical results unless the dispatcher is being ignored.

    Congested parameters are used on purpose: with light loading the
    backlog rarely holds more than one lot, so every policy picks the
    only candidate and trivially agrees.
    """
    a = _run("fifo", seed=7, num_lots=200, run_length=200.0)
    b = _run("priority", seed=7, num_lots=200, run_length=200.0)
    print("FIFO    ", a["throughput"], round(a["avg_cycle_time"], 3), a["wip"])
    print("PRIORITY", b["throughput"], round(b["avg_cycle_time"], 3), b["wip"])
    diff = (a["throughput"] != b["throughput"] or a["avg_cycle_time"] != b["avg_cycle_time"])
    assert diff, f"policies identical: throughput={a['throughput']} cycle={a['avg_cycle_time']}"


def test_metrics_have_expected_keys():
    result = _run("fifo")
    for key in ["policy", "throughput", "avg_cycle_time", "wip", "ontime_rate", "utilization"]:
        assert key in result

def test_two_tool_stage_processes_lots_concurrently():
    """Two lots on a 2-tool stage must overlap in time.

    If processing is serial, the last finish time equals the summed
    processing time (busy_time). If concurrent, it equals the longer of
    the two, which is strictly smaller.
    """
    fab = Fab(FabConfig(
        policy="fifo", route=("wet_etch",), downtime_scale=1000.0, seed=1,
    ))
    finish_times = []
    dones = []
    for i in range(2):
        lot = Lot(lot_id=i, arrival_time=0.0)
        done = fab.env.event()
        done.callbacks.append(lambda _e: finish_times.append(fab.env.now))
        fab.backlogs["wet_etch"].append((lot, done))
        fab.wakeups["wet_etch"].put(1)
        dones.append(done)
    fab.env.run(until=simpy.AllOf(fab.env, dones))
    group = fab.groups["wet_etch"]
    assert len(finish_times) == 2
    assert max(finish_times) < group.busy_time

def test_same_seed_gives_same_lot_work_across_policies():
    """Common random numbers: policies must see identical arrivals and work."""
    a = Fab(FabConfig(num_lots=30, run_length=300.0, policy="fifo", seed=3))
    b = Fab(FabConfig(num_lots=30, run_length=300.0, policy="priority", seed=3))
    a.run()
    b.run()
    assert a.stage_time == b.stage_time
    assert [l.arrival_time for l in a.all_lots] == [l.arrival_time for l in b.all_lots]


def test_srpt_has_finite_remaining_time_at_first_stage():
    fab = Fab(FabConfig(num_lots=10, run_length=200.0, policy="srpt", seed=5))
    fab.run()
    assert fab.all_lots
    assert all(l.remaining_time < float("inf") for l in fab.all_lots)

def test_due_mode_does_not_change_work_or_arrivals():
    """Random due dates must not disturb arrivals or lot work content."""
    kw = dict(num_lots=40, run_length=400.0, seed=3, due_factor=5.0)
    a = Fab(FabConfig(due_mode="work", **kw))
    b = Fab(FabConfig(due_mode="random", **kw))
    a.run()
    b.run()
    assert a.stage_time == b.stage_time
    assert [l.arrival_time for l in a.all_lots] == [l.arrival_time for l in b.all_lots]
    assert [l.due_time for l in a.all_lots] != [l.due_time for l in b.all_lots]


def test_random_due_slack_is_independent_of_work():
    def slack_work_corr(mode):
        fab = Fab(FabConfig(
            num_lots=300, run_length=1000.0, seed=11, due_factor=5.0,
            due_mode=mode, priority_due_mult=1.0,
        ))
        fab.run()
        route = fab.config.route
        slack = [l.due_time - l.arrival_time for l in fab.all_lots]
        work = [sum(fab.stage_time[(l.lot_id, n)] for n in route) for l in fab.all_lots]
        return statistics.correlation(slack, work)

    assert slack_work_corr("work") > 0.99
    assert abs(slack_work_corr("random")) < 0.25