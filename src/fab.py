"""Discrete-event semiconductor fab simulator and experiment runner."""

import argparse
import json
import random
from dataclasses import dataclass

import simpy

from machines import MachineGroup
from maintenance import MaintenancePolicy
from scheduling import Lot, get_policy


@dataclass
class LotRecord:
    lot_id: int
    arrival_time: float
    completion_time: float
    due_time: float
    priority: int

    def cycle_time(self):
        return self.completion_time - self.arrival_time

    def ontime(self):
        if self.due_time is None:
            return True
        return self.completion_time <= self.due_time


@dataclass
class FabConfig:
    num_lots: int = 300
    arrival_mean: float = 2.0
    run_length: float = 400.0
    seed: int = 42
    policy: str = "fifo"
    priority_frac: float = 0.2
    due_slack: float = 50.0
    capacity_multiplier: float = 1.0
    downtime_scale: float = 1.0
    # >1 means more lots: the mean inter-arrival time is arrival_mean / arrival_scale.
    arrival_scale: float = 1.0
    # Stop generating arrivals after this clock time (None = no cutoff).
    arrival_end: float | None = None
    # If set, due = arrival + slack, where slack is built from due_factor
    # (see due_mode). If None, the legacy fixed-slack rule is used.
    due_factor: float | None = None
    # "work":   slack = due_factor x the lot's own total work.
    # "random": slack = due_factor x mean route work x U(0.5, 1.5), drawn
    #           independently of the lot's work (same mean slack).
    due_mode: str = "work"
    # Slack multiplier for priority lots when due_factor is set.
    # 0.5 = priority lots get tighter due dates; 1.0 = no priority signal.
    priority_due_mult: float = 0.5
    # Weibull shape for tool failure. >1 = wear-out; 1.0 = exponential.
    failure_shape: float = 2.5
    # Optional MaintenanceConfig (None = purely reactive repair).
    maintenance: object | None = None
    # Stage two: noisy wear sensor.
    sensor_noise: float = 0.15
    log_sensor: bool = False
    route: tuple = ("wet_etch", "lithography", "deposition", "etch", "inspection")


MACHINE_DEFAULTS = {
    "wet_etch": dict(num_machines=2, process_time_mean=4.0, process_time_cv=0.2, mtbf=80.0, mttr=8.0),
    "lithography": dict(num_machines=1, process_time_mean=6.0, process_time_cv=0.15, mtbf=120.0, mttr=12.0),
    "deposition": dict(num_machines=2, process_time_mean=3.0, process_time_cv=0.1, mtbf=100.0, mttr=6.0),
    "etch": dict(num_machines=2, process_time_mean=3.5, process_time_cv=0.12, mtbf=90.0, mttr=7.0),
    "inspection": dict(num_machines=1, process_time_mean=2.0, process_time_cv=0.05, mtbf=150.0, mttr=5.0),
}


class Fab:
    """A multi-stage fab with pluggable dispatching."""

    def __init__(self, config):
        self.config = config
        self.env = simpy.Environment()
        # Arrival stream: inter-arrival times and priorities only. It never
        # depends on dispatching, so all policies see the same arrivals.
        self.rng = random.Random(config.seed)
        self.groups = {}
        mpol = None
        mc = config.maintenance
        if mc is not None and mc.policy != "none":
            mpol = MaintenancePolicy(
                mc,
                {n: MACHINE_DEFAULTS[n]["mtbf"] * config.downtime_scale
                 for n in config.route},
                config.failure_shape,
            )
        for i, name in enumerate(config.route):
            d = MACHINE_DEFAULTS[name]
            self.groups[name] = MachineGroup(
                name=name,
                env=self.env,
                num_machines=max(1, round(d["num_machines"] * config.capacity_multiplier)),
                process_time_mean=d["process_time_mean"],
                process_time_cv=d["process_time_cv"],
                mtbf=d["mtbf"] * config.downtime_scale,
                mttr=d["mttr"] * config.downtime_scale,
                failure_shape=config.failure_shape,
                seed=config.seed + i,
                maintenance=mpol,
                sensor_noise=config.sensor_noise,
                log_sensor=config.log_sensor,
            )
        self.records = []
        self.arrived = 0
        self.all_lots = []   # every lot that arrived, finished or not
        self.stage_time = {}  # (lot_id, stage name) -> pre-sampled processing time
        self.policy = get_policy(config.policy)
        self.backlogs = {name: [] for name in config.route}
        # Each arrival puts one token. A Store can block repeatedly,
        # unlike a simpy.Event, which stays triggered forever once succeeded.
        self.wakeups = {name: simpy.Store(self.env) for name in config.route}
        self.dispatchers = [
            self.env.process(self._dispatch(name)) for name in config.route
        ]

    def _dispatch(self, name):
        """Dispatch lots for stage *name* according to the active policy.

        The loop waits for a tool to be free *then* chooses from the full
        backlog. Processing happens in a separate process per lot, so this
        loop can immediately seat the next lot on another free tool.
        """
        backlog = self.backlogs[name]
        wakeups = self.wakeups[name]
        group = self.groups[name]
        while True:
            if not backlog:
                yield wakeups.get()  # blocks until a lot arrives
                continue
            store = simpy.Store(self.env, capacity=1)
            self.env.process(group.acquire_any(store))
            tool, req = yield store.get()
            waiting = [lot for lot, _ in backlog]
            chosen = self.policy(waiting, self.env.now) if waiting else None
            if chosen is None:
                tool.release(req)
                continue
            idx = next(
                i for i, (lot, _) in enumerate(backlog) if lot is chosen
            )
            lot, done = backlog.pop(idx)
            self.env.process(self._process_on_tool(group, tool, req, lot, done))

    def _process_on_tool(self, group, tool, req, lot, done):
        """Hold *tool* for one lot's processing time, then release it."""
        try:
            pt = self.stage_time.get((lot.lot_id, group.name))
            if pt is None:  # lot injected without pre-sampled times (tests)
                pt = group.process_time(self.rng)
            group.busy_time += pt
            yield self.env.timeout(pt)
            group.processed += 1
            done.succeed()
        finally:
            tool.release(req)

    def run_lot(self, lot):
        """Send the lot through each stage, waiting for its done event."""
        route = self.config.route
        for k, name in enumerate(route):
            # True remaining processing time (this stage and later ones),
            # so SRPT has real information at every stage.
            lot.remaining_time = sum(
                self.stage_time.get((lot.lot_id, s), 0.0) for s in route[k:]
            )
            done = self.env.event()
            self.backlogs[name].append((lot, done))
            yield self.wakeups[name].put(1)
            yield done
        lot.completion_time = self.env.now
        self.records.append(
            LotRecord(
                lot_id=lot.lot_id,
                arrival_time=lot.arrival_time,
                completion_time=lot.completion_time,
                due_time=lot.due_time,
                priority=lot.priority,
            )
        )

    def arrivals(self):
        cfg = self.config
        mean = cfg.arrival_mean / cfg.arrival_scale
        for i in range(cfg.num_lots):
            yield self.env.timeout(self.rng.expovariate(1.0 / mean))
            if cfg.arrival_end is not None and self.env.now > cfg.arrival_end:
                return
            priority = 1 if self.rng.random() < cfg.priority_frac else 0
            # Per-lot RNG: the lot's work content is the same under every policy.
            lot_rng = random.Random(f"{cfg.seed}-lot-{i}")
            for name in cfg.route:
                self.stage_time[(i, name)] = self.groups[name].process_time(lot_rng)
            work = sum(self.stage_time[(i, n)] for n in cfg.route)
            if cfg.due_factor is not None:
                prio_mult = cfg.priority_due_mult if priority else 1.0
                if cfg.due_mode == "work":
                    slack = cfg.due_factor * work * prio_mult
                elif cfg.due_mode == "random":
                    # Own RNG stream: does not disturb arrivals or work content.
                    due_rng = random.Random(f"{cfg.seed}-due-{i}")
                    mean_work = sum(
                        MACHINE_DEFAULTS[n]["process_time_mean"] for n in cfg.route
                    )
                    slack = (
                        cfg.due_factor * mean_work
                        * due_rng.uniform(0.5, 1.5) * prio_mult
                    )
                else:
                    raise ValueError(f"unknown due_mode: {cfg.due_mode!r}")
            else:
                slack = cfg.due_slack + priority * 20.0
            lot = Lot(
                lot_id=i,
                priority=priority,
                arrival_time=self.env.now,
                due_time=self.env.now + slack,
            )
            self.all_lots.append(lot)
            self.arrived += 1
            self.env.process(self.run_lot(lot))

    def run(self):
        self.env.process(self.arrivals())
        self.env.run(until=self.config.run_length)
        return self.metrics()

    def failure_log(self) -> list[dict]:
        """All failure/repair records, one row per event."""
        rows = []
        for group in self.groups.values():
            rows.extend(group.failure_log)
        return sorted(rows, key=lambda r: r["down_at"])

    def metrics(self):
        completed = [r for r in self.records if r.completion_time is not None]
        cycle = [r.cycle_time() for r in completed]
        ontime = [r.ontime() for r in completed]
        util = {name: g.utilization for name, g in self.groups.items()}
        by_prior = {}
        for p in sorted({r.priority for r in completed}):
            subset = [r for r in completed if r.priority == p]
            sub_cycle = [r.cycle_time() for r in subset]
            by_prior[str(p)] = {
                "n": len(subset),
                "avg_cycle_time": (sum(sub_cycle) / len(sub_cycle)) if sub_cycle else 0.0,
            }
        return {
            "policy": self.config.policy,
            "throughput": len(completed),
            "avg_cycle_time": (sum(cycle) / len(cycle)) if cycle else 0.0,
            "wip": self.arrived - len(completed),
            "ontime_rate": (sum(ontime) / len(ontime)) if ontime else 0.0,
            "cycle_by_priority": by_prior,
            "utilization": util,
        }


def run_experiment(configs):
    return [Fab(cfg).run() for cfg in configs]


def main():
    parser = argparse.ArgumentParser(description="AI fab simulation")
    parser.add_argument("--policy", default="fifo")
    parser.add_argument("--lots", type=int, default=300)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--arrival-mean", type=float, default=2.0)
    parser.add_argument(
        "--failure-shape",
        type=float,
        default=2.5,
        help="Weibull shape; 1.0 = exponential (memoryless)",
    )
    args = parser.parse_args()
    from scheduling import available_policies

    names = available_policies() if args.all else [args.policy]
    configs = [
        FabConfig(
            num_lots=args.lots,
            policy=n,
            arrival_mean=args.arrival_mean,
            failure_shape=args.failure_shape,
        )
        for n in names
    ]
    for res in run_experiment(configs):
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()