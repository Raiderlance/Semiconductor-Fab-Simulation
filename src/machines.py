"""Machine group models for the semiconductor fab simulation.

Each tool is its own capacity-1 Resource. While up, a tool runs for a
Weibull-distributed lifetime, then fails and is repaired. If a maintenance
policy is attached (maintenance.py) a tool may be taken down earlier for
preventive maintenance (PM), which resets its age.

Stage two adds a noisy wear sensor: at every check the tool reads
    sensor = age / lifetime + noise
i.e. the fraction of its (hidden) life used up, plus Gaussian noise. The
lifetime itself is drawn exactly as before, so reactive behaviour is
unchanged. Sensor noise uses its own RNG stream per tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math
import random

import simpy


@dataclass
class MachineGroup:
    """A pool of identical machines with failure/repair behaviour."""

    name: str
    env: simpy.Environment
    num_machines: int
    process_time_mean: float
    process_time_cv: float = 0.1
    mtbf: float | None = None
    mttr: float | None = None
    failure_shape: float = 2.5
    seed: int | None = None
    # Optional MaintenancePolicy (None = purely reactive).
    maintenance: object | None = None
    # Stage two: sensor noise (std dev, in units of "fraction of life"),
    # optional logging of every sensor reading, and the logging interval.
    sensor_noise: float = 0.15
    log_sensor: bool = False
    sensor_interval: float = 5.0

    # Runtime state ------------------------------------------------------
    machines: list[simpy.Resource] = field(init=False)
    busy_time: float = field(default=0.0, init=False)
    down_time: float = field(default=0.0, init=False)
    pm_time: float = field(default=0.0, init=False)
    processed: int = field(default=0, init=False)
    failure_log: list[dict] = field(default_factory=list, init=False)
    pm_log: list[dict] = field(default_factory=list, init=False)
    sensor_log: list[dict] = field(default_factory=list, init=False)
    last_repair: list[float] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.machines = [
            simpy.Resource(self.env, capacity=1) for _ in range(self.num_machines)
        ]
        self._fail_rngs = [
            random.Random(f"{self.seed}-fail-{i}") for i in range(self.num_machines)
        ]
        self._sensor_rngs = [
            random.Random(f"{self.seed}-sensor-{i}") for i in range(self.num_machines)
        ]
        self.last_repair = [0.0] * self.num_machines
        if self.mtbf and self.mttr:
            for idx in range(self.num_machines):
                self.env.process(self._failure_cycle(idx))

    # -- helpers -------------------------------------------------
    def _time_to_failure(self, rng: random.Random) -> float:
        """Sample time to next failure; the mean lifetime equals `mtbf`."""
        assert self.mtbf
        k = self.failure_shape
        if k <= 1.0:
            return rng.expovariate(1.0 / self.mtbf)
        scale = self.mtbf / math.gamma(1.0 + 1.0 / k)
        return scale * rng.weibullvariate(1.0, k)

    def process_time(self, rng: random.Random) -> float:
        """Sample a processing time from *rng* (gamma: positive and skewed)."""
        if self.process_time_cv <= 0:
            return self.process_time_mean
        shape = 1.0 / (self.process_time_cv ** 2)
        scale = self.process_time_mean / shape
        return rng.gammavariate(shape, scale)

    def acquire_any(self, result):
        """Get the first free tool and put (tool, req) into *result*."""
        pending = [tool.request() for tool in self.machines]
        try:
            yield simpy.AnyOf(self.env, pending)
        finally:
            for req in pending:
                if not req.triggered:
                    req.cancel()
        chosen = None
        for tool, req in zip(self.machines, pending):
            if not req.triggered:
                continue
            if chosen is None:
                chosen = (tool, req)
            else:
                tool.release(req)
        if chosen is None:
            raise RuntimeError("acquire_any woke with no triggered request")
        yield result.put(chosen)

    def _failure_cycle(self, tool_id: int):
        """Alternate up / down periods for a single machine.

        Without a policy and without sensor logging this is the original
        model (one timeout per lifetime). Otherwise the lifetime is walked
        in steps so the policy (or the log) can look at the tool every step.
        """
        assert self.mtbf and self.mttr
        machine = self.machines[tool_id]
        rng = self._fail_rngs[tool_id]
        srng = self._sensor_rngs[tool_id]
        pol = self.maintenance
        use_pm = pol is not None and pol.cfg.policy != "none"
        stepped = use_pm or self.log_sensor
        step = pol.cfg.check_interval if use_pm else self.sensor_interval
        while True:
            life_start = self.env.now
            ttf = self._time_to_failure(rng)
            pm_due = False
            if not stepped:
                yield self.env.timeout(ttf)
            else:
                elapsed = 0.0
                while True:
                    left = ttf - elapsed
                    if step >= left:
                        yield self.env.timeout(left)  # natural failure time
                        break
                    yield self.env.timeout(step)
                    elapsed += step
                    sensor = elapsed / ttf + srng.gauss(0.0, self.sensor_noise)
                    if self.log_sensor:
                        self.sensor_log.append(
                            {
                                "group": self.name,
                                "tool": tool_id,
                                "t": self.env.now,
                                "age": elapsed,
                                "sensor": sensor,
                                "remaining": ttf - elapsed,
                            }
                        )
                    if use_pm and pol.should_pm(
                        self.name, elapsed, sensor=sensor, remaining=ttf - elapsed
                    ):
                        pm_due = True
                        break
            with machine.request() as req:
                yield req  # only go down once the machine is free
                if pm_due and (self.env.now - life_start) < ttf:
                    # Genuine PM: the failure time has not passed yet.
                    pm_for = pol.pm_duration(self.mttr)
                    self.pm_time += pm_for
                    self.pm_log.append(
                        {
                            "group": self.name,
                            "tool": tool_id,
                            "at": self.env.now,
                            "age": self.env.now - life_start,
                            "duration": pm_for,
                        }
                    )
                    yield self.env.timeout(pm_for)
                    self.last_repair[tool_id] = self.env.now
                    continue
                # Failure (including a failure time that passed while PM
                # was waiting for the tool to become free).
                down_for = rng.expovariate(1.0 / self.mttr)
                self.down_time += down_for
                down_at = self.env.now
                self.failure_log.append(
                    {
                        "group": self.name,
                        "tool": tool_id,
                        "down_at": down_at,
                        "up_at": down_at + down_for,
                        "down_for": down_for,
                        "time_since_last_repair": down_at - self.last_repair[tool_id],
                    }
                )
                self.last_repair[tool_id] = down_at + down_for
                yield self.env.timeout(down_for)

    @property
    def capacity(self) -> int:
        return self.num_machines

    @property
    def utilization(self) -> float:
        """Fraction of available machine-time spent processing."""
        total = self.env.now * self.num_machines - self.down_time - self.pm_time
        return self.busy_time / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, n={self.num_machines})"