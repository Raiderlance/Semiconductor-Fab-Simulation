"""Preventive-maintenance (PM) policies for the fab simulator.

A policy answers one question, asked every `check_interval` time units for
each tool that is up: "should this tool go down for PM now?"

  none     never (purely reactive repair)
  age      PM once tool age >= age_frac x group MTBF
  risk     stage one: learned P(fail | age, group) from a 1-D table
  oracle   stage one: true calendar-age hazard (ceiling for stage one)
  sensor   stage two: learned P(fail | age, sensor, group) from a 2-D table
  perfect  stage two: knows the exact remaining life; PM once it is
           <= perfect_lead (an unreachable ceiling)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

POLICIES = ("none", "age", "risk", "oracle", "sensor", "perfect")


@dataclass
class MaintenanceConfig:
    policy: str = "none"
    check_interval: float = 5.0
    # PM duration = pm_factor x the group's MTTR (deterministic).
    pm_factor: float = 0.5
    # "age" policy: PM when age >= age_frac x group MTBF.
    age_frac: float = 0.8
    # "risk" / "oracle" / "sensor": PM when P(fail within horizon) >= this.
    risk_threshold: float = 0.6
    horizon: float = 60.0
    # "risk" / "sensor": path to the JSON table.
    table_path: str | None = None
    # "perfect": PM once true remaining life <= this.
    perfect_lead: float = 15.0


class MaintenancePolicy:
    def __init__(self, cfg, mtbf_by_group, failure_shape):
        if cfg.policy not in POLICIES:
            raise ValueError(f"unknown maintenance policy: {cfg.policy!r}")
        self.cfg = cfg
        self.mtbf = dict(mtbf_by_group)
        self.shape = failure_shape
        self.table = None
        self.grid_step = 1.0
        if cfg.policy in ("risk", "sensor"):
            if not cfg.table_path:
                raise ValueError(f"{cfg.policy} policy needs table_path")
            data = json.loads(Path(cfg.table_path).read_text(encoding="utf-8"))
            self.table = data["table"]
            missing = set(self.mtbf) - set(self.table)
            if missing:
                raise ValueError(f"table has no rows for {sorted(missing)}")
            if cfg.policy == "risk":
                self.grid_step = float(data["grid_step"])
            else:
                self.age_step = float(data["age_step"])
                self.s_min = float(data["s_min"])
                self.s_step = float(data["s_step"])

    def pm_duration(self, mttr):
        return self.cfg.pm_factor * mttr

    # -- stage one ------------------------------------------------------
    def table_risk(self, group, age):
        row = self.table[group]
        x = age / self.grid_step
        i = int(x)
        if i >= len(row) - 1:
            return row[-1]  # beyond the grid: hold the last value
        f = x - i
        return row[i] * (1.0 - f) + row[i + 1] * f

    def true_risk(self, group, age):
        mtbf, k, h = self.mtbf[group], self.shape, self.cfg.horizon
        if k <= 1.0:
            return 1.0 - math.exp(-h / mtbf)
        scale = mtbf / math.gamma(1.0 + 1.0 / k)
        return 1.0 - math.exp((age / scale) ** k - ((age + h) / scale) ** k)

    def risk(self, group, age):
        if self.cfg.policy == "risk":
            return self.table_risk(group, age)
        return self.true_risk(group, age)

    # -- stage two ------------------------------------------------------
    def sensor_risk(self, group, age, sensor):
        rows = self.table[group]
        i = min(max(int(round(age / self.age_step)), 0), len(rows) - 1)
        j = min(max(int(round((sensor - self.s_min) / self.s_step)), 0),
                len(rows[0]) - 1)
        return rows[i][j]

    def should_pm(self, group, age, sensor=None, remaining=None):
        p = self.cfg.policy
        if p == "none":
            return False
        if p == "age":
            return age >= self.cfg.age_frac * self.mtbf[group]
        if p == "perfect":
            return remaining is not None and remaining <= self.cfg.perfect_lead
        if p == "sensor":
            return (sensor is not None
                    and self.sensor_risk(group, age, sensor) >= self.cfg.risk_threshold)
        return self.risk(group, age) >= self.cfg.risk_threshold