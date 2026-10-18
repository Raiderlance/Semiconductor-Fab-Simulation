"""Closed-loop maintenance study: reactive vs age-based vs learned-risk vs oracle.

Every scenario runs on the same seeds, so differences are paired. Arrivals
and lot work content are identical across scenarios. Failure draws are NOT
perfectly matched, because a PM consumes a fresh lifetime draw and shifts
the tool's RNG sequence, so expect some extra noise.

Scenario syntax:  reactive | age:<frac of MTBF> | risk:<threshold> | oracle:<threshold>

Usage (from src/):
    # 1. tune thresholds on tuning seeds
    python run_maintenance.py --seeds 15 --seed0 2000 --out ../data/maint_tune.csv
    # 2. evaluate the chosen ones on held-out seeds
    python run_maintenance.py --seeds 30 --seed0 1000 --scenarios reactive age:0.75 risk:0.6 oracle:0.6 --out ../data/maint_eval.csv
"""

import argparse
import csv
from pathlib import Path

from fab import Fab, FabConfig
from maintenance import MaintenanceConfig
from run_experiments import cohort_row, mean_ci

PAIRED = ["mean_age", "p95_age", "mean_tardiness", "ontime_rate"]
MEANS = ["mean_age", "n_failures", "n_pm", "litho_fail_frac", "litho_pm_frac",
         "n_unfinished"]
DEFAULT_SCENARIOS = [
    "reactive",
    "age:0.6", "age:0.75", "age:0.9", "age:1.05",
    "risk:0.5", "risk:0.6", "risk:0.7", "risk:0.8",
    "oracle:0.6", "oracle:0.7",
]


def parse_spec(spec, args):
    name, _, param = spec.partition(":")
    common = dict(check_interval=args.check_interval, pm_factor=args.pm_factor)
    if name == "reactive":
        return None
    if name == "age":
        return MaintenanceConfig(policy="age", age_frac=float(param), **common)
    if name == "risk":
        return MaintenanceConfig(policy="risk", risk_threshold=float(param),
                                 table_path=args.table, **common)
    if name == "oracle":
        return MaintenanceConfig(policy="oracle", risk_threshold=float(param),
                                 horizon=args.horizon, **common)
    raise ValueError(f"bad scenario: {spec!r}")


def run_one(seed, mcfg, args):
    fab = Fab(FabConfig(
        num_lots=100000,
        seed=seed,
        policy=args.dispatch,
        arrival_mean=args.arrival_mean,
        arrival_end=args.arrival_end,
        run_length=args.arrival_end + args.drain,
        due_factor=args.due_factor,
        maintenance=mcfg,
    ))
    fab.run()
    row = cohort_row(fab, args.warmup, args.arrival_end)
    end = fab.env.now
    litho = fab.groups["lithography"]
    row["n_failures"] = sum(len(g.failure_log) for g in fab.groups.values())
    row["n_pm"] = sum(len(g.pm_log) for g in fab.groups.values())
    row["litho_fail_frac"] = litho.down_time / (end * litho.num_machines)
    row["litho_pm_frac"] = litho.pm_time / (end * litho.num_machines)
    return row


def main():
    p = argparse.ArgumentParser(description="Closed-loop maintenance study")
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--seed0", type=int, default=2000)
    p.add_argument("--stride", type=int, default=10,
                   help="seed spacing (avoids shared failure RNG streams)")
    p.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    p.add_argument("--table", default="../data/risk_table.json")
    p.add_argument("--pm-factor", type=float, default=0.5,
                   help="PM duration as a fraction of MTTR")
    p.add_argument("--check-interval", type=float, default=5.0)
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--dispatch", default="fifo")
    p.add_argument("--arrival-mean", type=float, default=7.5)
    p.add_argument("--warmup", type=float, default=500.0)
    p.add_argument("--arrival-end", type=float, default=3000.0)
    p.add_argument("--drain", type=float, default=1000.0)
    p.add_argument("--due-factor", type=float, default=5.0)
    p.add_argument("--out", default="../data/maint_study.csv")
    args = p.parse_args()

    specs = list(args.scenarios)
    if "reactive" not in specs:
        specs.insert(0, "reactive")
    seeds = [args.seed0 + k * args.stride for k in range(args.seeds)]
    print(f"{len(specs)} scenarios x {len(seeds)} seeds = "
          f"{len(specs) * len(seeds)} runs  (pm_factor={args.pm_factor}, "
          f"dispatch={args.dispatch}, arrival_mean={args.arrival_mean})\n")

    results = {s: {} for s in specs}
    csv_rows = []
    for seed in seeds:
        for s in specs:
            row = run_one(seed, parse_spec(s, args), args)
            for m, v in row.items():
                results[s].setdefault(m, []).append(v)
            csv_rows.append({"scenario": s, "seed": seed, **row})

    print("Paired difference vs reactive (scenario - reactive), mean +/- 95% CI")
    print(f"{'scenario':14s}" + "".join(f"{m:>23s}" for m in PAIRED))
    for s in specs:
        if s == "reactive":
            continue
        cells = []
        for m in PAIRED:
            diffs = [a - b for a, b in zip(results[s][m], results["reactive"][m])]
            mu, hw = mean_ci(diffs)
            cells.append(f"{mu:>13.3f} +/-{hw:>6.3f}")
        print(f"{s:14s}" + "".join(cells))

    print("\nMeans across seeds")
    print(f"{'scenario':14s}" + "".join(f"{m:>17s}" for m in MEANS))
    for s in specs:
        cells = [f"{mean_ci(results[s][m])[0]:>17.3f}" for m in MEANS]
        print(f"{s:14s}" + "".join(cells))

    print("\nLowest mean cohort age per policy family (use for tuning):")
    fams = {}
    for s in specs:
        fams.setdefault(s.split(":")[0], []).append(s)
    for fam, ss in fams.items():
        best = min(ss, key=lambda s: mean_ci(results[s]["mean_age"])[0])
        print(f"  {fam:9s} {best}  (mean_age {mean_ci(results[best]['mean_age'])[0]:.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nper-seed rows written to {out}")


if __name__ == "__main__":
    main()