"""Stage two: does a wear-sensor model beat a plain age rule?

For each PM cost:
  1. TUNE: try a few settings of each policy on tuning seeds and keep the one
     that leaves the bottleneck (lithography) down the least.
  2. EVALUATE: compare the winners on fresh held-out seeds.

"litho down %" = share of the run the lithography tool spends broken or in PM.
"vs reactive" = paired difference in average cycle time (negative = better).

Usage (from src/):
    python run_stage2.py --out ../data/stage2.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from fab import Fab, FabConfig
from maintenance import MaintenanceConfig
from run_experiments import cohort_row, mean_ci

KEYS = ["mean_age", "n_failures", "n_pm", "litho_down"]


def make_cfg(spec, pm_factor, args):
    name, _, param = spec.partition(":")
    common = dict(check_interval=args.check_interval, pm_factor=pm_factor)
    if name == "reactive":
        return None
    if name == "age":
        return MaintenanceConfig(policy="age", age_frac=float(param), **common)
    if name == "sensor":
        return MaintenanceConfig(policy="sensor", risk_threshold=float(param),
                                 table_path=args.table, **common)
    if name == "perfect":
        return MaintenanceConfig(policy="perfect", perfect_lead=float(param), **common)
    raise ValueError(f"bad scenario: {spec!r}")


def run_one(seed, spec, pm_factor, args, noise):
    fab = Fab(FabConfig(
        num_lots=100000, seed=seed, policy="fifo",
        arrival_mean=args.arrival_mean, arrival_end=args.arrival_end,
        run_length=args.arrival_end + args.drain, due_factor=5.0,
        sensor_noise=noise, maintenance=make_cfg(spec, pm_factor, args),
    ))
    fab.run()
    row = cohort_row(fab, args.warmup, args.arrival_end)
    litho = fab.groups["lithography"]
    end = fab.env.now
    return {
        "mean_age": row["mean_age"],
        "n_failures": sum(len(g.failure_log) for g in fab.groups.values()),
        "n_pm": sum(len(g.pm_log) for g in fab.groups.values()),
        "litho_down": (litho.down_time + litho.pm_time) / (end * litho.num_machines),
    }


def evaluate(phase, specs, seeds, pm_factor, args, noise):
    print(f"  {phase}: {len(specs)} policies x {len(seeds)} seeds "
          f"= {len(specs) * len(seeds)} runs ...", flush=True)
    res = {s: {k: [] for k in KEYS} for s in specs}
    rows = []
    for seed in seeds:
        for s in specs:
            r = run_one(seed, s, pm_factor, args, noise)
            for k in KEYS:
                res[s][k].append(r[k])
            rows.append({"phase": phase, "pm_factor": pm_factor,
                         "policy": s, "seed": seed, **r})
    return res, rows


def show(title, specs, res):
    print(f"\n{title}")
    print(f"{'policy':14s}{'failures':>10s}{'PMs':>8s}{'litho down %':>14s}"
          f"{'cycle time':>12s}{'vs reactive (95% CI)':>26s}")
    base = res["reactive"]["mean_age"]
    for s in specs:
        r = res[s]
        if s == "reactive":
            diff = "-"
        else:
            d = [a - b for a, b in zip(r["mean_age"], base)]
            mu, hw = mean_ci(d)
            diff = f"{mu:+.2f} +/- {hw:.2f}"
        print(f"{s:14s}{statistics.mean(r['n_failures']):>10.0f}"
              f"{statistics.mean(r['n_pm']):>8.0f}"
              f"{100 * statistics.mean(r['litho_down']):>14.1f}"
              f"{statistics.mean(r['mean_age']):>12.2f}{diff:>26s}")


def main():
    p = argparse.ArgumentParser(description="Stage two maintenance study")
    p.add_argument("--table", default="../data/sensor_table.json")
    p.add_argument("--pm-factors", type=float, nargs="+", default=[0.5, 0.2])
    p.add_argument("--age-grid", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    p.add_argument("--sensor-grid", type=float, nargs="+",
                   default=[0.4, 0.5, 0.6, 0.7, 0.8])
    p.add_argument("--perfect-lead", type=float, default=15.0)
    p.add_argument("--tune-seeds", type=int, default=15)
    p.add_argument("--tune-seed0", type=int, default=2000)
    p.add_argument("--eval-seeds", type=int, default=30)
    p.add_argument("--eval-seed0", type=int, default=1000)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--check-interval", type=float, default=5.0)
    p.add_argument("--arrival-mean", type=float, default=7.5)
    p.add_argument("--warmup", type=float, default=500.0)
    p.add_argument("--arrival-end", type=float, default=3000.0)
    p.add_argument("--drain", type=float, default=1000.0)
    p.add_argument("--out", default="../data/stage2.csv")
    args = p.parse_args()

    meta = json.loads(Path(args.table).read_text(encoding="utf-8"))
    noise = float(meta["sensor_noise"])  # must match what the model was trained on
    tune = [args.tune_seed0 + k * args.stride for k in range(args.tune_seeds)]
    held = [args.eval_seed0 + k * args.stride for k in range(args.eval_seeds)]
    age_grid = [f"age:{x}" for x in args.age_grid]
    sens_grid = [f"sensor:{x}" for x in args.sensor_grid]
    print(f"sensor_noise={noise}  horizon of model={meta['horizon']}  "
          f"PM costs (x MTTR)={args.pm_factors}")

    all_rows = []
    for pmf in args.pm_factors:
        print(f"\n=== PM cost = {pmf} x repair time ===")
        tspecs = ["reactive"] + age_grid + sens_grid
        tres, trows = evaluate("tune", tspecs, tune, pmf, args, noise)
        best_age = min(age_grid, key=lambda s: statistics.mean(tres[s]["litho_down"]))
        best_sens = min(sens_grid, key=lambda s: statistics.mean(tres[s]["litho_down"]))
        show(f"TUNING seeds (PM cost {pmf}): winners = {best_age}, {best_sens}",
             tspecs, tres)

        especs = ["reactive", best_age, best_sens, f"perfect:{args.perfect_lead}"]
        eres, erows = evaluate("eval", especs, held, pmf, args, noise)
        show(f"HELD-OUT seeds (PM cost {pmf}) -- the result to report", especs, eres)
        all_rows += trows + erows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nper-seed rows written to {out}")


if __name__ == "__main__":
    main()