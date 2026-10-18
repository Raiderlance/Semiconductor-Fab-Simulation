"""Multi-seed policy comparison on a fixed post-warm-up cohort, with load
sweeps, tail metrics and due-date studies.

Every policy runs on the same seeds (common random numbers), so differences
are paired. Metrics cover lots that ARRIVE in [warmup, arrival_end]. Lots
still unfinished when the run ends are counted, not dropped: their age at
the end is a lower bound on their cycle time (censoring), and they count as
missed deadlines for on-time rate.

Tail metrics are computed per seed (p95 and max over that seed's cohort) and
then averaged across seeds.

Due-date designs (see DESIGNS): slack depends on the lot's own work or not
("work" vs "rand"), and priority lots get tighter due dates or not
("_p" vs "_np"). Compare paired differences WITHIN a design; absolute
on-time rates are not comparable across designs because mean slack differs.

Usage (from src/):
    python run_experiments.py --seeds 30
    python run_experiments.py --seeds 30 --sweep 8.5 8.0 7.5 7.0
    python run_experiments.py --seeds 30 --due-factors 3 5 8 --out ../data/due_sweep.csv
    python run_experiments.py --seeds 30 --designs work_p work_np rand_p rand_np \
        --out ../data/due_design.csv
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

from fab import Fab, FabConfig
from scheduling import available_policies

# Two-sided 95% t critical values, by degrees of freedom.
T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

# design name -> (due_mode, priority_due_mult)
DESIGNS = {
    "work_p": ("work", 0.5),    # current default
    "work_np": ("work", 1.0),   # work-proportional, no priority signal
    "rand_p": ("random", 0.5),  # independent of work, priority tighter
    "rand_np": ("random", 1.0), # independent of work, no priority signal
}

METRICS = [
    "n_cohort", "n_unfinished",
    "mean_age", "mean_age_normal", "mean_age_high",
    "p95_age", "p95_age_normal", "p95_age_high",
    "max_age", "max_age_normal",
    "ontime_rate", "mean_tardiness", "offered_load",
]
PAIRED = [
    "mean_age", "mean_age_normal", "mean_age_high",
    "p95_age", "p95_age_normal", "max_age_normal",
    "ontime_rate", "mean_tardiness",
]
KEY = ["mean_age_normal", "mean_age_high", "ontime_rate", "mean_tardiness"]


def mean_ci(xs):
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    if n < 2:
        return m, float("nan")
    t = T975.get(n - 1, 1.96)
    return m, t * statistics.stdev(xs) / math.sqrt(n)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _max(xs):
    return max(xs) if xs else float("nan")


def percentile(xs, q):
    """Linear-interpolated percentile, q in [0, 1]."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def cohort_row(fab, warmup, arrival_end):
    end = fab.env.now
    cohort = [l for l in fab.all_lots if warmup <= l.arrival_time <= arrival_end]

    def finish(l):
        return l.completion_time if l.completion_time is not None else end

    ages = [finish(l) - l.arrival_time for l in cohort]
    normal = [finish(l) - l.arrival_time for l in cohort if l.priority == 0]
    high = [finish(l) - l.arrival_time for l in cohort if l.priority == 1]
    ontime = sum(
        1 for l in cohort
        if l.completion_time is not None and l.completion_time <= l.due_time
    )
    tardiness = [max(0.0, finish(l) - l.due_time) for l in cohort]
    litho_work = sum(fab.stage_time[(l.lot_id, "lithography")] for l in cohort)
    window = arrival_end - warmup
    return {
        "n_cohort": len(cohort),
        "n_unfinished": sum(1 for l in cohort if l.completion_time is None),
        "mean_age": _mean(ages),
        "mean_age_normal": _mean(normal),
        "mean_age_high": _mean(high),
        "p95_age": percentile(ages, 0.95),
        "p95_age_normal": percentile(normal, 0.95),
        "p95_age_high": percentile(high, 0.95),
        "max_age": _max(ages),
        "max_age_normal": _max(normal),
        "ontime_rate": ontime / len(cohort) if cohort else float("nan"),
        "mean_tardiness": _mean(tardiness),
        # Nominal lithography load over the arrival window (ignores downtime).
        "offered_load": litho_work / window if window > 0 else float("nan"),
    }


def run_scenario(args, arrival_mean, due_factor, design, policies, seeds):
    """Run every policy on every seed for one (load, due_factor, design)."""
    due_mode, prio_mult = DESIGNS[design]
    results = {pol: {m: [] for m in METRICS} for pol in policies}
    csv_rows = []
    for seed in seeds:
        for pol in policies:
            cfg = FabConfig(
                num_lots=100000,
                seed=seed,
                policy=pol,
                arrival_mean=arrival_mean,
                arrival_end=args.arrival_end,
                run_length=args.arrival_end + args.drain,
                due_factor=due_factor,
                due_mode=due_mode,
                priority_due_mult=prio_mult,
            )
            fab = Fab(cfg)
            fab.run()
            row = cohort_row(fab, args.warmup, args.arrival_end)
            for m in METRICS:
                results[pol][m].append(row[m])
            csv_rows.append({
                "arrival_mean": arrival_mean, "due_factor": due_factor,
                "design": design, "policy": pol, "seed": seed, **row,
            })
    return results, csv_rows


def print_full(label, results, policies):
    print("=" * 100)
    print(label)
    print("=" * 100)
    print("\nMean +/- 95% CI across seeds (metrics down, policies across)")
    print(f"{'metric':18s}" + "".join(f"{p:>21s}" for p in policies))
    for m in METRICS:
        cells = []
        for pol in policies:
            mu, hw = mean_ci(results[pol][m])
            cells.append(f"{mu:>11.3f} +/-{hw:>6.3f}")
        print(f"{m:18s}" + "".join(cells))
    if "fifo" in results:
        others = [p for p in policies if p != "fifo"]
        print("\nPaired difference vs fifo (policy - fifo), mean +/- 95% CI")
        print(f"{'metric':18s}" + "".join(f"{p:>21s}" for p in others))
        for m in PAIRED:
            cells = []
            for pol in others:
                diffs = [a - b for a, b in zip(results[pol][m], results["fifo"][m])]
                mu, hw = mean_ci(diffs)
                cells.append(f"{mu:>11.3f} +/-{hw:>6.3f}")
            print(f"{m:18s}" + "".join(cells))
    print()


def print_compact(label, results, policies):
    print("-" * 100)
    print(label)
    if "fifo" not in results:
        return
    mu, hw = mean_ci(results["fifo"]["ontime_rate"])
    mt, ht = mean_ci(results["fifo"]["mean_tardiness"])
    print(f"FIFO baseline: ontime_rate {mu:.3f} +/- {hw:.3f}   "
          f"mean_tardiness {mt:.3f} +/- {ht:.3f}")
    print("Paired difference vs fifo (policy - fifo), mean +/- 95% CI")
    print(f"{'policy':10s}" + "".join(f"{m:>23s}" for m in KEY))
    for pol in policies:
        if pol == "fifo":
            continue
        cells = []
        for m in KEY:
            diffs = [a - b for a, b in zip(results[pol][m], results["fifo"][m])]
            mu, hw = mean_ci(diffs)
            cells.append(f"{mu:>13.3f} +/-{hw:>6.3f}")
        print(f"{pol:10s}" + "".join(cells))
    print()


def main():
    p = argparse.ArgumentParser(description="Multi-seed policy comparison")
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--arrival-mean", type=float, default=7.5,
                   help="mean inter-arrival time (used when --sweep is not given)")
    p.add_argument("--sweep", type=float, nargs="+", default=None,
                   help="list of arrival_mean values, e.g. --sweep 8.5 8.0 7.5 7.0")
    p.add_argument("--warmup", type=float, default=500.0)
    p.add_argument("--arrival-end", type=float, default=3000.0)
    p.add_argument("--drain", type=float, default=1000.0,
                   help="extra time after arrivals stop so the cohort can finish")
    p.add_argument("--due-factor", type=float, default=5.0,
                   help="due slack = factor x work (see --designs)")
    p.add_argument("--due-factors", type=float, nargs="+", default=None,
                   help="sweep several due factors, e.g. --due-factors 3 5 8")
    p.add_argument("--designs", nargs="+", choices=list(DESIGNS), default=["work_p"],
                   help="due-date designs to run (default: work_p)")
    p.add_argument("--full", action="store_true",
                   help="print the full tables for every scenario")
    p.add_argument("--out", default="../data/policy_due_study.csv")
    args = p.parse_args()

    policies = available_policies()
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    loads = args.sweep if args.sweep else [args.arrival_mean]
    factors = args.due_factors if args.due_factors else [args.due_factor]
    scenarios = [(am, df, d) for am in loads for df in factors for d in args.designs]
    show_full = args.full or len(scenarios) == 1

    print(f"{len(scenarios)} scenario(s) x {len(policies)} policies x "
          f"{len(seeds)} seeds = {len(scenarios) * len(policies) * len(seeds)} runs\n")

    all_rows = []
    for am, df, design in scenarios:
        results, rows = run_scenario(args, am, df, design, policies, seeds)
        all_rows.extend(rows)
        label = (f"arrival_mean={am}  due_factor={df}  design={design}  "
                 f"seeds={len(seeds)}  warmup={args.warmup}  "
                 f"arrival_end={args.arrival_end}  drain={args.drain}")
        if show_full:
            print_full(label, results, policies)
        else:
            print_compact(label, results, policies)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["arrival_mean", "due_factor", "design", "policy", "seed"]
            + METRICS,
        )
        w.writeheader()
        w.writerows(all_rows)
    print(f"per-seed rows written to {out}")


if __name__ == "__main__":
    main()