"""Run the simulation across many seeds and dump the tool failure log to CSV.

Each row is one failure/repair cycle of one tool. The CSV also records
run_length and failure_shape so it is self-describing: the trainer needs the
run length to build censoring-correct observations (tools that have NOT
failed by the end of the run still contribute observations).

Seeds are spaced by --seed-stride (default 10). MachineGroup seeds are
config.seed + group_index, so adjacent seeds would otherwise share failure
RNG streams across groups.

Usage (from src/):
    python export_failure_log.py --seeds 50 --failure-shape 2.5 --out ../data/failure_v2_weibull.csv
    python export_failure_log.py --seeds 50 --failure-shape 1.0 --out ../data/failure_v2_exp.csv
"""

import argparse
import csv
from pathlib import Path

from fab import Fab, FabConfig


FIELDS = [
    "seed",
    "group",
    "tool",
    "down_at",
    "up_at",
    "down_for",
    "time_since_last_repair",
    "is_first_failure",
    "run_length",
    "failure_shape",
]


def _run_one(seed, num_lots, run_length, failure_shape, arrival_mean):
    fab = Fab(
        FabConfig(
            num_lots=num_lots,
            seed=seed,
            run_length=run_length,
            failure_shape=failure_shape,
            arrival_mean=arrival_mean,
        )
    )
    fab.run()
    rows = fab.failure_log()
    # For the first failure of each tool, time_since_last_repair is really
    # "time since simulation start", so it is flagged.
    seen = set()
    for row in rows:
        key = (row["group"], row["tool"])
        row["seed"] = seed
        row["is_first_failure"] = int(key not in seen)
        row["run_length"] = run_length
        row["failure_shape"] = failure_shape
        seen.add(key)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Dump failure log to CSV")
    parser.add_argument("--lots", type=int, default=100000,
                        help="arrival cap; the run ends at --run-length anyway")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, default=0,
                        help="number of seeds to sweep (0 = single run)")
    parser.add_argument("--seed-stride", type=int, default=10,
                        help="spacing between seeds (avoids shared RNG streams)")
    parser.add_argument("--run-length", type=float, default=2000.0)
    parser.add_argument("--arrival-mean", type=float, default=7.5,
                        help="mean inter-arrival time (7.5 = realistic load)")
    parser.add_argument("--out", default="../data/failure_log.csv")
    parser.add_argument(
        "--failure-shape",
        type=float,
        default=2.5,
        help="Weibull shape; 1.0 = exponential (memoryless)",
    )
    args = parser.parse_args()

    n = args.seeds if args.seeds > 0 else 1
    seeds = [args.seed + k * args.seed_stride for k in range(n)]

    all_rows = []
    for seed in seeds:
        all_rows.extend(
            _run_one(seed, args.lots, args.run_length,
                     args.failure_shape, args.arrival_mean)
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"wrote {len(all_rows)} failure records across {len(seeds)} seed(s) "
          f"to {out_path} (shape={args.failure_shape}, run_length={args.run_length})")


if __name__ == "__main__":
    main()