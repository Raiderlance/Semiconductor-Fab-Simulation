"""Run reactive simulations and log every sensor reading.

Each row: one tool, one moment (every 5 time units while the tool is up):
  age        time since the tool's last repair
  sensor     noisy wear reading (age / lifetime + noise)
  remaining  TRUE time left before failure (used only to build labels)

Seeds are spaced by --stride so no two runs share failure RNG streams.
Training seeds (42, 52, ...) differ from the tuning (2000+) and
evaluation (1000+) seeds used later.

Usage (from src/):
    python export_sensor_log.py --seeds 50 --out ../data/sensor_log.csv
"""

import argparse
import csv
from pathlib import Path

from fab import Fab, FabConfig

FIELDS = ["seed", "sensor_noise", "group", "tool", "t", "age", "sensor", "remaining"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--run-length", type=float, default=2000.0)
    p.add_argument("--arrival-mean", type=float, default=7.5)
    p.add_argument("--sensor-noise", type=float, default=0.15)
    p.add_argument("--out", default="../data/sensor_log.csv")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for k in range(args.seeds):
            seed = args.seed + k * args.stride
            fab = Fab(FabConfig(
                num_lots=100000, seed=seed, run_length=args.run_length,
                arrival_mean=args.arrival_mean,
                sensor_noise=args.sensor_noise, log_sensor=True,
            ))
            fab.run()
            for g in fab.groups.values():
                for r in g.sensor_log:
                    w.writerow({"seed": seed, "sensor_noise": args.sensor_noise, **r})
                    n += 1
    print(f"wrote {n} sensor readings across {args.seeds} seeds to {out} "
          f"(sensor_noise={args.sensor_noise})")


if __name__ == "__main__":
    main()