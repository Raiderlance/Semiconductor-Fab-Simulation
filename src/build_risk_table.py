"""Train the failure-risk model and export it as a lookup table.

The simulator's maintenance policy reads this table instead of calling
sklearn at runtime. Features are (age, group), matching train_failure_model.

Usage (from src/):
    python build_risk_table.py --in ../data/failure_v2_weibull.csv --out ../data/risk_table.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from fab import MACHINE_DEFAULTS
from train_failure_model import build_observations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="../data/failure_v2_weibull.csv")
    p.add_argument("--out", default="../data/risk_table.json")
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--max-age", type=float, default=500.0)
    p.add_argument("--grid-step", type=float, default=1.0)
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    shape = float(df["failure_shape"].iloc[0])
    obs = build_observations(df, args.horizon, args.step).reset_index(drop=True)
    groups = list(MACHINE_DEFAULTS)

    def design(age, group):
        X = pd.DataFrame({"age": np.asarray(age, dtype=float)})
        for h in groups:
            X[f"g_{h}"] = (np.asarray(group) == h).astype(float)
        return X

    X = design(obs["age"], obs["group"])
    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(X, obs["label"])

    ages = np.arange(0.0, args.max_age + args.grid_step, args.grid_step)
    table = {}
    for g in groups:
        G = design(ages, [g] * len(ages))
        table[g] = [float(v) for v in clf.predict_proba(G[X.columns])[:, 1]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "horizon": args.horizon,
                "grid_step": args.grid_step,
                "failure_shape": shape,
                "trained_on": args.inp,
                "n_obs": int(len(obs)),
                "table": table,
            }
        ),
        encoding="utf-8",
    )
    print(f"trained on {len(obs)} observations from {args.inp}")
    print(f"wrote {out}\n")
    print("age at which learned risk first reaches 0.3 / 0.5 / 0.7 / 0.8:")
    for g in groups:
        row = np.array(table[g])
        cells = []
        for t in (0.3, 0.5, 0.7, 0.8):
            idx = np.argmax(row >= t) if (row >= t).any() else None
            cells.append("never" if idx is None else f"{ages[idx]:.0f}")
        print(f"  {g:12s} (MTBF {MACHINE_DEFAULTS[g]['mtbf']:.0f}): " + "  ".join(cells))


if __name__ == "__main__":
    main()