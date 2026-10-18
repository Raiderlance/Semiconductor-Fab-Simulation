"""Train the wear-sensor failure model and export it as a lookup table.

Label: "the tool fails within the next HORIZON time units", where failure
time is the tool's latent lifetime. Evaluation is GroupKFold by seed.

The feature-set comparison answers: does the sensor add anything beyond age?

Usage (from src/):
    python train_sensor_model.py --in ../data/sensor_log.csv --out ../data/sensor_table.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from fab import MACHINE_DEFAULTS

GROUPS = list(MACHINE_DEFAULTS)


def make_X(df, feats):
    X = pd.DataFrame(index=np.arange(len(df)))
    for f in feats:
        X[f] = df[f].to_numpy(dtype=float)
    groups = df["group"].to_numpy()
    for g in GROUPS:
        X[f"g_{g}"] = (groups == g).astype(float)
    return X


def cv_auc(df, feats, folds):
    X = make_X(df, feats)
    y = df["label"].to_numpy()
    aucs = []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups=df["seed"]):
        clf = HistGradientBoostingClassifier(random_state=0)
        clf.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X.iloc[te])[:, 1]))
    return np.array(aucs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="../data/sensor_log.csv")
    p.add_argument("--out", default="../data/sensor_table.json")
    p.add_argument("--horizon", type=float, default=30.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-age", type=float, default=400.0)
    args = p.parse_args()

    df = pd.read_csv(args.inp)
    noise = float(df["sensor_noise"].iloc[0])
    df["label"] = (df["remaining"] <= args.horizon).astype(int)
    print(f"{len(df)} readings, {df['seed'].nunique()} seeds, "
          f"sensor_noise={noise}, horizon={args.horizon}, "
          f"positive rate={df['label'].mean():.3f}")
    print(f"GroupKFold by seed, {args.folds} folds. AUC mean +/- sd\n")

    sets = {
        "group only": [],
        "age only": ["age"],
        "sensor only": ["sensor"],
        "age + sensor + group": ["age", "sensor"],
    }
    for name, feats in sets.items():
        a = cv_auc(df, feats, args.folds)
        print(f"{name:24s} {a.mean():.3f} +/- {a.std(ddof=1):.3f}")
    print(f"{'perfect remaining life':24s} 1.000 (by construction: the upper bound)\n")

    clf = HistGradientBoostingClassifier(random_state=0)
    clf.fit(make_X(df, ["age", "sensor"]), df["label"])

    age_step, s_min, s_max, s_step = 2.0, -0.4, 1.6, 0.04
    ages = np.arange(0.0, args.max_age + age_step / 2, age_step)
    sens = s_min + s_step * np.arange(int(round((s_max - s_min) / s_step)) + 1)
    table = {}
    for g in GROUPS:
        A, S = np.meshgrid(ages, sens, indexing="ij")
        grid = pd.DataFrame({"age": A.ravel(), "sensor": S.ravel(), "group": g})
        prob = clf.predict_proba(make_X(grid, ["age", "sensor"]))[:, 1]
        table[g] = np.round(prob.reshape(len(ages), len(sens)), 4).tolist()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "horizon": args.horizon, "sensor_noise": noise,
        "age_step": age_step, "s_min": s_min, "s_step": s_step,
        "table": table,
    }), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()