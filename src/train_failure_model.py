"""Tool-failure prediction from the simulated failure log, with ablations.

Observations are per physical tool. At each sample time t (every STEP units,
up to run_length - horizon) where the tool is UP, we record:
  age               time since the tool's last repair (or since t=0)
  n_failures_so_far failures of this tool before t
  group             machine group (one-hot)
and the label "this tool fails within the next HORIZON units".

Sampling ends at a fixed time (run_length - horizon), independent of when
failures happen. Tools that never fail in a seed still contribute negatives.
Ending sampling at the last failure instead would leak the label.

Evaluation is GroupKFold by seed, so rows from one trajectory never straddle
the split. Feature-set ablations separate what the group alone predicts
(different MTBFs) from what age adds. An oracle that knows the true hazard
gives a ceiling (approximate: it ignores the delay a failure spends waiting
for its tool to become free).
"""

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from fab import MACHINE_DEFAULTS


def build_observations(df, horizon, step):
    """One row per (seed, tool, sample time) with the tool up at that time."""
    run_length = float(df["run_length"].iloc[0])
    events = {
        key: g.sort_values("down_at")
        for key, g in df.groupby(["seed", "group", "tool"])
    }
    rows = []
    for seed in sorted(df["seed"].unique()):
        for group, d in MACHINE_DEFAULTS.items():
            for tool in range(d["num_machines"]):
                ev = events.get((seed, group, tool))
                downs = ev["down_at"].to_numpy() if ev is not None else np.array([])
                ups = ev["up_at"].to_numpy() if ev is not None else np.array([])
                t = step
                while t <= run_length - horizon:
                    n_down = int(np.searchsorted(downs, t, side="right"))
                    in_repair = n_down > 0 and ups[n_down - 1] > t
                    if not in_repair:
                        last_up = ups[n_down - 1] if n_down > 0 else 0.0
                        nxt = downs[n_down] if n_down < len(downs) else math.inf
                        rows.append(
                            {
                                "seed": seed,
                                "group": group,
                                "tool": tool,
                                "t": t,
                                "age": t - last_up,
                                "n_failures_so_far": n_down,
                                "label": int(nxt <= t + horizon),
                            }
                        )
                    t += step
    return pd.DataFrame(rows)


def true_failure_prob(groups, age, horizon, shape):
    """P(fail within horizon | survived to age) under the true model."""
    mtbf = groups.map(lambda g: MACHINE_DEFAULTS[g]["mtbf"]).to_numpy(dtype=float)
    age = np.asarray(age, dtype=float)
    if shape <= 1.0:
        return 1.0 - np.exp(-horizon / mtbf)
    scale = mtbf / math.gamma(1.0 + 1.0 / shape)
    return 1.0 - np.exp((age / scale) ** shape - ((age + horizon) / scale) ** shape)


def cv_auc(obs, feats, n_splits):
    aucs = []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(obs, obs["label"], groups=obs["seed"]):
        clf = HistGradientBoostingClassifier(random_state=0)
        clf.fit(obs.iloc[tr][feats], obs.iloc[tr]["label"])
        p = clf.predict_proba(obs.iloc[te][feats])[:, 1]
        aucs.append(roc_auc_score(obs.iloc[te]["label"], p))
    return np.array(aucs)


def oracle_auc(obs, horizon, shape, n_splits):
    score = true_failure_prob(obs["group"], obs["age"], horizon, shape)
    aucs = []
    gkf = GroupKFold(n_splits=n_splits)
    for _, te in gkf.split(obs, obs["label"], groups=obs["seed"]):
        aucs.append(roc_auc_score(obs.iloc[te]["label"], score[te]))
    return np.array(aucs)


def report(path, horizon, step, n_splits):
    df = pd.read_csv(path)
    shape = float(df["failure_shape"].iloc[0])
    obs = build_observations(df, horizon, step)
    obs = obs.reset_index(drop=True)
    dummies = pd.get_dummies(obs["group"], prefix="g").astype(float)
    obs = pd.concat([obs, dummies], axis=1)
    gcols = list(dummies.columns)

    print("=" * 78)
    print(f"{path}  (failure_shape={shape}, horizon={horizon}, step={step})")
    print(f"failure rows: {len(df)}  seeds: {obs['seed'].nunique()}  "
          f"observations: {len(obs)}  positive rate: {obs['label'].mean():.3f}")
    print(f"GroupKFold by seed, {n_splits} folds. AUC mean +/- sd across folds")
    print("-" * 78)

    sets = {
        "group only": gcols,
        "age only": ["age"],
        "age + group": ["age"] + gcols,
        "age + group + n_failures": ["age", "n_failures_so_far"] + gcols,
    }
    for name, feats in sets.items():
        a = cv_auc(obs, feats, n_splits)
        print(f"{name:28s} {a.mean():.3f} +/- {a.std(ddof=1):.3f}")
    a = oracle_auc(obs, horizon, shape, n_splits)
    print(f"{'oracle (true hazard)':28s} {a.mean():.3f} +/- {a.std(ddof=1):.3f}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", nargs="+", default=["../data/failure_log.csv"])
    p.add_argument("--horizon", type=float, default=60.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    for path in args.inp:
        report(path, args.horizon, args.step, args.folds)


if __name__ == "__main__":
    main()