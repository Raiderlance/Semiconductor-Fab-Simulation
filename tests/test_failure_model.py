import pandas as pd

from train_failure_model import build_observations


def test_observations_per_tool_age_and_label():
    df = pd.DataFrame([{
        "seed": 1, "group": "wet_etch", "tool": 0,
        "down_at": 100.0, "up_at": 110.0, "down_for": 10.0,
        "time_since_last_repair": 100.0, "is_first_failure": 1,
        "run_length": 300.0, "failure_shape": 2.5,
    }])
    obs = build_observations(df, horizon=60.0, step=10.0)
    o = obs[(obs["group"] == "wet_etch") & (obs["tool"] == 0)].set_index("t")

    assert o.loc[50.0, "age"] == 50.0 and o.loc[50.0, "label"] == 1
    assert o.loc[30.0, "label"] == 0          # failure is 70 away, horizon 60
    assert o.loc[40.0, "label"] == 1          # exactly 60 away counts
    assert 100.0 not in o.index               # tool is in repair
    assert o.loc[110.0, "age"] == 0.0         # age restarts at repair
    assert o.loc[110.0, "n_failures_so_far"] == 1
    assert o.loc[110.0, "label"] == 0         # no later failure
    assert o.index.max() == 240.0             # run_length - horizon, not last failure