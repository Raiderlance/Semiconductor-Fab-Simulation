import json

from fab import Fab, FabConfig, MACHINE_DEFAULTS
from maintenance import MaintenanceConfig


def _fab(maint=None, log=False, noise=0.15, seed=3, run_length=1200.0):
    return Fab(FabConfig(num_lots=100000, seed=seed, run_length=run_length,
                         arrival_mean=7.5, maintenance=maint,
                         sensor_noise=noise, log_sensor=log))


def _pms(fab):
    return [e for g in fab.groups.values() for e in g.pm_log]


def test_sensor_logging_does_not_change_failures():
    a, b = _fab(log=False), _fab(log=True)
    a.run()
    b.run()
    fa = [(r["group"], r["tool"], round(r["down_at"], 3)) for r in a.failure_log()]
    fb = [(r["group"], r["tool"], round(r["down_at"], 3)) for r in b.failure_log()]
    assert fa == fb


def test_sensor_is_age_over_life_when_noise_is_zero():
    fab = _fab(log=True, noise=0.0)
    fab.run()
    rows = [r for g in fab.groups.values() for r in g.sensor_log]
    assert rows
    for r in rows:
        life = r["age"] + r["remaining"]
        assert abs(r["sensor"] - r["age"] / life) < 1e-9


def test_perfect_policy_prevents_most_failures():
    base = _fab()
    perfect = _fab(MaintenanceConfig(policy="perfect", perfect_lead=15.0,
                                     pm_factor=0.5))
    base.run()
    perfect.run()
    assert len(perfect.failure_log()) < 0.5 * len(base.failure_log())
    assert _pms(perfect)


def test_sensor_policy_reads_table(tmp_path):
    n_s = 51
    row = [[1.0 if -0.4 + 0.04 * j >= 0.78 else 0.0 for j in range(n_s)]
           for _ in range(201)]
    table = {g: row for g in MACHINE_DEFAULTS}
    path = tmp_path / "sensor.json"
    path.write_text(json.dumps({"age_step": 2.0, "s_min": -0.4, "s_step": 0.04,
                                "table": table}))
    fab = _fab(MaintenanceConfig(policy="sensor", risk_threshold=0.5,
                                 table_path=str(path)))
    fab.run()
    assert _pms(fab)