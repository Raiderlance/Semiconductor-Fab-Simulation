import json

from fab import Fab, FabConfig, MACHINE_DEFAULTS
from maintenance import MaintenanceConfig


def _fab(maint, seed=1, run_length=1500.0):
    return Fab(FabConfig(num_lots=100000, seed=seed, run_length=run_length,
                         arrival_mean=7.5, maintenance=maint))


def _pms(fab):
    return [e for g in fab.groups.values() for e in g.pm_log]


def test_policy_none_matches_no_policy():
    a = _fab(None)
    b = _fab(MaintenanceConfig(policy="none"))
    assert a.run() == b.run()
    assert a.failure_log() == b.failure_log()


def test_age_policy_triggers_pm_after_threshold():
    fab = _fab(MaintenanceConfig(policy="age", age_frac=0.5, pm_factor=0.5))
    fab.run()
    pms = _pms(fab)
    assert pms
    for e in pms:
        assert e["age"] >= 0.5 * MACHINE_DEFAULTS[e["group"]]["mtbf"] - 1e-9
    assert sum(g.pm_time for g in fab.groups.values()) > 0


def test_risk_policy_reads_table(tmp_path):
    table = {g: [min(1.0, a / 200.0) for a in range(0, 501)] for g in MACHINE_DEFAULTS}
    path = tmp_path / "risk.json"
    path.write_text(json.dumps({"grid_step": 1.0, "table": table}))
    fab = _fab(MaintenanceConfig(policy="risk", risk_threshold=0.5,
                                 table_path=str(path)))
    fab.run()
    pms = _pms(fab)
    assert pms
    assert all(e["age"] >= 100.0 - 1e-9 for e in pms)