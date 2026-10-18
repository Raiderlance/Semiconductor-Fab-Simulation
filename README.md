# AI-Driven Semiconductor Fab Simulation

A simulation of a simplified chip factory, used to ask two questions:

1. **Scheduling:** when several wafer lots are waiting for a machine, does the order we pick them in matter?
2. **Maintenance:** if we predict which machines are about to break and service them first, does the factory run better?

This is a portfolio project. It does not model semiconductor physics and is not a replica of any commercial product. It models queues, limited machines, breakdowns and decisions.

## Key findings

- **Scheduling moves waiting around; it does not create capacity.** Over 30 seeds, priority dispatch makes high-priority lots about 20 units faster and normal lots about 5 slower, with no change in the average. SRPT has the lowest average cycle time but leaves some lots waiting four times longer than under FIFO. When lithography is saturated, no rule changes throughput.
- **Predicting failures helps only if the prediction is precise and servicing is cheap.** A rule that services machines at a fixed age shows no measurable benefit when a service takes half a repair time. A model that reads a wear sensor cuts bottleneck downtime from 8.3% to 6.2% (service = 0.5 x repair time) and to 2.7% (service = 0.2 x repair time), on seeds it was never trained or tuned on.
- **The sensor is synthetic and built to be informative.** This is a result about the simulator, not about real fabs. See [Limitations](#limitations).

## Plain-language glossary

| Term | Meaning |
|---|---|
| Fab | A semiconductor factory |
| Wafer lot | A batch of wafers moving through the factory together |
| Stage / tool | A processing step (e.g. lithography) and the machines that do it |
| Bottleneck | The stage that limits the whole factory. Here, lithography (1 machine) |
| Throughput | Lots finished per unit time |
| Cycle time | Time a lot spends in the factory. Lower is better |
| WIP | Work in progress: lots arrived but not finished |
| Dispatch policy | The rule for choosing which waiting lot goes next (FIFO, priority, EDD, SRPT) |
| MTBF / MTTR | Mean time between failures / mean time to repair |
| PM | Preventive maintenance: taking a machine down on purpose before it fails |
| PM cost | In this project, how *long* a PM takes as a fraction of a repair (0.5 = half). Not money |
| Weibull | A failure distribution where machines wear out (shape > 1: risk rises with age) |
| AUC | How well a model ranks risky cases above safe ones. 0.5 = guessing, 1.0 = perfect |
| Seed | A random-number starting point. Different seeds give different but reproducible runs |

## How it works

```
wafer lot --> [ stage queue ] --> dispatcher --> free tool --> next stage
```

- **Route:** wet_etch (2 tools) -> lithography (1) -> deposition (2) -> etch (2) -> inspection (1).
- **Dispatch:** each stage waits for a free tool, then asks the active policy which waiting lot to run. Every tool is its own capacity-1 SimPy resource, so one tool can fail without affecting its siblings.
- **Fair comparisons:** each lot's work content and due date use their own random stream, so every policy sees identical lots.
- **Failures:** each tool has a hidden Weibull lifetime (shape 2.5, so it wears out), then an exponential repair time.
- **Wear sensor (stage two):** at each 5-unit check, a tool reads `age / lifetime + noise` (noise sd 0.15), a noisy measure of the fraction of life used.
- **Preventive maintenance:** a policy is asked every 5 units whether each running tool should go down for PM. A PM resets the tool's age. PM applies to all five stages.

### Dispatch policies

| Policy | Rule |
|---|---|
| `fifo` | Earliest arrival first |
| `priority` | Higher lot priority first |
| `edd` | Earliest due date first |
| `srpt` | Shortest remaining processing time first |

Add a policy by decorating a function with `@register_policy("name")` in `src/scheduling.py`.

### Maintenance policies

| Policy | Services a tool when... |
|---|---|
| `none` (reactive) | never; it is repaired only after failing |
| `age` | its age reaches a fraction of the group's MTBF |
| `risk` | a learned P(fail soon given age, group) reaches a threshold (stage one) |
| `oracle` | the true age-based hazard reaches a threshold (stage one benchmark) |
| `sensor` | a learned P(fail within 30 units given age, sensor, group) reaches a threshold (stage two) |
| `perfect` | its true remaining life is 15 units or less (stage two benchmark; untuned) |

## Run it yourself

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                # from the project root

cd src
python fab.py --all --lots 50            # quick demo of all dispatch policies
```

All scripts below are run from `src/`.

**Scheduling study (30 seeds):**
```bash
python run_experiments.py --seeds 30   # writes ../data/policy_due_study.csv
```

**Stage one: age-based failure model and maintenance:**
```bash
python export_failure_log.py --seeds 50 --failure-shape 2.5 --out ../data/failure_v2_weibull.csv
python export_failure_log.py --seeds 50 --failure-shape 1.0 --out ../data/failure_v2_exp.csv
python train_failure_model.py --in ../data/failure_v2_weibull.csv ../data/failure_v2_exp.csv
python build_risk_table.py --in ../data/failure_v2_weibull.csv --out ../data/risk_table.json
python run_maintenance.py --seeds 15 --seed0 2000 --out ../data/maint_tune.csv
python run_maintenance.py --seeds 30 --seed0 1000 --scenarios reactive age:0.9 risk:0.5 oracle:0.7 --out ../data/maint_eval.csv
```

**Stage two: wear sensor:**
```bash
python export_sensor_log.py --seeds 50 --out ../data/sensor_log.csv
python train_sensor_model.py --in ../data/sensor_log.csv --out ../data/sensor_table.json
python run_stage2.py --out ../data/stage2.csv
```

Seed ranges never overlap: training 42+, evaluation 1000+, tuning 2000+ (each 10 apart so runs do not share failure random streams).

`configs/base_config.yaml` documents simulation defaults and what-if scenarios, but the scripts do not read it yet. Settings come from the command line and from `FabConfig` in `src/fab.py`.

## Results

### 1. Scheduling

**Main study: 30 seeds, realistic load.** `python run_experiments.py --seeds 30` (arrival mean 7.5, lithography about 79% loaded, due date = 5 x the lot's work, high-priority lots get half the slack). Metrics cover lots that arrive after a 500-unit warm-up. All lots finish. Differences are paired against FIFO, with 95% CIs.

| Policy | Overall cycle time | Normal lots | High-priority lots | p95 cycle | Worst normal lot | On-time |
|---|---|---|---|---|---|---|
| FIFO | 44.80 | 45.09 | 43.61 | 86.8 | 107 | 87.0% |
| EDD | -0.40 ± 0.17 | +3.30 ± 0.46 | -15.25 ± 2.03 | +5.9 ± 1.0 | +12.5 ± 3.1 | +3.7 pts ± 0.8 |
| Priority | -0.09 ± 0.11 | +4.85 ± 1.06 | -20.12 ± 5.02 | +10.1 ± 2.7 | +17.0 ± 4.7 | +3.8 pts ± 0.8 |
| SRPT | -4.30 ± 2.14 | -4.37 ± 2.10 | -3.86 ± 3.56 | +40.2 ± 12.7 | +350 ± 88 | +4.6 pts ± 3.6 |

FIFO row is absolute; other rows are differences from FIFO (negative cycle time is better).

- **Priority and EDD redistribute waiting.** High-priority lots finish about 15 to 20 units sooner, normal lots about 3 to 5 units later, and the overall average barely moves.
- **SRPT has the best average but starves long lots.** The average falls by 4.3, but the p95 rises by 40 and the worst normal lot waits about 460 units against about 110 under FIFO.
- **EDD is no longer identical to FIFO,** because due dates here depend on each lot's work. (With the old fixed-offset due dates, EDD and FIFO were nearly the same order.)

**Stress test: overloaded fab, single seed.** Seed 42, 300 lots, arrival mean 2.0 (default). Lithography is saturated, so only 54 lots finish whatever the policy. One run per policy, so no CIs.

| Policy | Throughput | On-time | Avg cycle | Normal lots (n) | Normal cycle | High-prio (n) | High-prio cycle |
|---|---|---|---|---|---|---|---|
| FIFO | 54 | 11.1% | 149.98 | 44 | 163.00 | 10 | 92.67 |
| EDD | 54 | 11.1% | 149.98 | 44 | 153.94 | 10 | 132.56 |
| SRPT | 54 | 14.8% | 112.67 | 46 | 116.18 | 8 | 92.52 |
| Priority | 54 | 57.4% | 96.85 | 24 | 181.57 | 30 | 29.07 |

When the bottleneck is saturated, scheduling cannot raise throughput. Blended averages here are not comparable across policies because different mixes of lots finish.

### 2. Stage one: predicting failures from age

5-fold cross-validation, split by seed, 60-unit horizon.

| Features | Weibull (shape 2.5) | Exponential (shape 1.0) |
|---|---|---|
| machine group only | 0.606 | 0.569 |
| age only | 0.717 | 0.509 |
| age + group | **0.775** | 0.563 |
| age + group + failures so far | 0.772 | 0.552 |
| oracle (true hazard) | 0.776 | 0.569 |

Under exponential failures a machine has no memory, so age is useless (0.509, chance). Under wear-out failures age adds about 0.17 AUC over group alone, and the model reaches the oracle's ceiling.

**Using it for maintenance** (a PM takes 0.5 x a repair; 30 held-out seeds):

| Policy | Failures | PMs | Litho down | Cycle time vs reactive |
|---|---|---|---|---|
| reactive | 300 | 0 | 8.3% | (44.78) |
| age rule (0.9 x MTBF) | 181 | 203 | 8.2% | -0.77 ± 2.80 |
| learned risk (0.5) | 64 | 721 | 8.8% | +0.24 ± 2.91 |
| oracle (0.7) | 133 | 339 | 8.0% | -1.40 ± 2.49 |

No policy is distinguishable from reactive. Learned risk prevents most failures but needs 721 PMs to do it, which costs about as much downtime as it saves.

### 3. Stage two: adding a wear sensor

AUC, 30-unit horizon, split by seed:

| Information the model sees | AUC |
|---|---|
| machine group only | 0.573 |
| age only | 0.760 |
| sensor only | 0.904 |
| age + sensor + group | 0.912 |

Maintenance results on 30 held-out seeds (thresholds tuned on 15 separate seeds for the lowest lithography downtime).

**A PM takes half a repair (0.5):**

| Policy | Failures | PMs | Litho down | Cycle time vs reactive |
|---|---|---|---|---|
| reactive | 300 | 0 | 8.3% | (44.78) |
| age rule (0.9 x MTBF) | 181 | 203 | 8.2% | -0.77 ± 2.80 |
| sensor model (0.7) | 7 | 426 | 6.2% | -4.80 ± 2.37 |
| perfect knowledge | 2 | 358 | 5.2% | -7.12 ± 2.49 |

**A PM takes a fifth of a repair (0.2):**

| Policy | Failures | PMs | Litho down | Cycle time vs reactive |
|---|---|---|---|---|
| reactive | 300 | 0 | 8.3% | (44.78) |
| age rule (0.6 x MTBF) | 110 | 423 | 6.0% | -4.59 ± 2.21 |
| sensor model (0.6) | 2 | 486 | 2.7% | -10.85 ± 2.77 |
| perfect knowledge | 2 | 367 | 2.4% | -10.85 ± 2.69 |

Notes: failure and PM counts are fab-wide. Litho down is lithography breakdown plus PM time over the whole run. Cycle time is noisy, so litho downtime is the more reliable column. Intervals are 95% CIs of the paired difference vs reactive; if the interval is wider than the number, treat it as can't tell. The sensor-vs-age gap was judged from CIs by eye, not with a paired test.

## Limitations

- The sensor is synthetic and built to be informative (`age / lifetime + noise`). Real sensors are messier.
- Noise level, check interval (5), prediction horizon (30 or 60), PM cost and the tuning grids are my choices.
- The age-rule grid (0.6, 0.75, 0.9) and the risk grid (0.5 to 0.8) had winners at their edges, so those baselines may be under-tuned.
- Failures are costed only as downtime. Scrapped lots are not modelled, which would favour prevention.
- Failure draws are not perfectly matched across maintenance policies (a PM consumes a fresh lifetime draw), so paired comparisons are somewhat noisier than they look.
- Stage one tunes on lot cycle time and stage two tunes on lithography downtime.
- In the CSV files, `mean_age` means average lot cycle time, not machine age.
- The scheduling results above are from one seed.

## Project structure

```
configs/base_config.yaml    # documented defaults and what-if scenarios (not read by the code yet)
src/
  fab.py                    # simulation engine, metrics, CLI
  machines.py               # machine groups, failures, wear sensor
  scheduling.py             # Lot dataclass, dispatch policies
  maintenance.py            # PM policies
  run_experiments.py        # multi-seed scheduling study
  export_failure_log.py     # stage one: failure data
  train_failure_model.py    # stage one: model and ablation
  build_risk_table.py       # stage one: risk lookup table
  run_maintenance.py        # stage one: maintenance study
  export_sensor_log.py      # stage two: sensor data
  train_sensor_model.py     # stage two: model and lookup table
  run_stage2.py             # stage two: maintenance study
tests/                      # pytest suite
data/                       # generated CSV/JSON (regenerable)
requirements.txt
pytest.ini
```

## Roadmap

- Widen the age and risk threshold grids and retune.
- Add a paired test for sensor vs age rule.
- Add noisier or biased sensors to test robustness.
- Cost scrapped lots in addition to downtime.
- Make the simulator read `configs/base_config.yaml` and add a what-if runner (capacity loss, demand changes) across all policies.
