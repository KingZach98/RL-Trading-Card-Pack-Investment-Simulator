# PF-21: hyperparameter comparison and final replicas

**Purpose:** select an agent configuration using validation evidence rather
than a lucky run, and retain genuinely independent final replicas of the
selected configuration. This note states the predetermined candidate grid,
training budget, seed lists, and selection rule **before** presenting the
actual results, per the PF-21 handoff from PF-20's frozen base environment
(`configs/environment.json`, `config_hash`
`57ce865515268c2e271ab34776c6095276a48e5cd71736399732092fe73d7330`).

All work in this note uses `src/packfolio/tune.py`
(`packfolio.tune.run_tuning` / `packfolio.tune.train_final_replicas`), which
only orchestrates repeated calls to the existing `packfolio.train.train` and
`packfolio.evaluate.evaluate_checkpoint_on_split`; it changes no training,
environment, or evaluation semantics.

## Predetermined before running

These were fixed in `packfolio/tune.py` before any candidate was trained or
evaluated, and must not be edited after seeing validation results — doing so
would defeat the purpose of a predetermined comparison (see PF-20's freeze
rationale in `docs/assumptions.md`).

**Candidate grid** (`TUNING_CANDIDATES`), one axis at a time against the
`configs/agent.yaml` baseline, holding every other hyperparameter and the
training budget fixed:

| Candidate | Override | Axis |
| --- | --- | --- |
| `baseline` | — | (reference) |
| `low_learning_rate` | `learning_rate = 0.0003` | learning rate |
| `high_learning_rate` | `learning_rate = 0.003` | learning rate |
| `slow_exploration_decay` | `epsilon_decay_steps = 3000` | exploration decay |
| `fast_exploration_decay` | `epsilon_decay_steps = 750` | exploration decay |

**Training budget:** `total_steps = 2000` for every candidate (the baseline's
value). `run_tuning` raises before training anything if a candidate's
overrides would change `total_steps`; reducing the shared budget requires its
own recorded decision, not a silent per-candidate change.

**Tuning seeds** (`TUNING_SEEDS`, shared across every candidate):
`20261010`, `20261011`, `20261012`. `train()` derives network initialization,
exploration draws, and the entire training-scenario stream deterministically
from a single `AgentConfig.seed`, so reusing the same seed across candidates
pairs each candidate's seed-*N* run against every other candidate's seed-*N*
run on matched initial weights and matched training scenarios — isolating the
hyperparameter effect instead of confounding it with seed variation.

**Final replica seeds** (`FINAL_REPLICA_SEEDS`, disjoint from the tuning
seeds — enforced by an assertion at module import time):
`20261013`, `20261014`, `20261015`, `20261016`, `20261017`. Final replicas
must be new, independent training runs, not a re-use of whichever tuning
seed happened to score best.

**Selection metric and rule** (`SELECTION_METRIC`, `SELECTION_RULE` in
`packfolio/tune.py`): the mean, over validation scenarios, of each episode's
`final_portfolio_value` (`packfolio.evaluate.EvaluationRow`, the same metric
PF-19 used for its development comparison). Among candidates with at least
one successful training seed, select the candidate whose mean-across-seeds
validation score is highest; ties would break by declaration order in
`TUNING_CANDIDATES`. Selection uses only
`configs/splits/validation.json` (seeds 2001–2003, owner Nasir); the module's
`_load_validation_split` helper asserts the loaded manifest's `split` field is
`"validation"` and raises clearly otherwise, so the tuning step cannot be
pointed at `configs/splits/final_test.json` (owner Saki) even by a path
mistake. **`final_test` was not read at any point during this work.**

## How it was run

```powershell
.\.venv\Scripts\python.exe -m packfolio.tune
```

This runs `run_tuning()` (5 candidates × 3 seeds = 15 training + validation
runs against the frozen environment) followed by `train_final_replicas()`
(5 independent training runs of the selected candidate). Every individual
run is wrapped in a `try`/`except`; a failure is recorded with its error
message and full traceback rather than crashing the batch, satisfying "save
all configurations, outcomes and failures." Full machine-readable output:

- `runs/pf21-tuning/tuning_results.json` — every candidate/seed outcome,
  the per-candidate mean validation scores, and the selection.
- `runs/pf21-final-replicas/final_replicas_manifest.json` — every final
  replica outcome (all retained, not just the best).

## Results

All 15 tuning runs and all 5 final-replica runs completed with
`status: "success"`; no failures were observed in this run.

| Candidate | seed 20261010 | seed 20261011 | seed 20261012 | mean |
| --- | --- | --- | --- | --- |
| `baseline` | 300.30 | 413.74 | 299.12 | 337.72 |
| `low_learning_rate` | 385.52 | 292.78 | 347.00 | **341.77** |
| `high_learning_rate` | 109.00 | 421.37 | 220.80 | 250.39 |
| `slow_exploration_decay` | 124.04 | 491.06 | 183.13 | 266.08 |
| `fast_exploration_decay` | 428.62 | 295.34 | 166.57 | 296.84 |

(Values are mean `final_portfolio_value` across the validation split's three
scenarios for that candidate/seed's checkpoint.)

**Selected candidate: `low_learning_rate`** (`learning_rate = 0.0003`,
mean validation score 341.77), per the predetermined rule above. The spread
across seeds within every candidate (for example `high_learning_rate` ranges
from 109.0 to 421.4) is at least as large as the spread between candidate
means, so this selection should be read as "best available evidence from
three seeds each," not as a sharp, low-variance winner — consistent with the
ticket's caution to compare against predetermined fixture expectations and
not over-read a small budget.

The selected hyperparameters are written out as
[`configs/agent_selected.yaml`](../configs/agent_selected.yaml): the
`configs/agent.yaml` baseline with `learning_rate` overridden to `0.0003`
and every other hyperparameter, including `total_steps`, unchanged.

### Final replicas

All five final replicas of `low_learning_rate` trained successfully and are
all retained in `runs/pf21-final-replicas/final_replicas_manifest.json`
(none were culled for scoring lower than another replica):

| Seed | Status | Run directory |
| --- | --- | --- |
| 20261013 | success | `runs/pf21-final-replicas/20261010T015008.025131Z-1854d265` |
| 20261014 | success | `runs/pf21-final-replicas/20261010T015010.102590Z-a3b92e2d` |
| 20261015 | success | `runs/pf21-final-replicas/20261010T015012.273007Z-633d28f0` |
| 20261016 | success | `runs/pf21-final-replicas/20261010T015014.421835Z-40844f94` |
| 20261017 | success | `runs/pf21-final-replicas/20261010T015016.590182Z-b79ff8f6` |

Each replica's directory contains its own manifest (model, resolved
configuration, commit, and training scenario stream), matching the PF-18
checkpoint/manifest format; see
[`docs/development_comparison.md`](development_comparison.md) for how PF-19
evaluated an earlier checkpoint through that same reload path.

## Status and provenance

The candidate grid, seed lists, and selection rule in this note were
designed by the assistant during this session — no pre-existing
"team-approved" candidate set existed in the repository before PF-21. As
with PF-20's freeze, this comparison should be read as **proposed, pending
team review**, not as a human-approved selection. Re-running
`python -m packfolio.tune` (same candidates, same two seed lists, same
frozen environment and validation split) will reproduce the same checkpoints
and the same selection, since every random draw involved is a deterministic
function of the recorded seeds.
