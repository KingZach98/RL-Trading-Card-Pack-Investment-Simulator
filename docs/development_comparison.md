# PF-19 Development Comparison

This is integration evidence, not final-test evidence or a final trained model.
Project Bible page 29 and GitHub issue PF-19 define the requirements: compare
four baselines and one reloaded DQN, inspect traces and accounting, generate a
table from saved episodes, and measure runtime. The three development seeds,
2,000-step pilot and four-file implementation are local implementation choices,
not additional Bible requirements.

## Run

From the repository root, with the locked Python 3.12 environment:

```powershell
.\.venv\Scripts\python.exe -B -m packfolio.compare --output outputs/pf19-development
```

Without a checkpoint, this trains exactly one pilot using the unchanged
`configs/agent.yaml` and the environment referenced by the validation manifest.
The pilot checkpoint, training configuration copies, metrics and manifest are
stored below the comparison directory. Evaluation reloads the saved checkpoint;
it never evaluates an in-memory training policy. There is no tuning or model
selection. A poor result is valid development evidence.

To reuse a compatible saved checkpoint instead:

```powershell
.\.venv\Scripts\python.exe -B -m packfolio.compare --checkpoint outputs/pilot/checkpoint.pt --output outputs/pf19-reloaded
```

The default manifest is `configs/splits/validation.json`, seeds 2001, 2002,
2003. Non-validation splits, including final test, are rejected. All strategies
use the same hash-matching configuration, initial cash, zero inventory,
capacity 10, NORMAL-ask normalization, fees, one-action limit and horizon.
Horizon remains 100; the team's 52-versus-100 decision is still pending.
No shared configuration is changed or frozen by this workflow.

## Saved Evidence

- `comparison.csv`: one row per policy/scenario, including all shared episode
  fields, information class, raw episode filename and full trace filename.
- `comparison_manifest.json`: development label, checkpoint content hash,
  configuration, scenario recipe version, package version, commit, settings
  and explicit episode links.
- Policy subdirectories: unchanged evaluation JSONL and full step JSONL.
- DQN subdirectory: existing evaluation manifest and settings provenance.
- EV episode directories: existing `policy_metadata.json` companions.
- `runtime_budget.json`: measured training (when performed), per-strategy
  evaluation, saved-file validation/table generation and total workflow times.
  Includes actual evaluation steps and separate validation replay steps.

The output directory must be new and below ignored `outputs/`. Runtime values
are wall-clock measurements for this machine, including stage-specific I/O;
they are development budget observations, not guarantees for larger runs.

The table is generated only after reading saved episode files with shared
schema readers. Every requested action is replayed through a fresh environment
with the same scenario seed. Every saved step must match the resulting full
StepInfo, including cash, inventory, opening draws, fees and terminal state.
Additional checks cover reward telescoping, horizon, liquidation to cash,
requested/executed counts, infeasible counts, complete unique pairing,
model/scenario provenance and matching market paths across strategies.
Recorded purchases, gross sales and terminal sales are also independently
reconciled against cash changes, a single fee and net portfolio values.
These are assertions, not a replacement accounting layer. No policy rerun
is used for reporting.

To regenerate the table and revalidate existing raw evidence:

```python
from packfolio.compare import build_comparison
build_comparison("outputs/pf19-development")
```

Keep the referenced checkpoint and split/config files available. Validation
rejects changed checkpoint content or configuration identity. This is not a
self-contained release archive; that broader work is outside PF-19.

## Interpretation Limits

EV_ONE_STEP is **model-informed**, not an equally uninformed learner or an
optimal-policy certificate. Its table label is validated against the saved
PF-13 metadata; retain those companions with the raw episode files.

PF-12 and PF-13 intentionally retain their known float32 affordability
ambiguity. Observationally eligible BUY requests can be rejected by strict
float64 accounting and executed as HOLD. Requested/executed counts and
infeasible counts are preserved rather than hidden. This convention remains
subject to reviewer/team approval, not resolved by successful integration tests.

Only one training seed and three development scenarios are used. Results
cannot support final performance, uncertainty or training-variability claims.
PF-20 freezing, PF-21 tuning and final-test evaluation are out of scope.

## Observed Development Pilot

Measured on 2026-10-08 with Python 3.12.11, locked NumPy 2.2.6 and
PyTorch 2.7.1 CPU. The unchanged training configuration produced one pilot:
training seed 20261005, 2,000 actual environment steps, 20 training episodes.
Its recorded training episode seeds do not overlap development seeds
2001-2003. No hyperparameters or checkpoint were selected using these results.

Raw evidence is under `outputs/pf19-development/`; a separate reload using
the same checkpoint is under `outputs/pf19-development-reloaded/`.
Both saved-file-derived comparison tables are byte-identical. Initial cash
is 100 simulation units; displayed values below are rounded to three decimals.

| Policy | Seed 2001 | Seed 2002 | Seed 2003 | Development Mean |
|---|---:|---:|---:|---:|
| CASH_ONLY | 100.000 | 100.000 | 100.000 | 100.000 |
| BUY_AND_HOLD | 97.000 | 114.000 | 116.000 | 109.000 |
| ALWAYS_OPEN | 448.875 | 299.000 | 590.563 | 446.146 |
| EV_ONE_STEP (model-informed) | 102.725 | 123.613 | 161.525 | 129.288 |
| DQN (development pilot) | 75.500 | 191.550 | 377.950 | 215.000 |

Each cell comes from the matching policy/scenario row in `comparison.csv`;
that row links its raw episode and full trace. The mean is the arithmetic
mean of the three saved terminal values, not an uncertainty estimate.

All 15 episodes ran 100 decisions and finished with zero sealed inventory.
Full-trace replay, independent cash/fee/value reconciliation and matched
market-path checks passed. CASH_ONLY held throughout. BUY_AND_HOLD bought
10 packs, then held until environment liquidation. ALWAYS_OPEN alternated
50 buys and 50 openings on each scenario. EV_ONE_STEP made 11, 14 and 16
buy/open cycles respectively. DQN executed 20/25/30 buys and 13/17/25 opens;
38/21/21 additional BUY requests were infeasible and converted to HOLD.
Baseline infeasible counts were zero on these scenarios, which does not
resolve the known float32 boundary ambiguity. No accounting defect was found.

### Measured Runtime Budget

These are observed wall-clock times, not a larger-experiment commitment.
The pilot and reload runs overlapped test execution on this machine; timing
is therefore indicative and should be remeasured before assigning a large budget.

- Initial workflow: 14.045 s, including 12.927 s for the training stage
  (setup, training and saving). The trainer's narrower loop measurement
  in `run_manifest.json` is 5.754 s; these measure different scopes.
- Separate reload workflow: 3.547 s total, with DQN evaluation 0.213 s,
  CASH_ONLY 0.048 s, BUY_AND_HOLD 0.052 s, ALWAYS_OPEN 0.094 s and
  EV_ONE_STEP 0.100 s. Saved-file validation/table generation took 0.581 s.
- Each comparison performs 1,500 evaluation steps and 1,500 validation
  replay steps. Initial training adds 2,000 steps. Import/startup before
  entering the workflow is not included; stage timings are not exhaustive
  and need not sum to the total.

Exact measurements remain in each `runtime_budget.json`. Reusing this
pilot avoids repeating training; this is budget evidence only, not tuning.

### Verification

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_compare.py tests/test_compare_integration.py
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B -m packfolio.validate
.\.venv\Scripts\python.exe -B -m packfolio.check_env
```

Results: 38 focused tests passed; 639 full-suite tests passed; 14/14 Monte
Carlo diagnostics passed; Gymnasium/PyTorch boundary checks passed for
100 steps with deterministic replay. No skips or missing dependencies.
The compatibility command uses an untrained test network, not the pilot.
One new corruption fixture initially failed because the shared schema
rejected it earlier than expected; it was corrected to be schema-valid but
trace-inconsistent. No production failure or teammate-code change resulted.
