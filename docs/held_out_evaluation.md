# PF-22 Held-Out Evaluation

This infrastructure implements Project Bible pages 14, 15 and 30: matched
held-out evaluation, all final replicas, downside/action metrics, per-model
paired comparisons, raw evidence and explicitly labelled uncertainty. It
does not implement PF-23 sensitivity or PF-24 figures. Implementation tests
preceded real evaluation. The PF-22 owner separately approved the real run,
which completed on 2026-10-13; any further execution requires explicit approval.

## Frozen Inputs

- Scenario manifest: `configs/splits/final_test_pf22.json`, 500 unique seeds
  3001-3500, selected by the PF-22 owner. See `held_out_scenario_protocol.md`.
- Environment: `configs/env_frozen.yaml`, horizon 100, initial cash 100,
  capacity 10, NORMAL-ask observation reference 10, selling fee 0.05.
- Configuration hash:
  `57ce865515268c2e271ab34776c6095276a48e5cd71736399732092fe73d7330`.
- Replica manifest: `runs/pf21-final-replicas/final_replicas_manifest.json`;
  all five checkpoints for training seeds 20261013-20261017.
- Selected agent configuration: `configs/agent_selected.yaml`; the existing
  2000-step weights are loaded, never retrained, extended or selected by test
  performance. Horizon, configuration and scenario budget are unchanged.

Preflight checks the final-test split, frozen hash, horizon, all five distinct
successful replicas, actual recorded training budget, selected agent settings,
checkpoint architecture/weights, capacity, normalization and overlap with
recorded model-training roots. Failure stops before creating output or running
an episode. It does not claim to reconstruct missing historical records.

## Execution

After explicit approval, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m packfolio.held_out --split-manifest configs/splits/final_test_pf22.json --replica-manifest runs/pf21-final-replicas/final_replicas_manifest.json --output results --bootstrap-seed 22 --bootstrap-resamples 2000
```

The output directory must not already exist. Do not overwrite prior evidence.
The command performs 2500 DQN episodes and 2000 baseline episodes, each with
100 decisions. These are 500 matched scenario roots, not 4500 independent
market worlds. Every baseline is evaluated once per scenario, never copied
into five nominally independent baseline datasets.

The DQN path reuses `evaluate_checkpoint_on_split`; ordinary baselines reuse
`evaluate_episode`; EV reuses `evaluate_ev_episode`. Every policy receives only
the public observation. There is no training, exploration fitting, model
selection, future-draw access or modification of shared evaluation schemas.
The runner has no training entry point.

## Outputs And Traceability

- `results/raw/`: unchanged shared episode JSONL and full step traces for all
  nine policy/model groups. EV companions retain its model-informed label;
  DQN directories retain existing evaluation manifests and settings hashes.
- `results/held_out_manifest.json`: all model identities, raw-file links,
  configuration/scenario versions, commit/dirty status, input artifact SHA-256
  digests, statistical settings and known limitations.
- `results/episodes.csv`: one row per unique policy/model/scenario, including
  training seed, scenario seed/ID, config hash, all original episode fields,
  requested and executed counts, infeasible counts, classification and source
  filenames. No numerical rounding is applied to stored CSV values.
- `results/summary.csv`: nine per-policy/per-replica rows with downside/action
  metrics and confidence intervals; each DQN row also has comparisons against
  all four baselines. There is no ranking or best-replica selection.
- `results/uncertainty.json`: formulas/settings, sampling unit, coverage
  limitations and descriptive mean/sample standard deviation of the five
  per-replica mean final values. This is not a pooled independent-market CI.

CSV generation reads saved records, not in-memory evaluation results. Before
reporting it rejects missing/duplicate/unknown scenarios, missing/duplicate
model groups, incompatible provenance/settings and malformed shared schemas.
Full traces must have the correct horizon, state continuity, strict action
feasibility, cash/inventory/fee accounting, terminal liquidation, reward
telescoping and action counts. Ledger checks reuse existing portfolio APIs
with recorded outcomes, not sampled future draws. Market regime paths must
match across all nine groups for each scenario. EV metadata is mandatory.

Regenerate summaries from saved evidence without policy predictions:

```powershell
.\.venv\Scripts\python.exe -B -m packfolio.held_out --aggregate-only --output results
```

This deliberately replaces only the derived CSV/uncertainty reports. Raw
evidence is untouched. Keep referenced input artifacts available and unchanged;
source-byte hashes are verified. Complete portable release packaging belongs
to PF-24, not this module.

## Aggregation Rules

Scenarios are sorted by full scenario ID before numerical aggregation. Every
model has exactly one observation per shared scenario; baseline records have
no training seed or model ID. For each fixed policy/model:

- Mean and median of terminal net portfolio values.
- Sample standard deviation with denominator `n-1` (`ddof=1`); undefined for
  a singleton, not silently zero.
- Loss probability: fraction with `final_value < initial_value - 1e-9` in
  simulation units. Equality and sub-tolerance roundoff are not losses.
- Fifth percentile: NumPy's explicit `method="linear"` empirical quantile.
- Each requested/executed action rate: total matching count divided by total
  decision steps. Infeasibility rate uses the same denominator.
- Per-replica paired difference against a baseline: subtract baseline final
  value from DQN final value on each matching scenario, then average.

Infeasible DQN requests are not executed trades: the environment converts them
to HOLD. Separate action-rate columns preserve this distinction. The known
PF-12/PF-13 float32 affordability ambiguity is also retained and disclosed;
the evaluator does not silently override requested actions.

## Confidence Intervals

Default: 95% percentile scenario-bootstrap intervals, 2000 resamples, independent
NumPy RNG seed 22. For each resample, draw `n` scenario indices with replacement
and calculate the mean. Take the 2.5th and 97.5th percentiles of resampled means,
using linear quantiles. For baseline comparisons, first form matched DQN-minus-
baseline differences and resample those same pairs, never each side independently.
The same index draws are used across model/baseline summaries. Bootstrap RNG
does not consume simulator streams or global NumPy RNG state.

These intervals describe **scenario uncertainty conditional on a fixed
checkpoint**. They exclude training variability, hyperparameter-selection
uncertainty and simulator/model misspecification. The across-replica descriptive
summary is kept separate. No crossed model/scenario bootstrap is claimed.
Intervals have individual nominal coverage; they are not simultaneous or
adjusted for the multiple model/baseline comparisons.
Percentile bootstrap coverage is approximate, especially for rare outcomes;
500 scenarios still give limited precision in the lower tail. Fixture runs
with fewer than 30 scenarios carry an explicit small-sample warning. More
resamples cannot compensate for too few original scenarios.

## Known Limitations And Verification

Historical tuning results and some original run records are unavailable
locally. Known recorded seed-disjointness is checked; complete historical
non-exposure is not asserted. The selected models have only 2000 training
interactions (20 full episodes), so convergence is not established. These
limitations are recorded, not repaired through retraining or test-based
selection. Existing PF-20/PF-21 approval-status documentation is not rewritten.

New integration tests use temporary two-scenario manifests and handcrafted,
untrained checkpoints. Their metadata are synthetic fixtures, not claims of
2000 actual training steps. They never load the real held-out manifest to
execute scenarios or mutate the real model bundle. Unit tests independently
check metric arithmetic and paired/bootstrap calculations.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_held_out.py tests/test_held_out_integration.py
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
```

Implementation verification on Python 3.12: 42 focused tests passed; 708
full-suite tests passed. Real-input metadata-only preflight accepted all five
replicas and 500 scenario recipes; no real environment reset or policy
prediction was performed. The selected and starter manifest digests are unchanged.

## Completed Run And Git Publication

The separately approved real run completed successfully: 4500 unique episodes,
2500 DQN and 2000 baseline, with nine groups covering the identical 500 roots.
All 450000 saved steps and required metrics were validated. Every checkpoint,
the frozen configuration and both scenario manifests retained their input
digests. All five models are reported; no model was selected using these results.
The focused tests were rerun before publication: 42 passed.

Git publication includes the implementation, tests, selected manifest, protocol
documentation and four compact result files: `episodes.csv`, `summary.csv`,
`uncertainty.json` and `held_out_manifest.json` (2777876 bytes combined).
The root `.gitattributes` rules apply only to this PF-22 manifest and these
four results, preventing line-ending conversion from changing recorded bytes.
No shared simulator, PF-20/PF-21 implementation or model weights are changed.

`results/raw/` contains 5519 files totaling 216262465 bytes and is deliberately
not staged. It remains intact locally. A separately transferable archive of
the complete `results/` tree, including the four compact files, is retained at
`outputs/pf22-publication/pf22-results.zip` under the existing ignored output
directory. The archive contains 5523 files, is 10746921 bytes, and passed ZIP
CRC, filename and size checks. Its SHA-256 is
`732723093507f3d8e09ab4aaa353e66acee4b3f26cdefffadf3b022a431b3df1`.
It has not been uploaded; transfer it separately for full-trace inspection.
Do not use broad `git add results/` or `git add -A` to publish raw traces or
the untracked Project Bible PDF accidentally.

The published episode CSV retains all shared raw episode fields and links to
the separately archived full evidence. A Git checkout alone provides the
tables, not the raw traces required by `--aggregate-only`. Restore the archive
and retain the referenced inputs to inspect/reaggregate the original run.
Historical absolute input paths in the run manifest remain unchanged; it is
not a fully portable release package. Portable release packaging is outside
this PF-22 publication.

Evaluation provenance honestly records commit
`b2ec4f176f4370745f099e377a99b547e02924f1` and a dirty worktree because the
PF-22 implementation was uncommitted when evaluated. The publication commit
does not rewrite those historical fields to claim a clean run.
