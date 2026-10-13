# PF-22 Held-Out Scenario Protocol

Status: **Selected by the PF-22 owner (Saki) on 2026-10-13.** No additional
team approval is required for this scenario-budget and seed-interval decision.
This selection is not authorization to execute held-out evaluation yet.

Prepared on 2026-10-13. This records the owner-selected evaluation budget and seed
recipe; it does not change PF-20 settings, PF-21 selection or trained weights.
The Project Bible page 14 proposes 500 scenarios as an adjustable experimental
budget. Page 30 mandates matched conditions, every final replica, downside
metrics, raw records and labelled uncertainty, not an unconditional count of 500.

## Exact Manifest

| Field | Value |
|---|---|
| Manifest | `configs/splits/final_test_pf22.json` |
| Split / owner | `final_test` / Saki |
| Seed-selection rule | Every integer from 3001 through 3500 inclusive, ascending, with no skips or replacement seeds |
| Scenario count | 500 unique roots |
| Environment reference | `../env_frozen.yaml`, relative to the manifest |
| Environment hash | `57ce865515268c2e271ab34776c6095276a48e5cd71736399732092fe73d7330` |
| Scenario recipe version | `packfolio-scenarios-v1` |
| Horizon | 100 steps |
| Initial cash / capacity / selling fee | 100 simulation units / 10 packs / 0.05 |
| Manifest SHA-256 | `f6b92a7b798d4e55a4f73dd75d66285cee7e2ae41d08989b74f68cccdc3adf34` |

The digest is SHA-256 of the actual UTF-8 manifest bytes, including whitespace
and final newline. It is separate from the environment's canonical config hash.
Changing manifest bytes requires recalculating and reviewing its digest.
Protocol status is outside the JSON because the existing strict split schema
does not permit additional approval fields. The loader does not enforce approval.

`configs/splits/final_test.json` is unchanged, including its original roots
3001, 3002 and 3003. They remain the first three entries in the selected manifest
and retain exactly the same scenario identities. Its byte digest remains
`c0c1ab147760871f11476115e8e301d4b6ee9678deccf98601862f17ead1d732`.
The new manifest is an extension of the starter test set, not an independent
second test set. Their deliberate overlap is not non-test contamination.

## Local Seed Audit

No selected seed overlapped the following locally available non-test records:

| Record category | Coverage | Overlap with 3001-3500 |
|---|---|---|
| Declared training manifest | 10 roots, 1001-1010 | None |
| Declared validation manifest | 3 roots, 2001-2003 | None |
| Recorded training runs | Six `run_manifest.json` files: five PF-21 replicas and the PF-19 pilot; 120 distinct `environment_episode_seeds` | None |
| Recorded training metrics | Six `training_metrics.jsonl` files; matching 120 distinct `environment_seed` values | None |
| PF-19 development/reload manifests | Two comparison manifests and two DQN evaluation manifests; roots 2001-2003 | None |
| Expected PF-21 tuning training stream | 60 reconstructed roots: 20 episodes for each tuning seed 20261010-20261012 | None |
| Expected PF-17 fixture training stream | 250 reconstructed roots for training seed 20261007 and the documented 1000-step / 4-step-horizon smoke budget | None |
| Documented PF-17 diagnostic rollouts | 20 reconstructed roots from rollout RNG seed 123 | None |

Recorded runs and metrics were checked under both `runs/` and ignored
`outputs/`. Other local artifact locations were inspected; no additional tuning
records were found. Seed fields and scenario identities were used for the audit,
not test performance. Static development/test seed declarations were also
reviewed; no explicit non-test scenario root in the selected interval was found.

Training RNG seeds are not scenario roots. To reconstruct the expected training
roots, the current trainer's recipe was followed: take child index 1 of
`numpy.random.SeedSequence(training_seed).spawn(4)`, obtain its uint64 seed,
construct `numpy.random.default_rng`, then draw the recorded number of uint64
episode seeds. This reproduces all six available run manifests exactly.
The diagnostic rollout instead uses `default_rng(123)` directly for 20 roots.
Only seed integers were reconstructed; no market or pack draws were generated.

**Coverage limitation:** `runs/pf21-tuning/tuning_results.json` and the original
candidate-run metadata are absent locally. The 60 tuning roots above are
expected roots based on the documented seeds, budget and deterministic trainer,
not independently recovered original records. The same distinction applies to
unavailable PF-17 run logs. Obtain missing records or explicit owner confirmation
before claiming a complete historical exposure audit. Unrecorded experiments
and transient test runs outside this repository cannot be ruled out.

The existing schema loader accepted the new manifest. Declared training,
validation and selected final splits passed `validate_split_manifests`; all
500 entries are unique and the frozen environment hash matches. No scenario
was reset, advanced, opened or evaluated during this preparation.

## Evaluation Constraints And Approval

Once evaluation is separately authorized, evaluate all five existing PF-21 replicas (training seeds
20261013-20261017) and CASH_ONLY, BUY_AND_HOLD, ALWAYS_OPEN and EV_ONE_STEP
on this exact manifest. Retain the 2000-step models and 100-step horizon.
Run each baseline once per scenario and each DQN replica once per scenario:
4500 distinct episode records, representing 500 shared scenario roots rather
than 4500 independent market worlds. Retain EV's model-informed classification.
There must be no training, tuning, checkpoint selection or seed replacement
based on held-out results.

The PF-22 owner has selected this interval and budget after the available
seed-disjointness, schema and compatibility checks passed. No additional team
approval is required for that decision. The coverage limitation above remains:
selection does not establish complete historical non-exposure where records
are missing.

Before execution, confirm the frozen environment and PF-21 model selection,
and predeclare PF-22 metrics, pairing and uncertainty rules. Existing
PF-20/PF-21 documents still mark their approvals pending; this owner decision
does not silently approve or rewrite those separate records. The restrictions
against running held-out episodes, inspecting test outcomes, training or
changing models remain in force until execution is separately authorized.
Any newly discovered non-test overlap or missing/invalid schema field is a
stop condition, not permission to choose substitute seeds.

Recheck the manifest digest without generating scenarios:

```powershell
Get-FileHash -Algorithm SHA256 configs/splits/final_test_pf22.json
```
