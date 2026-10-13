# PF-23 Fee Sensitivity

## Protocol

Evaluate the predeclared 2.5%, 5% and 10% selling fees with the same 500
scenario seeds (3001-3500), 100-step horizon, capacity 10 and reference price 10.
Only the selling fee differs. All five PF-21 DQN checkpoints remain frozen:
no exploration, retraining, tuning or selection using held-out outcomes.
The three rule baselines are unchanged. EV_ONE_STEP is model-informed and
recomputes analytical decisions with each variant's published fee.

The adapter uses the genuine frozen 5% configuration and scenario identity
for market and pack randomness, while accounting uses the actual variant fee.
Ordinary variant environments would use different random streams because the
fee changes the configuration hash. Both identities are recorded explicitly in
the PF-23 manifest; existing simulator hashes and replay rules are unchanged.

## Evidence And Commands

The 5% reference is read from PF-22, not rerun or copied. Preflight verifies
its source digests, models, 500 scenario identities, raw accounting, matching
market paths, coincident pack draws, EV labels and episode CSV consistency.
The adapter's 5% equivalence is tested against PackfolioEnv with scripted
actions, exact observation bytes and every StepInfo field.

The corrected experiment completed successfully using:

```powershell
.\.venv\Scripts\python.exe -B -m packfolio.fee_sensitivity --output outputs/pf23-corrected --summary results/fee_sensitivity_corrected.csv
```

Authoritative evidence is `outputs/pf23-corrected/episodes.jsonl.gz` and its
unchanged `manifest.json`. The official `results/fee_sensitivity.csv` is a
byte-identical copy of `results/fee_sensitivity_corrected.csv`.
`results/fee_sensitivity_manifest.json` links the official CSV to that run and
the archive by SHA-256. The run retains 9,000 new episodes / 900,000 steps;
the 4,500 original 5% episodes are reused. All 27 groups cover seeds 3001-3500.
Evaluation and recording took 274.15 seconds, excluding preflight/aggregation.
Recorded provenance remains commit `76301a4` with a dirty tree: PF-23 code was
not yet committed during execution. Publication does not rewrite that history.

The first run omitted the configured baseline reference price. Its invalid
CSV is preserved at `outputs/pf23/fee_sensitivity_failed.csv`, alongside its
original episodes and manifest. It is excluded from published results.

Regenerate the table from saved files without running any policy:

```powershell
.\.venv\Scripts\python.exe -B -m packfolio.fee_sensitivity --aggregate-only --output outputs/pf23-corrected
Copy-Item -LiteralPath results/fee_sensitivity_corrected.csv -Destination results/fee_sensitivity.csv
```

Aggregation validates all expected fee/model/seed combinations and full traces
before writing the 27 strategy-by-fee summaries. Each baseline runs once per
scenario at each new fee; the 5% baseline is reused once, not replicated.

The smallest reviewer handoff is the Git deliverables plus the separately
distributed `outputs/pf23-publication/pf23-evidence.zip` (24.63 MB). It contains
the corrected compressed episodes (16.72 MB), original run manifest, corrected
CSV and verified PF-22 reference archive. No failed-run evidence is included.
Archive SHA-256 and member hashes are recorded in the publication manifest.
Its four members passed CRC and byte-for-byte checks. A Git checkout alone
does not contain the raw evidence needed for aggregation. Preserve both the
archive and original runs; distributing the archive is still required.

## Metrics And Interpretation

The table reports mean/median final value, sample standard deviation, loss
probability, fifth percentile, requested/executed action frequencies and
infeasible-action count/rate using PF-22 definitions. Paired mean changes are
variant minus 5% for the same policy and scenario, not DQN minus baseline.
95% percentile intervals use 2,000 scenario-bootstrap resamples with seed 23,
linear quantiles and common resample indices across comparisons. The 5% paired
change and its interval are exactly zero.

Mean final portfolio values:

| Strategy | 2.5% | 5% | 10% |
| --- | ---: | ---: | ---: |
| CASH_ONLY | 100.00 | 100.00 | 100.00 |
| BUY_AND_HOLD | 97.47 | 95.07 | 90.28 |
| ALWAYS_OPEN | 438.17 | 417.02 | 370.93 |
| EV_ONE_STEP | 177.27 | 172.13 | 100.00 |
| DQN 20261013 | 455.19 | 438.97 | 402.52 |
| DQN 20261014 | 466.53 | 447.79 | 413.55 |
| DQN 20261015 | 115.09 | 112.21 | 106.46 |
| DQN 20261016 | 286.84 | 275.03 | 244.49 |
| DQN 20261017 | 384.73 | 369.90 | 329.22 |

All five DQN replicas improve at the lower fee and decline at the higher fee;
their paired 95% intervals exclude zero, without multiple-comparison adjustment.
ALWAYS_OPEN changes by +21.15 [20.49, 21.80] and -46.09 [-50.06, -42.95]
against 5%. Its loss probability rises from 6.6% to 10.8% at 10%, and its fifth
percentile falls from 75.84 to 7.67: tail sensitivity is much larger than the
mean alone suggests. BUY_AND_HOLD's loss probability is 61%, 65% and 74.2%.

BUY_AND_HOLD's action frequencies are identical across fees. ALWAYS_OPEN's
executed opening rate is 49.44%, 49.37% and 48.86%, with zero infeasible
requests. DQN seed 20261016's infeasibility rises from 34.91% to 37.01% at
10%; seed 20261015 stays near 55.6% and executes very few openings. Retain the
requested/executed distinction when interpreting these policies.
EV at 10% chooses only HOLD because buying has negative one-step advantage
in every regime; its mean 100 and zero loss are verified behavior, not the
initial baseline bug. This myopic policy does not establish that multi-step
opening opportunities are unprofitable.

Frozen weights do not imply identical actions: changed sale proceeds affect
cash, feasibility and later observations. EV's fee-aware recomputation is an
information advantage, not evidence that DQN learned fee adaptation.

For PF-24, discuss each replica's paired changes, loss/tail changes and
requested-versus-executed behavior alongside baseline changes. Preserve all
five replicas; do not select a winner or redesign the protocol after observing
the results. The CSV and linked episode evidence support regenerable figures;
use the official corrected CSV and linked archive, never the failed CSV.
These are synthetic simulator findings, not real NFL-market forecasts. Review
the large tail deterioration and replica variability alongside uncertainty;
do not select models or change the protocol from these held-out outcomes.

## Limitations

The fixed-model scenario bootstrap excludes training and selection uncertainty
and is not adjusted for multiple comparisons. Historical PF-21 tuning records
are incomplete; the unchanged 2,000-step training budget does not establish
convergence. The PF-12/PF-13 float32 affordability ambiguity remains. Adapter
orchestration duplicates PackfolioEnv and requires parity tests against future
changes. Absolute input paths in provenance require the same accessible inputs
for aggregation; the output is not yet a portable PF-24 release package.
