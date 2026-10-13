# Packfolio Findings And Reproduction

## Scope And Evidence

These results describe the synthetic Packfolio simulator, not real NFL card
prices, profitability or market forecasting. Initial cash is 100, horizon 100
and capacity 10. All nine strategies share 500 scenarios, seeds 3001-3500.
Five frozen DQN checkpoints are retained; no held-out model selection occurred.
PF-22 contains 4,500 episodes; corrected PF-23 adds 9,000 episodes at the
predeclared 2.5% and 10% fees, reusing the original 5% reference.
Sources are [PF-22 summaries](../results/summary.csv),
[uncertainty rules](../results/uncertainty.json) and
[corrected PF-23 summaries](../results/fee_sensitivity.csv).
The release manifest records artifact hashes and original run provenance.

## Performance And Downside

See [final values](../results/figures/final_values.png) and
[downside](../results/figures/downside.png). At the base 5% selling fee:

| Strategy | Mean final value | Loss probability | Fifth percentile | Infeasible requests |
| --- | ---: | ---: | ---: | ---: |
| ALWAYS_OPEN | 417.02 | 6.6% | 75.84 | 0.00% |
| BUY_AND_HOLD | 95.07 | 65.0% | 70.40 | 0.00% |
| CASH_ONLY | 100.00 | 0.0% | 100.00 | 0.00% |
| DQN 20261013 | 438.97 | 3.4% | 120.76 | 31.07% |
| DQN 20261014 | 447.79 | 2.0% | 119.97 | 1.61% |
| DQN 20261015 | 112.21 | 39.2% | 80.00 | 55.61% |
| DQN 20261016 | 275.03 | 23.6% | 57.98 | 34.91% |
| DQN 20261017 | 369.90 | 6.8% | 84.36 | 26.58% |
| EV_ONE_STEP | 172.13 | 24.2% | 60.99 | 0.00% |

All DQN replicas exceed CASH_ONLY and BUY_AND_HOLD in mean final value.
Four exceed EV_ONE_STEP; seed 20261015 does not. Against ALWAYS_OPEN,
seeds 20261013 and 20261014 have paired mean differences +21.94
[0.26, 43.44] and +30.77 [11.11, 51.46]; the other three have negative
paired intervals. Seed 20261013's lower bound is close to zero, and none of
these intervals is multiple-comparison adjusted. This is not evidence that
DQN generally outperforms the strongest simple baseline.

The descriptive average of the five replica means is 328.78, with sample
standard deviation 139.45. It is not an ensemble policy's value or an
independent-market confidence interval: replicas share scenarios, and five
training seeds give limited evidence about training variability.

Mean performance alone hides risk. Seeds 20261013/20261014 have lower loss
probabilities and higher fifth percentiles than ALWAYS_OPEN. Seed 20261015
has a positive mean despite frequent losses; seed 20261016 has a lower fifth
percentile than EV_ONE_STEP. Loss means terminal value below initial cash
minus tolerance 1e-9; fifth percentiles use linear interpolation.

## Behaviour And Training

[Action behaviour](../results/figures/action_behaviour.png) distinguishes
requested from executed actions: infeasible requests execute HOLD.
Seed 20261015 requests BUY on 65.468% of steps but executes BUY on 10.494%;
its executed opening rate is only 0.486%. ALWAYS_OPEN executes openings on
49.372% of steps. These differences are consistent with the outcome gap,
but do not establish a causal diagnosis of training failure.
Low infeasibility alone also does not determine the best policy.

[Training history](../results/figures/training_history.png) plots actual
exploratory episode returns and episode-mean losses for every replica.
Each log has 20 completed episodes / 2,000 interaction steps. Returns depend
on exploration and changing scenario draws; these short, noisy histories
are not convergence evidence or repeated greedy validation measurements.
Historical tuning records and intermediate validation histories are missing;
no missing curve was reconstructed or fabricated.

## Fee Sensitivity

See [fee sensitivity](../results/figures/fee_sensitivity.png).
DQN weights stay frozen, with no exploration, updates or retraining.
EV_ONE_STEP is model-informed and recomputes decisions using each fee.

| Strategy | Mean at 2.5% | Mean at 5% | Mean at 10% |
| --- | ---: | ---: | ---: |
| ALWAYS_OPEN | 438.17 | 417.02 | 370.93 |
| BUY_AND_HOLD | 97.47 | 95.07 | 90.28 |
| CASH_ONLY | 100.00 | 100.00 | 100.00 |
| DQN 20261013 | 455.19 | 438.97 | 402.52 |
| DQN 20261014 | 466.53 | 447.79 | 413.55 |
| DQN 20261015 | 115.09 | 112.21 | 106.46 |
| DQN 20261016 | 286.84 | 275.03 | 244.49 |
| DQN 20261017 | 384.73 | 369.90 | 329.22 |
| EV_ONE_STEP | 177.27 | 172.13 | 100.00 |

All DQN replicas improve at the lower fee and decline at the higher fee;
their within-replica paired intervals exclude zero. ALWAYS_OPEN changes
by +21.15 [20.49, 21.80] and -46.09 [-50.06, -42.95] versus 5%.
Its loss probability rises from 6.6% to 10.8% at 10%, while its fifth
percentile falls from 75.84 to 7.67. Tail robustness is weaker than its
high mean suggests. BUY_AND_HOLD's actions remain identical across fees.

EV at 10% requests only HOLD: buying has negative one-step advantage in
every regime. Its cash-only outcome is verified fee-aware behaviour, not
the first-run baseline bug. A myopic decision rule does not rule out
profitable multi-step buying/opening sequences.
Frozen weights can still produce different actions because fees change
cash and feasibility. This experiment tests robustness, not learned fee
adaptation. The PF-23 adapter retains genuine base scenario randomness
while using variant fees for accounting; ordinary variant configurations
would change the hash-derived random streams.

## Representative Episodes

[Episode examples](../results/figures/episode_examples.png) use the episode
nearest each strategy's median final value, breaking ties by lowest scenario
seed. Selected seeds are ALWAYS_OPEN 3468, BUY_AND_HOLD 3004, CASH_ONLY 3001,
DQN 20261013: 3025, 20261014: 3133, 20261015: 3014, 20261016: 3270,
20261017: 3048, and EV_ONE_STEP 3147.
These are per-strategy examples, not paired same-market comparisons.
They illustrate paths around typical terminal outcomes, not every source
of downside or an intentionally selected success.

## Uncertainty And Limitations

95% percentile scenario-bootstrap intervals use 2,000 resamples, seed 22
for PF-22 and seed 23 for PF-23, with linear quantiles. Paired differences
resample matched scenario indices. Intervals condition on fixed policies:
they omit training/selection uncertainty and multiple-comparison adjustment.
PF-23 recomputes base mean intervals with seed 23, so they need not equal
PF-22's seed-22 intervals despite using identical base episodes.
The 2,000-step budget, missing tuning/validation evidence, synthetic incentives
and float32 affordability ambiguity limit interpretation. Rounded observations
can imply BUY eligibility while strict portfolio accounting rejects it.
All runs retain the original requested/executed distinction and every replica.

## Clean-Checkout Guide

Independent reproduction is **pending**, not established by local plotting
tests. Obtain the PF-23 evidence ZIP from the project owner through a separate
file transfer; it is not tracked in Git and no public download is recorded.
Place it at the archive path in `results/release_manifest.json`, verify its
SHA-256 using `Get-FileHash -Algorithm SHA256`, and check the listed artifact
hashes before use. Exact byte hashes protect evidence and checkpoints.
Configuration/training text may differ only through CRLF/LF conversion;
do not silently accept a failed historical byte-hash check. Canonical config
hashes and explicitly labelled LF-normalized training-log hashes are separate
checks, not replacements for recorded raw-byte provenance.

Verify the listed release inputs from the repository root (a mismatch stops):
```powershell
$release = Get-Content -LiteralPath results/release_manifest.json -Raw | ConvertFrom-Json
$files = @($release.artifacts) + @($release.checkpoints | ForEach-Object { $_.checkpoint; $_.training_log; $_.training_manifest }) + @($release.external_evidence.required_archive)
$files | ForEach-Object { if ((Get-FileHash -LiteralPath $_.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $_.sha256) { throw "Hash mismatch: $($_.path)" } }
```

In a clean checkout with these extraction targets absent:
```powershell
uv sync --locked --extra dev --extra agent
Expand-Archive -LiteralPath outputs/pf23-publication/pf23-evidence.zip -DestinationPath outputs/pf24-evidence
Copy-Item -LiteralPath outputs/pf24-evidence/pf23-corrected -Destination outputs/pf23-corrected -Recurse
Copy-Item -LiteralPath outputs/pf24-evidence/fee_sensitivity_corrected.csv -Destination results/fee_sensitivity_corrected.csv
Expand-Archive -LiteralPath outputs/pf24-evidence/pf22-results.zip -DestinationPath outputs/pf24-reference
Copy-Item -LiteralPath outputs/pf24-reference/raw -Destination results/raw -Recurse
```

The archive contains corrected PF-23 evidence and the PF-22 reference archive,
not failed-run results. Existing official tables remain unchanged.
PF-22/PF-23 run manifests contain absolute input paths; PF-23 additionally
records an absolute reference directory and summary destination.
The plotting loader currently checks those original paths. A different
checkout needs explicit filesystem mapping to the recorded repository root
and matching source bytes, or a separately approved PF-24 path resolver.
Do not rewrite signed historical manifests or bypass hash failures.
Training source-path metadata is historical; model reload uses relative
checkpoint/configuration filenames. Portable plotting is not yet verified.

Once those historical paths are genuinely accessible:
```powershell
.\.venv\Scripts\python.exe -B -m packfolio.plots --output results/figures-reproduced
.\.venv\Scripts\python.exe -B -m pytest tests/test_plots.py -q -p no:cacheprovider
```
Use a new figure directory; the generator refuses existing outputs.
Fixed-runtime fixture tests verify byte-identical rendering; cross-platform
font/render differences require checking the numerical inputs as well.

The existing checkpoint runner supports an independent evaluation without
training. After separate reproduction approval, use the preselected lowest
training seed (not the best-scoring model), writing only to a new directory:
```powershell
.\.venv\Scripts\python.exe -B -m packfolio.evaluate --checkpoint runs/pf21-final-replicas/20261010T015008.025131Z-1854d265/checkpoint.pt --split-manifest configs/splits/final_test_pf22.json --output outputs/pf24-independent-evaluation
```
This replays 500 existing scenarios for one frozen replica. Compare saved
steps and stable episode fields with PF-22 seed 20261013, not Git commit,
timestamp or source-path metadata, which legitimately differ.
This command has not been executed in this documentation phase.

## Independent Handoff Checklist

- Record member, date, published commit, dependency versions and clean tracked status.
- Obtain the archive, verify its hash, and restore evidence without overwriting other runs.
- Record any required path mapping and newline handling; stop on unexplained mismatches.
- Regenerate figures and compare numerical inputs, labels and selected scenario seeds.
- Run the approved frozen-checkpoint evaluation; compare all matched traces and stable fields.
- Record failures as well as successes and attach commands/output hashes.
- Only then mark independent evaluation reproduction complete; record other cross-member module checks separately.
