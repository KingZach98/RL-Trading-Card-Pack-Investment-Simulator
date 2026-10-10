# PF-20 Simulator Audit and Assumptions

This document is the PF-20 "Work" deliverable: a review of the probability
model, model-informed baseline, price transitions, fees, and action
constraints, an explanation of the dominant simple strategy found in PF-19,
and a record of the defects/discrepancies found and how they were resolved.
It is written evidence to support the freeze decided in
[`docs/decisions.md`](decisions.md) Section 8, not a substitute for that
sign-off.

Status: **Proposed — pending team review**, same as the freeze it supports.
A poor or dominated final return by itself is not evidence of a bug; this
document exists so that any result can be checked against the assumptions
recorded here before anyone calls it one.

## 1. Scope and method

Reviewed, file by file:

- `src/packfolio/packs.py` — the probability model (pack outcome sampling).
- `src/packfolio/market.py` — price/regime transitions.
- `src/packfolio/portfolio.py` — fees and action constraints (buy/sell/open).
- `src/packfolio/env.py` — step timing, terminal liquidation, reward.
- `src/packfolio/baselines.py` — `CASH_ONLY`, `BUY_AND_HOLD`, `ALWAYS_OPEN`.
- `src/packfolio/ev_baseline.py` — the model-informed `EV_ONE_STEP` baseline.

Cross-checked against:

- `docs/decisions.md` (PF-01 decision register and PF09-D01 replacement).
- `docs/ev_baseline.md` (PF-13's EV baseline write-up and known limitations).
- `docs/development_comparison.md` (PF-19's four-baseline-plus-DQN results).
- `configs/environment.json`, `configs/nfl_pack.json` (the operative config).

`python -m packfolio.validate` (the existing Monte Carlo diagnostic CLI) was
run against the frozen base config and both fee-sensitivity variants to check
the probability model and price transitions at scale; results are in
Section 4.

## 2. Code review: no defects found

Reading `portfolio.py` and `env.py` line by line against the PF-01 decision
register found no code defects:

- **Step timing** matches PF01-D16: a pack bought at timestep `t` is sealed
  inventory at `t`; opening/selling/holding act on the state as of `t`; the
  market transitions once per step, after the action is applied.
- **Terminal liquidation** matches PF01-D17: at `t == horizon`, remaining
  sealed inventory is liquidated to cash at the current regime's ask (not
  opened and not left as unrealized inventory value), and no further actions
  or market transitions are accepted past that point. This matches
  `tests/test_evaluate_reload.py`'s and `tests/test_env.py`'s coverage of
  terminal transitions not bootstrapping beyond the horizon (see
  `docs/learning_check.md` for the PF-17 agent-side check of this same
  property).
- **Fees** are applied exactly once per sale (sealed sale or opened-content
  sale), consistent with PF01-D09, and are configuration-only: scenario draws
  stay gross, with the fee and cash changes handled entirely in
  `portfolio.py`/`env.py` (`configs/README.md` already documents this split;
  the code matches it).
- **Action constraints** (`_can_buy`, `_has_sealed_pack` in `baselines.py`;
  the inventory-capacity and affordability checks in `portfolio.py`)
  uniformly convert an infeasible action to `HOLD` rather than silently
  succeeding or raising, for every policy including the trained DQN — this
  was already confirmed for the DQN pilot in PF-19
  (`docs/development_comparison.md` records 38/21/21 infeasible BUY
  requests converted to HOLD across the three development seeds).
- The only known limitation is the float32 affordability-rounding interval
  documented in `docs/ev_baseline.md` (PF-13): very close to an exact
  affordability boundary, float32 truncation of the observation can disagree
  with the float64 ground truth by one rounding step. This was already
  identified, labeled, and accepted as a documented limitation rather than a
  defect in PF-13 — PF-20 does not change that conclusion, and does not find
  any new instance of it that was previously unlabeled.

No changes to `env.py`, `portfolio.py`, `packs.py`, `market.py`, or
`baselines.py` were made as part of PF-20.

## 3. Why `ALWAYS_OPEN` dominates (not a bug)

PF-19's development comparison (`docs/development_comparison.md`) found:

| Policy | Seed 2001 | Seed 2002 | Seed 2003 | Mean |
|---|---|---|---|---|
| CASH_ONLY | 100.000 | 100.000 | 100.000 | 100.000 |
| BUY_AND_HOLD | 109.000 | 109.000 | 109.000 | 109.000 |
| ALWAYS_OPEN | 448.875 | 299.000 | 590.563 | 446.146 |
| EV_ONE_STEP (model-informed) | 102.725 | 123.613 | 161.525 | 129.288 |
| DQN (development pilot) | 75.500 | 191.550 | 377.950 | 215.000 |

`ALWAYS_OPEN` beats every other policy, including the trained DQN pilot, by a
wide margin. The ticket's purpose is explicitly "explain any dominant simple
strategy rather than forcing every action to be attractive" — so this section
derives *why*, using only the frozen config's own values
(`configs/nfl_pack.json`, `configs/environment.json`):

Pack outcome probabilities and base gross values (`nfl_pack.json`):

| Outcome | Probability | Base gross value |
|---|---|---|
| base_bundle | 0.70 | 5.0 |
| rookie_bundle | 0.20 | 15.0 |
| autograph_bundle | 0.09 | 75.0 |
| premium_bundle | 0.01 | 250.0 |

Expected gross value at card multiplier `1.0` (the `NORMAL` regime):

```
EV = 0.70*5.0 + 0.20*15.0 + 0.09*75.0 + 0.01*250.0 = 15.75
```

This is exactly the `pack_mean_gross_value:multiplier_1` diagnostic checked
by `packfolio.validate` (Section 4): observed `15.71`, expected `15.75`,
within tolerance.

Net of the 5% selling fee (base frozen config), compared against each
regime's pack ask (`configs/environment.json`'s `market.quotes`):

| Regime | Card multiplier | Gross EV (`15.75 * multiplier`) | Net EV (`* 0.95`) | Pack ask | Edge (net EV − ask) |
|---|---|---|---|---|---|
| LOW | 0.75 | 11.8125 | 11.221875 | 8.0 | **+3.22** |
| NORMAL | 1.00 | 15.75 | 14.9625 | 10.0 | **+4.96** |
| HIGH | 1.60 | 25.20 | 23.94 | 12.0 | **+11.94** |

Buying and opening a pack is **unconditionally positive expected value in
every market regime** at the current pack values and ask prices. This is a
structural property of the configured values — the PF09-D01 pack odds/values
and the PF01-D22-derived market quotes — not an environment or accounting
defect. `ALWAYS_OPEN` simply executes this positive-EV cycle on every
step it can, which is why it dominates:

- `BUY_AND_HOLD` only captures the edge once (ten packs bought then held,
  per `docs/development_comparison.md`), so it earns a single-shot gain
  instead of repeatedly re-applying the edge.
- `EV_ONE_STEP` is a *myopic*, information-boundary-respecting baseline
  (`docs/ev_baseline.md`): it reacts to the realized price and regime each
  step rather than unconditionally repeating the same action, and it is also
  constrained to a single action per step like every other policy — it still
  beats `CASH_ONLY`/`BUY_AND_HOLD` but does not match `ALWAYS_OPEN`'s
  unconditional repetition of the dominant cycle.
- The DQN pilot (2,000 steps, a development-only training budget, not a
  tuned final agent) partially discovers the same edge but has not converged
  to repeat it as consistently as the hard-coded `ALWAYS_OPEN` baseline, and
  also showed many infeasible-BUY-converted-to-HOLD events reducing its
  realized return.

Remaining positive-EV even at the fee doubled to 10% (`env_frozen_fee_high.yaml`):

| Regime | Net EV (`* 0.90`) | Pack ask | Edge |
|---|---|---|---|
| LOW | 10.63125 | 8.0 | **+2.63** |
| NORMAL | 14.175 | 10.0 | **+4.175** |
| HIGH | 22.68 | 12.0 | **+10.68** |

So doubling the fee reduces the edge but does not eliminate the dominant
strategy; the fee-sensitivity variants measure how much the edge shrinks,
not whether it disappears. This is an explicit, labeled property of this
version of the task, carried forward into the freeze rather than resolved by
retuning pack values, fees, or action constraints to make every action
equally attractive — doing the latter would be exactly the "unreported way
to favor the RL agent" this ticket exists to prevent. Any future team
decision to change pack values, fees, or market quotes specifically to
remove this dominance must go through the normal PF-01 decision-register
process and create a new version, not be folded silently into this freeze.

## 4. Diagnostic verification (`packfolio.validate`)

`python -m packfolio.validate --config <path>` samples 100,000 pack draws and
50,000 market transitions per regime and checks observed frequencies/means
against expected values within statistical tolerance
(`tolerance = max(0.005, 5 * binomial standard error)` for probabilities,
`max(0.1, 5 * standard error)` for means). Run against all three frozen
configs on 2026-10-08:

| Config | `config_hash` | Result |
|---|---|---|
| `configs/env_frozen.yaml` | `57ce865515268c2e271ab34776c6095276a48e5cd71736399732092fe73d7330` | `"passed": true` — all 14 diagnostics (4 pack frequencies, 1 pack mean, 9 market transitions) within tolerance |
| `configs/env_frozen_fee_low.yaml` | `b4c57d02b506d60f85ab4adcfeb18a61672a8af5f508deabd96eeb6b94a3bbde` | `"passed": true` — same 14 diagnostics, all within tolerance (fee does not affect pack/market distributions) |
| `configs/env_frozen_fee_high.yaml` | `1ca1c65ec8b610c6edf4c231f8b84d12531735ef6baeebc45fa8e0399079c31d` | `"passed": true` — same 14 diagnostics, all within tolerance |

This confirms the probability model and price transitions behave as
configured at scale, for the base config and both predetermined fee
variants, before any final testing.

## 5. Assumption labels

Explicitly synthetic/placeholder assumptions carried into the freeze
(already labeled in `configs/nfl_pack.json`'s `description` field and
`configs/README.md`, repeated here for a single audit-facing list):

- All prices, probabilities, and market regimes are synthetic; they are not
  real product odds or market prices (`configs/nfl_pack.json`).
- The pack product is one fictional NFL-themed product; no real manufacturer
  or licensed set is modeled (PF01-D02).
- Market quotes and pack values share one currency unit; starting cash,
  quotes, and pack values were rescaled by ÷100 from the PF-01 draft
  proposals, recorded in `docs/decisions.md` PF20-D01/PF20-D02.
- The horizon is fixed at 100 steps (`docs/decisions.md` PF20-D03), not the
  52 originally drafted in PF01-D04.
- The float32 affordability-rounding interval near an exact boundary is a
  known, accepted limitation (`docs/ev_baseline.md`), not something this
  freeze attempts to eliminate.
- `EV_ONE_STEP`'s own documented limitations (`docs/ev_baseline.md`) —
  myopic, one-step horizon, information-boundary rules — are unchanged by
  this freeze.

## 6. Freeze summary

- **Base configuration**: `configs/env_frozen.yaml`, `config_hash`
  `57ce865515268c2e271ab34776c6095276a48e5cd71736399732092fe73d7330` —
  identical resolved settings to `configs/environment.json` (same hash), so
  freezing it changes nothing about the currently running pipeline.
- **Fee-sensitivity variants** (predetermined before final testing, not
  chosen afterward): `configs/env_frozen_fee_low.yaml` (2.5% fee) and
  `configs/env_frozen_fee_high.yaml` (10% fee); every other field matches
  the base.
- **Final-test protocol**: `configs/splits/final_test.json` is recorded
  as-is, owned by Saki, and was not used to tune any of the above.
- **Versioning**: any later change to the base config, either fee variant,
  `nfl_pack.json`, or the simulator/scenario semantics creates a new version
  and reruns every affected comparison (`docs/decisions.md` Section 7's
  standing note, reaffirmed in Section 8).
- **Approval**: this document and the Section 8 decision record are
  proposed, not approved. PF-21 should not tune against this environment
  until the Section 8.4 sign-off is complete.
