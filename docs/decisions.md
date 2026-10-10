# Packfolio RL MVP Decision Record

| Field | Value |
|---|---|
| PF task | PF-01 |
| Document status | **Draft — team review required** |
| Owner | Zachery |
| Reviewer | Tim |
| Target review | 2026-10-01 |
| Related specification | [`docs/spec.md`](spec.md) |
| GitHub issue | [#1 — Approve the MVP and record the design decisions](https://github.com/KingZach98/RL-Trading-Card-Pack-Investment-Simulator/issues/1) |

> Nothing in this register is approved merely because it appears in this file.
> The proposed choices come from the project bible and are prepared for team
> review. Update each row during the meeting, record replacements explicitly,
> and complete all four sign-offs before closing PF-01.

## 1. Status vocabulary

Use exactly one of these statuses for each decision:

| Status | Meaning |
|---|---|
| `Proposed — pending team review` | Suggested choice; not yet approved. |
| `Approved` | Accepted by all four team members. |
| `Replaced` | Rejected and superseded by a documented replacement value/rule. |
| `Unresolved` | A required answer or external confirmation is still missing. |
| `Deferred` | Deliberately moved outside Version 1 without blocking the MVP. |

For a `Replaced` decision, write the replacement in the **Review outcome**
column and update `docs/spec.md` in the same pull request.

## 2. Decision register

| ID | Decision | Proposed choice | Status | Review outcome / notes |
|---|---|---|---|---|
| PF01-D01 | Research question | Can a reinforcement-learning agent achieve a higher mean final net portfolio value than simple baselines when buying, opening, holding, and selling NFL-themed packs in a simulated market? | Proposed — pending team review | |
| PF01-D02 | Product scope | One fictional NFL-themed pack product; no real manufacturer or set. | Proposed — pending team review | |
| PF01-D03 | Data status | All Version 1 prices, probabilities, and regimes are synthetic and must be labeled as such. | Proposed — pending team review | |
| PF01-D04 | Episode horizon | `52` weekly decisions. | Proposed — pending team review | |
| PF01-D05 | Starting cash | `10,000` virtual kr. | Proposed — pending team review | |
| PF01-D06 | Starting inventory | `0` sealed packs. | Proposed — pending team review | |
| PF01-D07 | Inventory capacity | Maximum `10` sealed packs. | Proposed — pending team review | |
| PF01-D08 | Starting regime | `NORMAL`. | Proposed — pending team review | |
| PF01-D09 | Selling fee | `5%` on sealed sales and opened-content sales; charged exactly once. | Proposed — pending team review | |
| PF01-D10 | Other frictions | No buy fee, interest, storage cost, sale delay, supply limit, or price impact. | Proposed — pending team review | |
| PF01-D11 | Action IDs/order | `0 HOLD`, `1 BUY_PACK`, `2 OPEN_AND_SELL`, `3 SELL_PACK`. | Proposed — pending team review | |
| PF01-D12 | Opening simplification | Opening samples a complete pack and immediately sells all contents; individual cards are not retained. | Proposed — pending team review | |
| PF01-D13 | Action limit | Exactly one requested action per week; buying and opening require separate weeks. | Proposed — pending team review | |
| PF01-D14 | Infeasible action | Convert an in-range but infeasible action to `HOLD`, advance time, apply no extra penalty, and log requested/executed actions. | Proposed — pending team review | |
| PF01-D15 | Invalid action ID | An action outside IDs `0..3` is a programming error and fails clearly. | Proposed — pending team review | |
| PF01-D16 | Step timing | Value current portfolio, act at current prices, advance/sample market once, revalue, liquidate if terminal, then calculate reward. | Proposed — pending team review | |
| PF01-D17 | Terminal liquidation | After the final market transition, automatically sell every remaining sealed pack at the terminal net quote with no duplicated fee. | Proposed — pending team review | |
| PF01-D18 | Early termination | Do not terminate because cash is low or inventory is empty; end only at the fixed task horizon. | Proposed — pending team review | |
| PF01-D19 | Reward | `(V_next - V_current) / V_initial`, with no separate terminal bonus. | Proposed — pending team review | |
| PF01-D20 | Discount factor | `gamma = 1.0` so cumulative reward equals proportional final portfolio gain. | Proposed — pending team review | |
| PF01-D21 | Pack outcomes | Probabilities `0.70/0.25/0.05` with base gross values `200/1,000/8,000` virtual kr. | Replaced | Project owner chose to keep the current four-outcome config on 2026-10-05. See PF09-D01 for the exact replacement. This does not complete team sign-off. |
| PF01-D22 | Market quotes | `LOW: 800, 0.75`; `NORMAL: 1,000, 1.00`; `HIGH: 1,200, 1.60`, where each pair is sealed ask and card multiplier. | Proposed — pending team review | |
| PF01-D23 | Market transitions | `LOW -> 0.70/0.25/0.05`; `NORMAL -> 0.15/0.70/0.15`; `HIGH -> 0.05/0.25/0.70`, ordered Low/Normal/High. | Proposed — pending team review | |
| PF01-D24 | Randomness | Separate deterministic market and pack streams derived from one scenario seed; unused timestep pack draws do not shift later draws. | Proposed — pending team review | |
| PF01-D25 | Public information | Policies cannot inspect seeds, scenario IDs, future regimes, future pack draws, or privileged diagnostics. | Proposed — pending team review | |
| PF01-D26 | DQN implementation | Proposed: Stable-Baselines3 DQN, subject to teacher permission. | **Unresolved — teacher confirmation required** | Question sent/asked on: ___; answer: ___ |
| PF01-D27 | Simulator version | Use simulator version `0.1` before the later experimental freeze; record config hashes with results. | Proposed — pending team review | |
| PF01-D28 | Source/artifact sharing | Source, synthetic configs, tests, and curated results are intended for version control; rules for model artifacts and any course restrictions require confirmation. | **Unresolved** | Owner for confirmation: ___ |
| PF01-D29 | Configuration format | Store assumptions in validated external configuration; exact YAML/TOML/JSON choice is finalized in PF-02/PF-03. | Proposed — pending team review | |
| PF01-D30 | Scope exclusions | No live APIs, scraping, forecasting, grading, multiple products, real-money trading, individual-card inventory, or full website in Version 1. | Proposed — pending team review | |
| PF01-D31 | Observation defaults | Use the eight-feature order in `docs/spec.md` and a price-normalization reference of `1,000` virtual kr; PF-03 freezes the exact schema. | Proposed — pending team review | |

## 3. Questions to take to the teacher

- [ ] May the team use Stable-Baselines3's existing DQN implementation?
- [ ] Are there restrictions on third-party Python libraries beyond normal
  attribution and dependency documentation?
- [ ] Must trained checkpoints be submitted, or is a documented retraining path
  sufficient?
- [ ] Are there course-specific rules for making the repository or results
  public?

Record the teacher's response here:

| Date | Asked by | Question(s) | Response | Decision IDs affected |
|---|---|---|---|---|
| | | | | |

## 4. Team-review agenda

Suggested meeting procedure:

1. Read the research question and Version 1 inclusions/exclusions.
2. Review PF01-D01 through PF01-D31 in order.
3. Change each proposed row to `Approved`, `Replaced`, `Unresolved`, or
   `Deferred`.
4. For every replacement, write the exact replacement—not only "change this."
5. Confirm that `docs/spec.md` matches the resulting decisions.
6. Assign an owner and due date to every unresolved external question.
7. Have all four members complete the sign-off table.
8. Have Tim perform the reviewer check and update the acceptance checklist.

## 5. Team sign-off

By signing off, each member confirms that they reviewed the research question,
MVP scope, exclusions, numerical defaults, action/step rules, terminal handling,
synthetic-data boundary, and unresolved items.

| Member | Project role | Reviewed? | Date | Approval or notes |
|---|---|---|---|---|
| Zachery | Environment/accounting; PF-01 owner | [ ] | | |
| Tim | Simulation; PF-01 reviewer | [ ] | | |
| Nasir | Agent/training | [ ] | | |
| Saki | Baselines/evaluation | [ ] | | |

## 6. PF-01 acceptance checklist

- [ ] All four members reviewed the scope and exclusions.
- [ ] Every numerical setting is `Approved` or `Replaced`.
- [ ] Every replaced value is reflected in `docs/spec.md`.
- [ ] The four-action MVP and action order are recorded.
- [ ] Step timing and terminal liquidation are recorded.
- [ ] Synthetic-data status is recorded.
- [ ] The DQN-library decision is recorded, including the teacher response.
- [ ] Simulator version `0.1` or its replacement is recorded.
- [ ] Remaining unresolved items have an owner and due date.
- [ ] Tim verified that `docs/spec.md` and this register agree.
- [ ] The approved documents are committed and linked from GitHub Issue #1.

PF-01 should remain open until all required boxes above are checked.

## 7. Change record after approval

Once PF-01 is approved, append changes here instead of silently editing the
rules. Shared interface changes must update the specification, configuration,
and affected tests together.

| Change ID | Date | Decision/spec section | Old value/rule | New value/rule | Reason | Approved by | Version impact |
|---|---|---|---|---|---|---|---|
| PF09-D01 | 2026-10-05 | PF01-D21; spec Section 8; interface Section 5.2 | Shared IDs `LOW_VALUE`, `MEDIUM_VALUE`, `HIGH_VALUE`; proposed odds `0.70/0.25/0.05` and values `200/1,000/8,000` | Keep configured IDs `base_bundle`, `rookie_bundle`, `autograph_bundle`, `premium_bundle`; odds `0.70/0.20/0.09/0.01`; base values `5/15/75/250` simulation units | Match shared pack results to the existing sampler for PF-09 without changing its draws | Project owner, via chat | Interface `2.0`; step info `2.0`; observation and evaluation row schemas stay `1.0` |

PF09-D01 records the project owner's choice for Issue #9. It does not claim
that all four team members have signed off on PF-01. The config files and
seeded scenario paths do not change in this prep batch.

After the PF-20 experimental freeze, any simulator change must create a new
version/configuration hash and rerun every affected comparison.

## 8. PF-20 — Environment audit and freeze

| Field | Value |
|---|---|
| PF task | PF-20 |
| Document status | **Proposed — pending team review** (not yet approved; see Section 8.4) |
| Depends on | PF-01 (environment rules), PF-09 (pack odds/values), PF-13 (EV baseline), PF-19 (strategy comparison) |
| Related documents | [`docs/assumptions.md`](assumptions.md) (full audit write-up), [`docs/development_comparison.md`](development_comparison.md) (dominant-strategy evidence), [`configs/env_frozen.yaml`](../configs/env_frozen.yaml), [`configs/env_frozen_fee_low.yaml`](../configs/env_frozen_fee_low.yaml), [`configs/env_frozen_fee_high.yaml`](../configs/env_frozen_fee_high.yaml) |

This section records the PF-20 audit findings and the proposed freeze. Like
Section 1, nothing here is approved merely because it appears in this file;
see 8.4 for the required sign-off before PF-21 treats this as final.

### 8.1 Audit outcome

- **Code review** of `market.py`, `packs.py`, `portfolio.py`, `env.py`,
  `baselines.py`, and `ev_baseline.py` found no code defects: fee
  application, terminal liquidation, step timing, and action constraints
  match the PF-01 decision register (notably PF01-D16/D17) and PF-13's
  documented information-boundary rules. See `docs/assumptions.md` Section 2
  for the file-by-file review.
- **No forcing of every action to be attractive.** `ALWAYS_OPEN` is a
  dominant simple strategy (mean terminal value 446.1 vs. 100.0/109.0/129.3
  for `CASH_ONLY`/`BUY_AND_HOLD`/`EV_ONE_STEP`, and 215.0 for the PF-19 DQN
  pilot — `docs/development_comparison.md`). `docs/assumptions.md` Section 3
  shows this is an unconditionally positive expected value of buying and
  opening a pack in every market regime at the current pack values and ask
  prices, not an environment bug and not something retuned to help the
  agent; it is an explicit, labeled property of the PF09-D01 pack values.
- **Undocumented currency rescale resolved.** `configs/environment.json`
  and `configs/nfl_pack.json` already use starting cash and market quotes
  divided by 100 from the PF-01 draft proposals (PF01-D05, PF01-D22),
  consistent with the PF09-D01 pack-value rescale, but this was never
  logged as its own change. PF20-D01/PF20-D02 below record the operative
  values explicitly.
- **Horizon ambiguity resolved.** PF01-D04 still proposes `52`, but every
  validated PF-13/17/18/19 artifact and the shipped `environment.json` use
  `100`. PF20-D03 formalizes `100` as the frozen value rather than leaving
  the register and the configs disagreeing.

### 8.2 Proposed changes (pending approval)

| Change ID | Date | Decision/spec section | Old value/rule | New value/rule | Reason | Proposed by | Version impact |
|---|---|---|---|---|---|---|---|
| PF20-D01 | 2026-10-08 | PF01-D05; `configs/environment.json` | Draft proposal: starting cash `10,000` virtual kr | Operative value: `100.0` simulation units | Starting cash was already rescaled by ÷100 alongside the PF09-D01 pack-value rescale, but never logged as its own change; this records the value already in use rather than changing it | PF-20 audit (pending team sign-off) | None — `environment.json` already used `100.0`; no `config_hash` change |
| PF20-D02 | 2026-10-08 | PF01-D22; `configs/environment.json` | Draft proposal: quotes `LOW: 800, 0.75`; `NORMAL: 1,000, 1.00`; `HIGH: 1,200, 1.60` | Operative quotes: `LOW: 8, 0.75`; `NORMAL: 10, 1.00`; `HIGH: 12, 1.60` | Same ÷100 rescale as PF20-D01, recorded for the market quotes | PF-20 audit (pending team sign-off) | None — `environment.json` already used these values; no `config_hash` change |
| PF20-D03 | 2026-10-08 | PF01-D04; `docs/ev_baseline.md` | Draft proposal: `52` weekly decisions | Operative value: `100` steps | Horizon `100` has been in de facto use since before PF-13 across every validated artifact; this resolves the stale register-vs-config disagreement instead of reopening it later | PF-20 audit (pending team sign-off) | None — resolves a documentation ambiguity only; `environment.json` already used `100` |
| PF20-D04 | 2026-10-08 | New — environment freeze | No frozen artifact existed before PF-20 | Base config `configs/env_frozen.yaml` (`config_hash` `57ce865515268c2e271ab34776c6095276a48e5cd71736399732092fe73d7330`, identical resolved settings to `configs/environment.json`); fee-sensitivity variants `configs/env_frozen_fee_low.yaml` (2.5% fee) and `configs/env_frozen_fee_high.yaml` (10% fee); final-test manifest `configs/splits/final_test.json` recorded as-is and not used to tune this freeze | Gives PF-21 one fixed, hashed task instead of an evolving simulator, with predetermined fee variants decided before final testing | PF-20 audit (pending team sign-off) | New config artifacts/versions; existing `environment.json` hash is unchanged |

PF20-D01 through PF20-D04 are **proposed**, not approved — unlike PF09-D01,
no team member or project owner has signed off on these yet. They become
effective, and should be copied into Section 7 as approved changes, only
after Section 8.4 is completed.

### 8.3 Frozen artifacts

| Artifact | Purpose | `config_hash` |
|---|---|---|
| `configs/env_frozen.yaml` | Frozen base environment (same resolved settings as `configs/environment.json`) | `57ce865515268c2e271ab34776c6095276a48e5cd71736399732092fe73d7330` |
| `configs/env_frozen_fee_low.yaml` | Fee-sensitivity variant: `selling_fee_rate` halved to `0.025` | `b4c57d02b506d60f85ab4adcfeb18a61672a8af5f508deabd96eeb6b94a3bbde` |
| `configs/env_frozen_fee_high.yaml` | Fee-sensitivity variant: `selling_fee_rate` doubled to `0.10` | `1ca1c65ec8b610c6edf4c231f8b84d12531735ef6baeebc45fa8e0399079c31d` |
| `configs/splits/final_test.json` | Final-test scenario manifest, recorded unchanged; owned by Saki and not used to tune the freeze | (see manifest `config_hash` field) |

`python -m packfolio.validate --config <path>` passes all probability and
market-transition diagnostics for all three frozen configs (fee does not
change pack/market distributions, only the selling fee). See
`docs/assumptions.md` Section 4 for the full diagnostic output.

### 8.4 Sign-off

By signing off, each member confirms they reviewed the audit outcome (8.1),
the proposed changes (8.2), and the frozen artifacts (8.3), and approves one
hashed base configuration as required by the PF-20 acceptance criteria.

| Member | Project role | Reviewed? | Date | Approval or notes |
|---|---|---|---|---|
| Zachery | Environment/accounting; PF-01 owner | [ ] | | |
| Tim | Simulation; PF-01 reviewer | [ ] | | |
| Nasir | Agent/training | [ ] | | |
| Saki | Baselines/evaluation; final-test manifest owner | [ ] | | |

PF-20 should remain open, and PF-21 should not start tuning against this
environment, until all four sign-offs above are recorded and PF20-D01
through PF20-D04 are copied into Section 7 as approved changes.
