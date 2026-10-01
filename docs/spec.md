# Packfolio RL MVP Simulator Specification

| Field | Value |
|---|---|
| PF task | PF-01 |
| Document status | **Draft — team approval required** |
| Specification version | `0.1-draft` |
| Proposed simulator version | `0.1` |
| Owner | Zachery |
| Reviewer | Tim |
| Target team review | 2026-10-01 |
| Decision record | [`docs/decisions.md`](decisions.md) |
| GitHub issue | [#1 — Approve the MVP and record the design decisions](https://github.com/KingZach98/RL-Trading-Card-Pack-Investment-Simulator/issues/1) |

> This is a review draft, not an approved specification. Unless stated otherwise,
> every choice below is **Proposed — pending team review**. It becomes the agreed
> MVP specification only after Zachery, Tim, Nasir, and Saki complete the sign-off
> in `docs/decisions.md`.

## 1. Purpose

Packfolio RL is a research simulator for studying sequential decisions involving
virtual cash and sealed NFL-themed card packs. An agent chooses when to buy a
sealed pack, open and sell its contents, sell the sealed pack, or hold. The goal
is to compare decision policies under the same documented synthetic market.

The project does **not** predict real card prices, recommend purchases, or place
real-money trades.

## 2. Research question

**Proposed research question:**

> Can a reinforcement-learning agent achieve a higher mean final net portfolio
> value than simple baseline strategies when buying, opening, holding, and
> selling NFL-themed card packs in a simulated market?

The claim produced by Version 1 must be limited to performance inside the frozen
simulator. Results must not be presented as evidence about what anyone should
buy in the real NFL-card market.

## 3. Version 1 scope

### Included

- One fictional NFL-themed sealed-pack product.
- Virtual cash and sealed-pack inventory.
- A fixed 52-week episode.
- Four discrete actions: hold, buy, open-and-sell, and sell sealed.
- Three visible synthetic market regimes.
- Three synthetic whole-pack outcome categories.
- Transaction fees on sealed and opened-content sales.
- Automatic terminal liquidation of remaining sealed inventory.
- Baseline policies and one DQN policy evaluated through the same environment.
- Reproducible, independently seeded market and pack randomness.

### Excluded

- Real products, manufacturers, sets, pull rates, or observed market prices.
- Live APIs, web scraping, or marketplace integration.
- Price forecasting, player-performance prediction, and grading.
- Borrowing, short selling, cash interest, and real-money orders.
- Multiple pack products.
- Holding or selling individual cards after opening.
- Auctions, sale delays, storage costs, price impact, or supply depletion.
- A production website or trading platform.

`OPEN_AND_SELL` is intentionally one bundled Version 1 operation. Retaining
individual cards would require a new state/action schema and is an extension.

## 4. Environment formulation

### 4.1 Episode

| Setting | Proposed value | Status |
|---|---:|---|
| Decision interval | One virtual week | Proposed — pending team review |
| Horizon, `T` | 52 decisions | Proposed — pending team review |
| Starting cash, `C0` | 10,000 virtual kr | Proposed — pending team review |
| Starting sealed inventory, `N0` | 0 packs | Proposed — pending team review |
| Inventory capacity, `Nmax` | 10 packs | Proposed — pending team review |
| Starting market regime | `NORMAL` | Proposed — pending team review |

Every strategy begins from the same initial state. An episode does not finish
early because cash is low or inventory is empty.

### 4.2 Conceptual Markov state

At decision time `t`, the conceptual state is:

```text
s_t = (cash_t, sealed_count_t, market_regime_t, T - t)
```

The current regime is visible. In Version 1, the current sealed-pack ask and the
card-value multiplier are deterministic functions of that regime. Remaining
time is part of the state because the task has a fixed horizon.

### 4.3 Observation schema

The proposed implementation observation contains eight `float32` features in
this fixed order:

| Setting | Proposed value | Status |
|---|---:|---|
| Observation length | 8 features | Proposed — pending team review |
| Price-normalization reference | 1,000 virtual kr | Proposed — pending team review |

| Position | Feature |
|---:|---|
| 0 | `cash / initial_cash` |
| 1 | `sealed_count / inventory_capacity` |
| 2 | `current_pack_ask / reference_price` |
| 3 | Current card-value multiplier |
| 4 | `remaining_steps / horizon` |
| 5 | `1` when regime is `LOW`, otherwise `0` |
| 6 | `1` when regime is `NORMAL`, otherwise `0` |
| 7 | `1` when regime is `HIGH`, otherwise `0` |

Internal accounting uses `float64`. Profitable cash values must not be clipped
to `1`. Exact observation bounds and schema compatibility behavior are frozen
in PF-03.

## 5. Actions and feasibility

The integer action order is part of the public interface:

| ID | Action | Feasible when | Immediate effect at current quotes |
|---:|---|---|---|
| 0 | `HOLD` | Always | Keep cash and inventory. |
| 1 | `BUY_PACK` | Cash covers the current ask and inventory is below capacity. | Pay one current ask price and add one sealed pack. |
| 2 | `OPEN_AND_SELL` | At least one sealed pack is owned. | Remove one pack, sample its whole-pack outcome, and add its net content-sale proceeds. |
| 3 | `SELL_PACK` | At least one sealed pack is owned. | Remove one pack and add its current net sealed-sale proceeds. |

Only one requested action is executed per week. Buying and opening therefore
require separate weekly decisions.

An action ID from `0` through `3` that is currently infeasible is converted to
`HOLD`. Time still advances, no extra penalty is invented, and both the requested
and executed actions are logged. An action outside `0` through `3` is a
programming error and must fail clearly.

## 6. Transaction and accounting rules

### 6.1 Prices and fees

Let:

- `B(m)` be the sealed-pack ask in market regime `m`.
- `f` be the selling-fee rate.
- `X` be the sampled gross value of an opened pack's contents.

Proposed transaction rules:

```text
buy cost              = B(m)
sealed sale proceeds  = (1 - f) * B(m)
opened sale proceeds  = (1 - f) * X
```

| Setting | Proposed value | Status |
|---|---:|---|
| Selling fee, `f` | 0.05 | Proposed — pending team review |
| Buy fee | 0 | Proposed — pending team review |
| Cash interest | 0 | Proposed — pending team review |
| Storage cost | 0 | Proposed — pending team review |

The selling fee is applied exactly once. Opening proceeds are not counted as
pure profit: opening also removes a sealed asset from the portfolio.

### 6.2 Liquidation value

At time `t`, portfolio liquidation value is:

```text
V_t = cash_t + sealed_count_t * current_net_sealed_quote_t
```

The current net sealed quote already includes the selling fee. Terminal
liquidation must not apply that fee a second time.

### 6.3 Reward

The proposed transition reward is:

```text
r_(t+1) = (V_(t+1) - V_t) / V_0
```

where `V_0` is the initial portfolio value. The proposed discount factor is
`gamma = 1.0`. Consequently, cumulative episode reward telescopes to:

```text
(V_T - V_0) / V_0
```

No separate terminal portfolio bonus is added.

## 7. Exact step order

For each `step(action)` at time `t`:

1. Read the current state and calculate `V_t` using current net quotes.
2. Validate the requested action and execute it at current prices. If the action
   is `OPEN_AND_SELL`, use the private pack draw assigned to timestep `t`.
3. Advance time exactly once and sample the next market regime.
4. Revalue remaining sealed inventory at the next regime's net quote.
5. If the new time is `T`, liquidate all remaining sealed packs at that net quote.
6. Calculate `V_(t+1)`, reward, termination flags, and diagnostics.
7. Return the next observation, reward, flags, and `info`.

The built-in horizon is task termination (`terminated=True`), not an external
interruption. An external interruption, if one is later introduced, is
truncation and must be handled separately.

## 8. Synthetic pack model

One opening samples a complete pack outcome from this proposed distribution:

| Outcome | Probability | Base gross bundle value | Status |
|---|---:|---:|---|
| `LOW_VALUE` | 0.70 | 200 virtual kr | Proposed — pending team review |
| `MEDIUM_VALUE` | 0.25 | 1,000 virtual kr | Proposed — pending team review |
| `HIGH_VALUE` | 0.05 | 8,000 virtual kr | Proposed — pending team review |

In regime `m`, gross opened value is:

```text
X = base_bundle_value * card_value_multiplier(m)
```

These categories, probabilities, and values are synthetic test assumptions.
They are not official rarity names, measured NFL pull rates, or observed resale
values. Draws are independent and with replacement. The sampler does not apply 
fees, mutate the portfolio, choose an action, or reveal unopened outcomes.

As an analytical check, the proposed distribution has expected gross value
`790` virtual kr at multiplier `1.0`, or expected net proceeds of `750.50`
virtual kr with the proposed 5% fee. These are derived checks, not additional
configuration settings.

## 9. Synthetic market model

### 9.1 Regime quotes

| Regime | Sealed ask | Card-value multiplier | Status |
|---|---:|---:|---|
| `LOW` | 800 | 0.75 | Proposed — pending team review |
| `NORMAL` | 1,000 | 1.00 | Proposed — pending team review |
| `HIGH` | 1,200 | 1.60 | Proposed — pending team review |

The net sealed-sale quote is derived as `(1 - selling_fee) * sealed_ask`.
At the proposed 5% fee, the corresponding net quotes are `760`, `950`, and
`1,140` virtual kr for `LOW`, `NORMAL`, and `HIGH`.

### 9.2 Regime transition probabilities

Rows are the current regime and columns are the next regime:

| Current \ Next | `LOW` | `NORMAL` | `HIGH` | Status |
|---|---:|---:|---:|---|
| `LOW` | 0.70 | 0.25 | 0.05 | Proposed — pending team review |
| `NORMAL` | 0.15 | 0.70 | 0.15 | Proposed — pending team review |
| `HIGH` | 0.05 | 0.25 | 0.70 | Proposed — pending team review |

The next market regime is independent of the policy's action. Version 1 starts
in `NORMAL`.

## 10. Randomness and information boundaries

- A scenario seed deterministically creates separate market and pack random
  streams using spawned random-number generators.
- The market path contains `T + 1` regimes.
- A private pack draw is assigned to each of the `T` action timesteps.
- Opening at timestep `t` uses draw `t`; an unused draw is skipped rather than
  shifting future draws.
- Policies compared on the same scenario receive the same market path and the
  same latent draw at any timestep on which they open.
- A policy may read only its public observation and permitted published model
  parameters. It must not access scenario IDs, random seeds, future regimes,
  future draws, or privileged `info` fields.
- Random generators are seeded on reset and are not repeatedly reseeded during
  an episode.

## 11. Public environment contract

The environment follows the Gymnasium interaction convention:

```python
reset(seed=None, options=None) -> (observation, info)
step(action) -> (observation, reward, terminated, truncated, info)
```

At minimum, transition diagnostics must make accounting auditable by recording:

- Requested and executed action.
- Invalid-action flag.
- Timestep and current/next regime.
- Cash and sealed count.
- Gross opening value when realized.
- Fee paid.
- Portfolio values before and after the transition.
- Reward and termination reason.

PF-03 freezes exact field names, data types, and result schemas.

## 12. Policy comparison boundary

Version 1 proposes the following policies:

- `CASH_ONLY`
- `BUY_AND_HOLD`
- `ALWAYS_OPEN`
- `EV_ONE_STEP`
- `DQN`

All policies receive the same initial resources, action limit, prices, fees,
capacity, horizon, terminal liquidation, and held-out scenario manifest. The
DQN does not learn or explore during final evaluation.

Use of an existing DQN library is **unresolved pending teacher confirmation**.
The proposed library is Stable-Baselines3. If existing implementations are not
permitted, the team must record the replacement decision before PF-15/PF-16.

## 13. Configuration requirements

Simulator assumptions must be loaded from validated configuration rather than
being duplicated as unexplained constants across modules.

| Configuration group | Required fields |
|---|---|
| Episode | `horizon`, `initial_cash`, `initial_inventory`, `inventory_capacity` |
| Transactions | `selling_fee` |
| Pack model | Outcome IDs, probabilities, and base gross values |
| Market model | Regime IDs, initial regime, sealed asks, card multipliers, transition matrix |
| Observation | `reference_price`, schema version |
| Reproducibility | Scenario seed or scenario manifest identifier |
| Provenance | Simulator version and configuration hash |

Configuration validation must reject:

- Non-finite or negative cash, prices, multipliers, or outcome values.
- A nonpositive horizon, inventory capacity, or reference price.
- Starting inventory outside `0..inventory_capacity`.
- A selling fee outside `0 <= fee < 1`.
- Pack probabilities that are negative or do not sum to one within tolerance.
- Market-transition probabilities that are negative or whose rows do not sum
  to one within tolerance.
- Missing, duplicated, or unknown regime/action/outcome IDs.

The exact file format and filenames are finalized in PF-02/PF-03. Whatever
format is selected must preserve all approved values and support deterministic
serialization or hashing.

## 14. Versioning and change control

- Proposed pre-freeze simulator version: `0.1`.
- PF-01 approval establishes the target consumed by PF-02 and PF-03.
- A later rule change must be recorded in `docs/decisions.md`, including its
  reason, affected modules/configurations/tests, approvers, and date.
- Changes to action IDs, observation order, accounting, reward, or termination
  invalidate incompatible checkpoints unless an explicit conversion exists.
- After the experimental freeze in PF-20, a simulator change requires a new
  version/configuration hash and rerunning affected comparisons.

## 15. Items requiring team confirmation

The team must resolve or explicitly defer the following during review:

- Approval or replacement of every proposed numerical value.
- Approval of the four-action bundled-opening simplification.
- Approval of action IDs and infeasible-action behavior.
- Approval of step timing and terminal liquidation.
- Permission to use Stable-Baselines3 DQN.
- Rules for sharing source code, results, and model artifacts.
- Configuration file format and dependency-version policy, if required for
  PF-02 to begin.

The authoritative status and sign-off for these items is maintained in
[`docs/decisions.md`](decisions.md).
