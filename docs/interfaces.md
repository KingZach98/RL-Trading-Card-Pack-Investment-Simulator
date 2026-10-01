# Packfolio Interface Contract

| Field | Value |
|---|---|
| PF task | PF-03 |
| Status | Frozen for Version 1 |
| Interface version | `1.0` |
| Observation schema version | `1.0` |
| Step info schema version | `1.0` |
| Evaluation row schema version | `1.0` |
| Source specification | [`docs/spec.md`](spec.md) |
| Decision record | [`docs/decisions.md`](decisions.md) |
| GitHub issue | [#3 - Freeze module interfaces, observations and result schemas](https://github.com/KingZach98/RL-Trading-Card-Pack-Investment-Simulator/issues/3) |

## 1. Purpose

This document fixes the data passed between Packfolio modules. It gives each
team member one set of names, types, and rules to build against.

The approved simulator rules stay in `docs/spec.md`. This document does not
repeat the model values or define how each module must implement its work.

The matching Python types live in `src/packfolio/types.py`.

## 2. Shared rules

- Shared records are immutable values. Code creates a new record for each
  update.
- Python `float` values hold simulator amounts. NumPy stores them as `float64`
  during math.
- Observations use NumPy `float32` values.
- Counts and indexes use Python `int` values.
- All numbers must be finite. JSON must not contain `NaN` or `Infinity`.
- An optional Python value uses `None`. Its JSON value is `null`. CSV leaves
  the field empty.
- Enum names use upper snake case. JSON and CSV store the enum name as text,
  except actions, which use their integer IDs.

The following constants name the current contracts:

```python
INTERFACE_VERSION = "1.0"
OBSERVATION_SCHEMA_VERSION = "1.0"
STEP_INFO_SCHEMA_VERSION = "1.0"
EVALUATION_ROW_SCHEMA_VERSION = "1.0"
```

Shared record constructors validate their own fields in `__post_init__`.
Records that cross a JSON or Gymnasium boundary provide `to_dict()` and
`from_dict()` methods. A loader rejects missing fields, extra fields, and values
of the wrong type. A float field may load a JSON integer or float, but not a
boolean. It stores the result as a float. An integer field rejects booleans.

## 3. Actions

`Action` is an `IntEnum` with these fixed values:

| ID | Name | Feasible when |
|---:|---|---|
| 0 | `HOLD` | Always |
| 1 | `BUY_PACK` | Cash covers the ask and inventory is below capacity |
| 2 | `OPEN_AND_SELL` | The portfolio has at least one sealed pack |
| 3 | `SELL_PACK` | The portfolio has at least one sealed pack |

`env.step()` accepts one integer scalar. It accepts `Action`, Python integer,
and NumPy integer values. It rejects booleans and non-integer values with
`TypeError`. It rejects values outside `0..3` with `ValueError`.

An in-range action can still be infeasible. The environment then executes
`HOLD`, advances time, and applies no extra penalty. Step info records both the
requested action and the executed action.

A policy uses this public call:

```python
choose_action(observation) -> Action
```

The policy must return one of the four actions. An adapter must turn output
from an external model into `Action` before it reaches the environment.

## 4. Observation

The observation is a NumPy array with shape `(8,)` and dtype `float32`. Its
order is part of the public contract.

| Index | `ObservationIndex` name | Value | Bound |
|---:|---|---|---|
| 0 | `CASH_RATIO` | `cash / initial_cash` | `0` to infinity |
| 1 | `SEALED_COUNT_RATIO` | `sealed_count / inventory_capacity` | `0` to `1` |
| 2 | `PACK_ASK_RATIO` | `pack_ask / reference_price` | `0` to infinity |
| 3 | `CARD_VALUE_MULTIPLIER` | Current card-value multiplier | `0` to infinity |
| 4 | `REMAINING_STEPS_RATIO` | `remaining_steps / horizon` | `0` to `1` |
| 5 | `REGIME_LOW` | `1` in `LOW`, else `0` | `0` or `1` |
| 6 | `REGIME_NORMAL` | `1` in `NORMAL`, else `0` | `0` or `1` |
| 7 | `REGIME_HIGH` | `1` in `HIGH`, else `0` | `0` or `1` |

Indexes 5 through 7 are one-hot. Exactly one of them must be `1`.

The Gymnasium observation space uses these config-safe bounds:

```python
low = [0, 0, 0, 0, 0, 0, 0, 0]
high = [inf, 1, inf, inf, 1, 1, 1, 1]
```

The observation builder requires `initial_cash > 0`,
`inventory_capacity > 0`, `reference_price > 0`, and `horizon > 0`. It must not
clip a cash ratio above `1`.

At reset, `REMAINING_STEPS_RATIO` is `1`. After the last step, it is `0`. The
terminal observation is built after terminal liquidation. Its sealed count is
zero, and it uses the terminal market snapshot.

## 5. Shared records

The classes in this section are frozen dataclasses with slots. They contain
data only and do not hold a reference to the environment or a mutable
portfolio.

### 5.1 Market enums and snapshot

`MarketRegime` is a `StrEnum` with these values:

- `LOW`
- `NORMAL`
- `HIGH`

`MarketSnapshot` has these fields:

| Field | Type | Rule |
|---|---|---|
| `regime` | `MarketRegime` | Current visible regime |
| `pack_ask` | `float` | Finite and not negative |
| `card_value_multiplier` | `float` | Finite and not negative |

The snapshot does not contain a sale fee or net sale quote. `portfolio.py`
owns that accounting. A market snapshot is public between simulator modules,
but a policy receives the observation rather than this record.

### 5.2 Pack outcome

`PackOutcomeId` is a `StrEnum` with these values:

- `LOW_VALUE`
- `MEDIUM_VALUE`
- `HIGH_VALUE`

`PackOutcome` has these fields:

| Field | Type | Rule |
|---|---|---|
| `outcome_id` | `PackOutcomeId` | Sampled whole-pack outcome |
| `base_gross_value` | `float` | Value before the market multiplier |
| `gross_value` | `float` | Value after the market multiplier and before fees |

Both values must be finite and not negative. This record never contains a fee,
net proceeds, cash, or inventory. `packs.py` returns the result and does not
apply it to a portfolio.

### 5.3 Portfolio snapshot

`PortfolioSnapshot` has these fields:

| Field | Type | Rule |
|---|---|---|
| `cash` | `float` | Finite and not negative |
| `sealed_count` | `int` | Not negative |

Only `portfolio.py` may calculate a new portfolio snapshot. The environment
stores the current snapshot and replaces it with the returned value. Portfolio
operations check the inventory capacity because it is not part of this record.

### 5.4 Portfolio update

`PortfolioUpdate` has these fields:

| Field | Type | Rule |
|---|---|---|
| `portfolio` | `PortfolioSnapshot` | New state after one accounting operation |
| `fee_paid` | `float` | Finite fee charged by that operation; not negative |

An action and terminal liquidation are separate accounting operations. On the
last step, the environment adds both `fee_paid` values when it builds
`StepInfo`.

### 5.5 Private scenario

`Scenario` has these fields:

| Field | Type | Rule |
|---|---|---|
| `market_path` | `tuple[MarketSnapshot, ...]` | Has `horizon + 1` snapshots |
| `pack_outcomes` | `tuple[PackOutcome, ...]` | Has one private outcome for each of the `horizon` action steps |

For action step `t`, the environment uses `market_path[t]`,
`pack_outcomes[t]` when a pack opens, and `market_path[t + 1]` after the market
advance. The full scenario is private. It must never reach a policy.

The record checks that `len(market_path) == len(pack_outcomes) + 1`.
`scenarios.py` also checks both lengths against the configured horizon.

### 5.6 Public module calls

These calls form the small public API between simulator modules. `MarketConfig`
and `PackConfig` mean the validated, read-only config sections from
`config.py`. PF-03 does not set their file format.

```python
# market.py
def snapshot_for(regime: MarketRegime,config: MarketConfig,) -> MarketSnapshot: ...

def advance_market(current_regime: MarketRegime,rng: numpy.random.Generator,config: MarketConfig,) -> MarketSnapshot: ...

# packs.py
def draw_pack_outcome(rng: numpy.random.Generator,market: MarketSnapshot,config: PackConfig,) -> PackOutcome: ...

# portfolio.py
def apply_action(portfolio: PortfolioSnapshot,action: Action,market: MarketSnapshot,*,selling_fee: float, inventory_capacity: int,
    pack_outcome: PackOutcome | None = None,) -> PortfolioUpdate: ...

def liquidate(portfolio: PortfolioSnapshot,market: MarketSnapshot,*,selling_fee: float,) -> PortfolioUpdate: ...

def liquidation_value( portfolio: PortfolioSnapshot,market: MarketSnapshot,*,selling_fee: float,) -> float: ...

# scenarios.py
def build_scenario(seed: int, horizon: int, market_config: MarketConfig, pack_config: PackConfig,) -> Scenario: ...
```

`snapshot_for()` does not draw a random value. `advance_market()` draws the
next regime once and returns its snapshot. This keeps the initial snapshot from
causing an extra market draw.

`draw_pack_outcome()` draws one outcome and applies the current market's card
value multiplier. `build_scenario()` calls it once for every action step, even
when the future policy will not open a pack. The scenario builder creates
separate market and pack generators from the scenario seed.

`apply_action()` receives an action that the environment has marked feasible.
It requires a pack outcome only for `OPEN_AND_SELL`. It raises `ValueError` when
these preconditions are false. `liquidate()` removes all sealed packs.
`liquidation_value()` does not change the snapshot.

## 6. Environment boundary

The environment follows the Gymnasium calls in the approved specification:

```python
reset(seed=None, options=None) -> (observation, info)
step(action) -> (observation, reward, terminated, truncated, info)
```

During evaluation, `evaluate.py` calls `reset(seed=scenario_seed)` and keeps the
matching `scenario_id`. The environment asks `scenarios.py` to build the full
private `Scenario` before it returns the first observation. The same seed and
config must build the same scenario. `reset()` returns an empty `info` mapping
in Version 1.

`step()` returns its normal Gymnasium values. Its `info` mapping uses the
`StepInfo` fields below. Policies must not receive this mapping.

`TerminationReason` is a `StrEnum` with these values:

- `HORIZON`

Version 1 has no outside truncation rule. It returns `truncated=False`. A later
truncation rule must add its reason through a schema change.

`StepInfo` has these fields:

| Field | Type | Rule |
|---|---|---|
| `schema_version` | `str` | Always `STEP_INFO_SCHEMA_VERSION` |
| `step_index` | `int` | Zero-based index of the requested action |
| `requested_action` | `Action` | Action passed to `step()` |
| `executed_action` | `Action` | Action applied by the environment |
| `action_was_infeasible` | `bool` | True when an in-range action became `HOLD` |
| `regime_before` | `MarketRegime` | Regime used to execute the action |
| `regime_after` | `MarketRegime` | Regime after the market transition |
| `cash_before` | `float` | Cash before the action |
| `cash_after` | `float` | Cash after the full step |
| `sealed_count_before` | `int` | Sealed packs before the action |
| `sealed_count_after` | `int` | Sealed packs after the full step |
| `pack_outcome_id` | `PackOutcomeId \| None` | Present only when a pack opens |
| `gross_opened_value` | `float \| None` | Present only when a pack opens |
| `fee_paid` | `float` | Sum of all fees in the step, including terminal liquidation |
| `portfolio_value_before` | `float` | Liquidation value before the action |
| `portfolio_value_after` | `float` | Liquidation value after the full step |
| `reward` | `float` | Reward returned by `step()` |
| `termination_reason` | `TerminationReason \| None` | Present only on the last step |

The Python `info` mapping stores action IDs as integers and enum values as their
text names. It uses `None` for the two pack fields when no pack opens. JSON
serialization changes `None` to `null`. An out-of-range action raises before
the environment creates `StepInfo`.

## 7. Evaluation row

One `EvaluationRow` records one complete policy run on one scenario. The row is
flat so it can be saved as JSON or CSV without a custom nested format.

| Field | Type | Rule |
|---|---|---|
| `schema_version` | `str` | Always `EVALUATION_ROW_SCHEMA_VERSION` |
| `interface_version` | `str` | Version that defines actions and shared records |
| `simulator_version` | `str` | Installed Packfolio version |
| `observation_schema_version` | `str` | Observation version used by the policy |
| `policy_id` | `str` | Stable policy name, such as `CASH_ONLY` or `DQN` |
| `model_id` | `str \| None` | Model or checkpoint name; null for fixed baselines |
| `training_seed` | `int \| None` | Training seed; null for fixed baselines |
| `scenario_id` | `str` | Stable ID from the evaluation manifest |
| `scenario_seed` | `int` | Seed used to build the scenario |
| `config_hash` | `str` | SHA-256 hash of the validated config |
| `git_commit` | `str` | Commit used for the run |
| `episode_steps` | `int` | Number of completed environment steps |
| `initial_portfolio_value` | `float` | Liquidation value at reset |
| `final_portfolio_value` | `float` | Liquidation value after the last step |
| `cumulative_reward` | `float` | Sum of step rewards |
| `requested_hold_count` | `int` | Requested action count |
| `requested_buy_pack_count` | `int` | Requested action count |
| `requested_open_and_sell_count` | `int` | Requested action count |
| `requested_sell_pack_count` | `int` | Requested action count |
| `executed_hold_count` | `int` | Executed action count |
| `executed_buy_pack_count` | `int` | Executed action count |
| `executed_open_and_sell_count` | `int` | Executed action count |
| `executed_sell_pack_count` | `int` | Executed action count |
| `infeasible_action_count` | `int` | Number of actions converted to `HOLD` |
| `terminated` | `bool` | Gymnasium task termination flag |
| `truncated` | `bool` | Gymnasium outside-stop flag |
| `termination_reason` | `TerminationReason` | Why the run ended |

All counts must be zero or greater. Each group of four action counts must sum
to `episode_steps`. `infeasible_action_count` cannot exceed `episode_steps`.
A completed Version 1 row has `terminated=True`, `truncated=False`, and
`termination_reason=HORIZON`.

## 8. Information boundary

The word "public" has two meanings in the code. A public Python type may be
used by several simulator modules. It is not always safe to give that value to
a policy.

| Data | Policy | Environment | Evaluator |
|---|---:|---:|---:|
| Current observation | Read | Build | Record if needed |
| Published static model config | Read | Read | Read |
| `MarketSnapshot` | No direct access | Read | Record if needed |
| Scenario ID | No | No | Read |
| Scenario seed | No | Use at reset | Read |
| Future market regimes | No | Use one step at a time | No during a run |
| Future or unopened pack outcomes | No | Use only for the current open action | No during a run |
| `StepInfo` | No | Build | Read |
| `EvaluationRow` | No | No | Build |

Published static config includes the approved model rules needed by a fixed
baseline. It must not include scenario IDs, seeds, generated market paths, or
generated pack draws.

The evaluator must call `choose_action()` with the observation only. It must
not pass `info`, the environment object, or the scenario object to a policy.

## 9. Module boundaries and state ownership

These are the required inputs and outputs at each module boundary. The calls in
Section 5.6 are public. Other helper names inside a module are not part of the
public contract.

| Module | Inputs | Output and owned state | Must not do |
|---|---|---|---|
| `scenarios.py` | Scenario seed, horizon, and validated model config | A `Scenario`; seed setup, separate random streams, and timestep mapping | Expose private future values to a policy |
| `market.py` | Current regime, market random stream, and validated market config | A `MarketSnapshot`; regime transition rules | Read or update cash, inventory, or actions |
| `packs.py` | Pack random stream, current `MarketSnapshot`, and validated pack config | A `PackOutcome`; whole-pack sampling | Apply fees or update cash and inventory |
| `portfolio.py` | `PortfolioSnapshot`, executed `Action`, `MarketSnapshot`, optional `PackOutcome`, fee, and capacity | A `PortfolioUpdate`; cash, inventory, fees, sale proceeds, and liquidation math | Advance time or sample random values |
| `env.py` | Action plus its private `Scenario` and current snapshots | Gymnasium step values; timestep, action checks, call order, reward, and flags | Reimplement market, pack, or portfolio math |
| `baselines.py` | Observation and published static config | `Action`; fixed policy state | Read private scenario or diagnostic data |
| `agents/dqn_agent.py` | Observation | `Action`; DQN state and output adapter | Read private scenario or diagnostic data |
| `evaluate.py` | Policy, scenario manifest, and environment factory | `EvaluationRow` values; scenario loop, records, and action counts | Change simulator state outside `reset()` and `step()` |

`market.py` and `packs.py` never receive a portfolio object. They return their
own value records. `env.py` passes those values to `portfolio.py`, which returns
a `PortfolioUpdate` with a new immutable snapshot.

The chosen action cannot affect the market transition. The environment
advances the market once on every step, including an infeasible action.

## 10. Exchange order

One evaluation step follows this data flow:

1. `evaluate.py` selects a scenario and calls `env.reset()`.
2. `env.py` returns the Version 1 observation.
3. The evaluator passes only that observation to `choose_action()`.
4. The evaluator passes the returned action to `env.step()`.
5. `env.py` coordinates market, pack, and portfolio work.
6. `env.py` returns the next observation, reward, flags, and `StepInfo` mapping.
7. `evaluate.py` records the step and builds one `EvaluationRow` at the end.

## 11. Deterministic fixtures

PF-03 uses three small JSON fixtures:

| File | Purpose |
|---|---|
| `tests/fixtures/market_snapshot_v1.json` | Proves that market data builds the expected observation values |
| `tests/fixtures/pack_outcome_v1.json` | Proves that a pack result can pass to the environment without portfolio data |
| `tests/fixtures/contract_exchange_v1.json` | Proves that a one-step completed episode uses one policy, step, and evaluation contract |

The market and pack fixtures have a top-level `interface_version`. The exchange
fixture has an `interface_version` plus nested `schema_version` values for its
step info and evaluation row. Tests load the files without random draws. The
fixture values are examples, not new simulator defaults.

The contract test must prove these points:

- A policy receives eight `float32` values and returns a valid `Action`.
- The environment-facing data can create a valid `StepInfo` value.
- The evaluator can combine that step with run metadata and create a valid
  `EvaluationRow`.
- Market and pack calls accept no portfolio object and return no portfolio
  fields.
- Missing, extra, or wrongly typed required fields fail clearly.

## 12. Version changes

A change is breaking when it changes an action ID, observation position,
observation meaning, dtype, required field name, required field type,
information boundary, or state owner. A breaking change raises the related
major schema version, such as `1.0` to `2.0`.

Use this version map:

| Change | Version to update |
|---|---|
| Action, enum, or shared record | `INTERFACE_VERSION` |
| Observation field, order, dtype, or meaning | `OBSERVATION_SCHEMA_VERSION` |
| `StepInfo` field or meaning | `STEP_INFO_SCHEMA_VERSION` |
| `EvaluationRow` field or meaning | `EVALUATION_ROW_SCHEMA_VERSION` |

Adding an optional field raises the minor version. A text fix that does not
change behavior needs no version change.

Each schema change must update this document, `types.py`, fixtures, and contract
tests in the same commit. A breaking observation or action change also marks
older policy checkpoints as incompatible unless a tested conversion exists.
