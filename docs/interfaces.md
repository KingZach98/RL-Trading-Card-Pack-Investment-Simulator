# Module interfaces

This is the current interface on `alternate-dev`: four observation values and
two continuous action values. It replaces the old eight-input DQN interface.

## Environment

```python
from packfolio.environment.trading_env import TradingCardEnv

env = TradingCardEnv(initial_boxes=100)
observation, info = env.reset(seed=None, options=None)
observation, reward, terminated, truncated, info = env.step(action)
```

Use a positive whole number for `initial_boxes`. The default is `1`.
The constructor does not validate this input. Paths are relative to the
repository root.

### Observation

The observation is a NumPy array with shape `(4,)` and dtype `float32`.

| Index | Value | Formula |
|---|---|---|
| 0 | Time passed | `current_step / (price_rows - 1)` |
| 1 | Boxes left | `remaining_boxes / initial_boxes` |
| 2 | Sealed-box price | `current_price / max_price_in_full_filtered_data` |
| 3 | Card market factor | Latest index factor on or before the current date |

The first three space bounds are `[0, 1]`. The last is `[0, infinity)`.
The terminal observation is four zeros.

### Action

Pass a NumPy array with shape `(2,)`, dtype `float32`, and values in `[-1, 1]`.
Integer action IDs are not the interface.

```text
liquidate_fraction = (action[0] + 1) / 2
open_share = (action[1] + 1) / 2
liquidate_count = floor(liquidate_fraction * remaining_boxes)
open_count = floor(open_share * liquidate_count)
sell_count = liquidate_count - open_count
keep_count = remaining_boxes - liquidate_count
```

| Action | Result |
|---|---|
| `[-1, -1]` | Keep all boxes |
| `[1, -1]` | Sell all boxes sealed |
| `[1, 1]` | Open all boxes and sell the contents |
| `[0, 1]` | Open half, rounded down; keep the rest |
| `[1, 0]` | Open half, rounded down; sell the rest sealed |

`convert_action(action)` returns `(open_count, sell_count)`. It checks shape,
but does not check bounds or finite values. Callers must pass valid actions.

### Reset and step

Reset starts at the first price row, restores the starting box count, clears
`realized_value`, and clears the end flag. `options` is unused. The seed reaches
Gymnasium, but not Python's global box-opening RNG.

A step opens and sells at the current date, adds proceeds, then advances one
price row if boxes remain. It returns proceeds divided by `100_000` as reward.
On the last row, it also sells any boxes left. Both end cases return
`terminated=True` and `truncated=False`. Stepping after the end raises
`RuntimeError`.

Reset info contains `date`, `remaining_boxes`, and `realized_value`.
Step info contains:

| Field | Meaning |
|---|---|
| `date` | Date used for this action |
| `opened_boxes`, `sold_boxes` | Counts removed by this action |
| `kept_boxes` | Boxes left after the full step |
| `opened_value`, `sold_value` | USD receipts from this action |
| `realized_value` | Total receipts since reset |
| `liquidated_boxes`, `liquidation_value` | Added only for the forced final sale |

A nonterminal observation describes the next date; step info describes the
date just used.

## Other calls

| Call | Returns or effect |
|---|---|
| `env.get_state()` | Current `date`, `remaining_boxes`, `sealed_price`, and `card_market_factor`; call while active |
| `box_sim.open_box()` | A list of 144 card dictionaries |
| `box_sim.calculate_box_value(box, date)` | USD value using listed prices and the date's factor |
| `data.market_data.get_card_market_factor(date)` | Latest available factor; raises `ValueError` if none exists |
| `card_converter.load_cards(path)` | Card dictionaries from a CSV |
| `model.predict(observation, deterministic=True)` | PPO action array and model state |
| `baseline_policy(env)` | Action array; current baselines receive the environment |
| `evaluate(initial_boxes, model)` | Final realized value; also saves a policy chart |

`box_sim` is in `packfolio.simulator`; `card_converter` is in `packfolio.data`.
The evaluation function is in `packfolio.evaluation.evaluate`.

## Files

| File | Used fields or output |
|---|---|
| `data/2020_panini_prizm_hobby_box_prices.csv` | `date`, `price_usd` |
| `data/2020_panini_prizm_card_index.csv` | `date`, `factor` |
| `data/2020_panini_prizm_cards.csv` | `card`, `ungraded`, `print_run`; prices are CSV text |
| `models/ppo_trading_card.zip` | Saved PPO model |
| `results/ppo_policy_{n}_boxes.png` | Policy actions on the sealed-price chart |
| `results/strategy_comparison_{n}_boxes.png` | Baseline means, spreads, and stored PPO reference |

There are no shared record classes, JSONL run manifests, or schema version
constants in the current code. Do not use the old contracts for this branch.
