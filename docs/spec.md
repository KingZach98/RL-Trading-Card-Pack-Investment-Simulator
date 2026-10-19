# Simulator specification

This describes the current code on `alternate-dev`. It replaces the earlier
cash-budget and DQN plan.

## Goal

Compare when to keep boxes sealed, open them, or sell them sealed. Run the same
study with 1, 10, and 100 starting boxes to compare proceeds and the spread of
outcomes. Results apply to this model, not to real-money trading.

## Starting state and market

- The product is a 2020 Panini Prizm Football Hobby Box.
- Each run starts with owned boxes and zero realized proceeds. There is no
  starting cash budget and no buying.
- `TradingCardEnv` defaults to one box. Training starts with 100.
- One step uses one row of monthly sealed-box price data.
- The price path starts in February 2021 and ends at the last CSV row.
- The card market factor comes from the last index row on or before that date.
- There are no fees, storage costs, or cash interest.

The inputs are `data/2020_panini_prizm_hobby_box_prices.csv`,
`data/2020_panini_prizm_cards.csv`, and
`data/2020_panini_prizm_card_index.csv`.

## Decisions

PPO returns two continuous values in `[-1, 1]`. The first sets what share of
the remaining boxes to remove from inventory. The second sets what share of
those boxes to open. Sell the rest sealed and keep all boxes not selected.

Counts round down to whole boxes. With one box, a partial removal can round
to zero. See [interfaces.md](interfaces.md) for the exact formulas.

Opening means selling the contents at once. The agent cannot hold single cards.

## Box openings

A box has 144 card slots: 12 packs with 12 cards each. The sampler starts with
24 rookies and 120 veterans, then replaces slots with numbered parallels,
silver parallels, inserts, and autographs.

The settings request nine numbered cards, two silvers, four inserts, and two
autographs. If a parallel version is missing, its base card stays in place.
Pull weights are model assumptions, not verified manufacturer pull odds.

Box value is the sum of available ungraded card prices times the date's card
market factor. Inserts and cards without a listed price add no value.

## Reward and episode end

```text
step_proceeds = opened_value + sealed_sale_value + forced_final_sale_value
reward = step_proceeds / 100_000
realized_value += step_proceeds
```

The forced final sale adds value only at the end of the price data. Reward
measures receipts, not profit or the change in total inventory value. The
reward sum equals final realized proceeds divided by 100,000.

End the episode when no boxes remain or the last price row has been used.
At the end of the data, sell any remaining boxes at the last sealed-box price.
Return a zero observation, `terminated=True`, and `truncated=False`.
Reject further steps until reset.

## Training and evaluation

- Use Stable-Baselines3 PPO with `MlpPolicy`, `gamma=1.0`, and 50,000 training steps.
- Train with 100 boxes and save `models/ppo_trading_card.zip`.
- Evaluate that same model once for each of 1, 10, and 100 boxes.
- Compare with keep all, sell now, open all, half open/half sell, and random.
- Run the random and opening baselines 100 times per inventory size.
- Print results and save PNG charts. Settings live in the scripts, not config files.

The baseline chart's PPO line uses hard-coded prior results. The evaluator does
not refresh those numbers or write the CSVs used by `results/plot_policy.py`.

## Current limits

- Training and evaluation use the same price path; there is no held-out period.
- Price scaling uses the maximum price from the full dataset, including later dates.
- Box draws use Python's global `random`; `reset(seed=...)` does not ensure replay.
- Card prices use one price list and a market factor, not each card's price history.
- Initial box counts and action bounds lack full checks.
- There are no checked-in test source files on this branch.
