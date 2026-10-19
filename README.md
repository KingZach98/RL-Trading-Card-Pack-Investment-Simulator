# Packfolio

Packfolio tests when to keep, open, or sell boxes of 2020 Panini Prizm Football
cards. Each run starts with owned boxes, not a cash budget. The current branch,
`alternate-dev`, uses historical sealed-box prices and Stable-Baselines3 PPO.

Results appear in the terminal and saved charts. There is no app screen.

## Setup

Use Python 3.12. From the repository root, run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib gymnasium stable-baselines3
$env:PYTHONPATH = "$PWD\src"
```

Skip the first two commands if your environment already has these packages.
Set `PYTHONPATH` in each new terminal. Run from the repository root so the
relative data and model paths work. This branch has no dependency or lockfile.

## Run

Run each command on its own:

```powershell
# Check the environment with the Stable-Baselines3 checker.
.\.venv\Scripts\python.exe -m packfolio.environment.trading_env

# Train PPO for 50,000 steps with 100 starting boxes.
.\.venv\Scripts\python.exe -m packfolio.training.train

# Load the saved model and test it with 1, 10, and 100 boxes.
.\.venv\Scripts\python.exe -m packfolio.evaluation.evaluate

# Run the fixed strategies and save comparison charts.
.\.venv\Scripts\python.exe -m packfolio.evaluation.baseline
```

Training saves `models/ppo_trading_card.zip` and replaces that file on later
runs. Train before evaluating, or use an existing model with matching spaces.

Evaluation saves `results/ppo_policy_{n}_boxes.png`. Baselines save
`results/strategy_comparison_{n}_boxes.png`. Here, `n` is the starting box count.

The baselines are keep all, sell now, open all, half open/half sell, and random.
The half-and-half case skips a one-box inventory. Random strategies and those
that open boxes use 100 runs per inventory size.

## Data and results

The `data/` folder holds card prices, sealed-box prices, and a card market index.
The environment uses box-price rows from February 2021 onward. Opened-card
values use listed ungraded prices times the card market factor for that date.

Optional: `python -m packfolio.evaluation.open_eval` runs a large opening study
and saves `results/open_value_distribution.png`. It also opens a plot window.

Keep these limits in mind:

- Reported values are gross proceeds in USD, not profit or percentage returns.
- The baseline chart uses saved PPO numbers in `PPO_VALUES`, not a fresh run.
- Evaluation does not write policy CSVs. `results/plot_policy.py` reads existing ones.
- A reset seed does not seed the box-opening draws.
- This branch has no test source files. The environment checker is the current check.

See [the spec](docs/spec.md) for the rules and [the interfaces](docs/interfaces.md)
for the data passed between modules.
