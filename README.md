# Packfolio

Python package: `packfolio` (Python 3.12). Core dependencies are NumPy 2.2.6,
Gymnasium 1.1.1, and Matplotlib 3.10.3. The optional DQN dependency is
PyTorch 2.7.1; tests use pytest 8.4.1. Pack sampling, market transitions,
shared contracts, reproducible scenarios, and the Gymnasium environment are
implemented. The DQN agent and training modules are still placeholders.

## Local setup

Install Python 3.12 and clone the repository. From the repository root, on
Windows (PowerShell):

```powershell
py -m pip install --user uv==0.9.1
py -m uv sync --locked --extra dev
.\.venv\Scripts\python.exe -m pytest -q
```

On macOS or Linux, install uv 0.9.1 outside the project environment using the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv sync --locked --extra dev
.venv/bin/python -m pytest -q
```

Add `--extra agent` to `uv sync` when developing the DQN agent. PyTorch is
not installed by default. The committed `uv.lock` pins the full dependency
graph; use `--locked` on `uv sync` to reject a stale lockfile. Run
`uv lock` and commit the updated lock whenever dependencies change.

On Windows, keep uv outside `.venv` and call it with `py -m uv`. A sync can
[remove tools installed inside the managed environment](https://docs.astral.sh/uv/concepts/projects/sync/#handling-of-extraneous-packages). The project itself
still runs with `.venv\Scripts\python.exe`, which uses Python 3.12.

Local secrets belong in `.env` (or `.env.*`), and generated results in
`outputs/`, `runs/`, or `checkpoints/`; these paths and model weights are
ignored by Git. Put versioned, non-secret scenario settings in `configs/`.

## Reproducible DQN training

The simulator assumptions are loaded from `configs/environment.json`;
independent DQN hyperparameters are loaded from `configs/agent.yaml`. From the
repository root, install the optional agent dependencies and start a run with:

```powershell
.\.venv\Scripts\uv.exe sync --locked --extra dev --extra agent
.\.venv\Scripts\uv.exe run --locked --extra dev --extra agent python -m packfolio.train
```

Override the inputs with `--environment-config`, `--agent-config`, and
`--output-root`. Each invocation creates a new unique run directory containing
the exact input files, a fully resolved environment configuration, a
`run_manifest.json`, per-episode `training_metrics.jsonl`, and a reloadable
`checkpoint.pt`. The manifest records the Git commit and dirty state, package
versions, seeds, runtime, environment config hash, and actual environment-step
count. Training uses replay sampling, a periodically synchronized target
network, epsilon-greedy exploration, and `gamma = 1.0`; invalid actions are
allowed by design and their per-episode rate is logged. Checkpoints can be
loaded for deterministic policy inference with
`packfolio.agents.dqn_agent.load_trained_agent`.

## Environment compatibility checks (PF-11)

The selected agent stack is a custom DQN using PyTorch, not Stable-Baselines3.
Install the locked agent dependencies, then run the mandatory check from the
repository root on Windows:

```powershell
py -m uv sync --locked --extra dev --extra agent
.\.venv\Scripts\python.exe -m packfolio.check_env
```

The command runs Gymnasium's checker, then sends observations through a small,
untrained CPU network with eight inputs and four action outputs. It converts
the selected action to a scalar, runs a full episode, checks the end guard,
and replays the same seed. It also checks conversion of rewards to tensors.
This follows the tensor/action boundary in the
[PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).
It does not train a DQN or test the future agent's quality.

Expect `Gymnasium and PyTorch boundary checks passed (100 steps, replay matched)`
with the shipped config. The command exits with an error if PyTorch is missing.
It prints the config hash, seed, inventory capacity, and reference price.
Use `--config`, `--seed`, `--inventory-capacity`, and `--reference-price` to
check other settings. Only the known warning about unbounded observation
maxima is hidden; other Gymnasium checker warnings stay visible.

Run the saved environment checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_env.py tests/test_env_actions.py tests/test_env_rewards.py tests/test_env_terminal.py tests/test_accounting.py tests/test_check_env.py -p no:cacheprovider
```

PyTorch-specific pytest cases skip when the optional dependency is absent.
Those skips do not count as a compatibility pass: run the mandatory command
with the agent extra before signing off PF-11. Nasir's actual DQN integration
still needs its own tests when that agent exists.

## Whole-pack sampling and expected value

`configs/nfl_pack.json` contains a **synthetic** NFL-themed five-card pack
distribution, not real product odds or market prices. Each outcome lists all
five cards; `base_bundle_value` is the gross value of that entire bundle at a
market multiplier of one, in a consistent simulation currency unit.

```python
import numpy as np

from packfolio.packs import expected_gross_value, load_pack_config, sample_pack

config = load_pack_config("configs/nfl_pack.json")
rng = np.random.default_rng(42)
draw = sample_pack(config, market_multiplier=1.2, rng=rng)
print(draw.outcome_id, draw.gross_bundle_value, draw.contents)
ev = expected_gross_value(config, market_multiplier=1.2)  # 18.9
```

Run the example from the repository root or pass your own configuration path.
The loader requires a pack identifier, a positive `cards_per_pack`, and nonempty
outcomes with unique identifiers, finite nonnegative probabilities and bundle
values, and exactly `cards_per_pack` card descriptions. Probabilities must sum
to one with **absolute tolerance `1e-9` and zero relative tolerance**. Accepted
rounding error is normalized consistently for sampling and analytical EV.
Invalid configuration or multipliers raise an explicit error; zero multipliers
are allowed. Reusing the same configuration and restoring a NumPy generator's
state reproduces the draws; no global random state is used.

The sampler returns a `PackDraw` with `outcome_id`, `gross_bundle_value`, and
the full `contents`. Gross value is `base_bundle_value * market_multiplier`.
Analytical EV is the probability-weighted sum of these gross bundle values;
it does not sample or consume random state.

**Handoff:** Zachery applies selling fees once outside this module, when a sale
occurs, and handles cash changes there. Saki uses `expected_gross_value` for
the EV baseline. Neither function deducts fees, pack purchase costs, or changes
cash.

## Reproducible scenarios

See [configuration and split handoff](configs/README.md) for seed ownership,
versioning, paired draw slots, and the policy privacy boundary.

```python
from packfolio.scenarios import Scenario, load_split_manifest

training = load_split_manifest("configs/splits/training.json")
scenario = Scenario(training.config, training.scenarios[0])
current_market = scenario.current_market
draw = scenario.open_pack()  # current timestep, first opening slot
next_market = scenario.advance()
```

Nasir receives separate training and validation manifests; Saki controls the
final-test manifest and validates all three splits for overlap before final
evaluation. Opening extra packs cannot advance the market RNG or shift pack
draw slots at future timesteps. Scenario identifiers encode the configuration
hash and root seed; retain the matching config and simulator version for replay.
