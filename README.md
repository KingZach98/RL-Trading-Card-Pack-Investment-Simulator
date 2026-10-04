# Packfolio

Python package: `packfolio` (Python 3.12). Core dependencies are NumPy 2.2.6,
Gymnasium 1.1.1, and Matplotlib 3.10.3. The optional DQN dependency is
PyTorch 2.7.1; tests use pytest 8.4.1. The pack sampler is implemented;
the other simulator modules are currently placeholders.

## Local setup

Install Python 3.12 and clone the repository. From the repository root, on
Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install uv==0.9.1
.\.venv\Scripts\uv.exe sync --locked --extra dev
.\.venv\Scripts\uv.exe run --locked pytest -q
```

On macOS or Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install uv==0.9.1
.venv/bin/uv sync --locked --extra dev
.venv/bin/uv run --locked pytest -q
```

Add `--extra agent` to `uv sync` when developing the DQN agent. PyTorch is
not installed by default. The committed `uv.lock` pins the full dependency
graph; keep `--locked` on installs and test runs to detect stale locks. Run
`uv lock` and commit the updated lock whenever dependencies change.

Local secrets belong in `.env` (or `.env.*`), and generated results in
`outputs/`, `runs/`, or `checkpoints/`; these paths and model weights are
ignored by Git. Put versioned, non-secret scenario settings in `configs/`.

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
