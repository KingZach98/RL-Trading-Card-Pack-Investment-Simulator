# Packfolio

Python package: `packfolio` (Python 3.12). Core dependencies are NumPy 2.2.6,
Gymnasium 1.1.1, and Matplotlib 3.10.3. The optional DQN dependency is
PyTorch 2.7.1; tests use pytest 8.4.1. The source modules are currently
placeholders; this foundation only provides packaging and import checks.

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
