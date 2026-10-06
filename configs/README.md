# Reproducible experiment configuration

`environment.json` is an editable synthetic starter: 100 timesteps, 100 initial
cash units, a 5% selling fee, and low/normal/high asks of 8/10/12. Market quotes
and pack values use the same currency unit. It references `nfl_pack.json`
relative to its own directory. These settings do not change the existing
reference market defaults. The fee is configuration only; scenario draws
remain gross, with fees and cash changes handled by the portfolio/environment.

`agent.yaml` contains DQN-only settings (seed, step budget, network size,
replay, optimizer, target updates, exploration, and gradient clipping). It is
loaded separately from the environment file, so changing a learning
hyperparameter does not modify market assumptions or their configuration hash.
The discount factor is fixed at `1.0` to match the final portfolio value
objective. Run training from the repository root with
`uv run --locked --extra agent --extra dev python -m packfolio.train`; every run
stores copies of both input files, the resolved environment, its checkpoint,
episode metrics, and a manifest in a unique directory under `runs/`.

`diagnostics/` holds small, hand-picked environment/pack/agent fixtures with a
known best action (always open, or always hold), used for PF-17 agent smoke
tests rather than final experiments. See
[`docs/learning_check.md`](../docs/learning_check.md) for what each fixture
tests, the fixed smoke budget, and what was observed when they last ran.

Split manifests are separate files, with no implicit final-test loading:

| Manifest | Owner | Seeds | Purpose |
| --- | --- | --- | --- |
| `splits/training.json` | Nasir | 1001-1010 | Fit policies |
| `splits/validation.json` | Nasir | 2001-2003 | Select/tune policies |
| `splits/final_test.json` | Saki | 3001-3003 | Final paired evaluation only |

Saki controls the final-test manifest and must keep it out of training/tuning.
The committed starter is not a secret or an access-control mechanism. For a
blind final test, Saki should retain the actual final manifest outside the
training workspace and distribute only final evaluation results.

Each manifest pins the resolved environment content with `config_hash`, a
SHA-256 digest covering horizon, cash, fee, market settings, and complete pack
configuration. Loaders reject unknown/missing fields and invalid values; pack
and market distributions retain their respective probability tolerances.
Editing behavior requires intentionally updating the hash in applicable
manifests. Paths and JSON key order are not part of the hash. Use
`load_environment_config(path).config_hash` to obtain the new digest.
Keep old configs and manifests to reproduce published experiments.

Before an experiment, the trusted evaluation coordinator calls
`validate_split_manifests(training, validation, final_test)` to reject duplicate
splits, overlapping root seeds (even across different configs), and overlapping
scenario identities. Nasir loads only training/validation in the training job;
Saki performs the all-split validation and loads final test in the final job.

A scenario ID is `<config_hash>:<root_seed>`; pair it with
`packfolio-scenarios-v1` and the hash-matching configuration to regenerate a
scenario using `ScenarioSpec.from_id`. Root seeds are integers in `[0, 2**64)`.
The version fixes canonical hashing, SHA-256 domain separation, explicit
NumPy PCG64 generation, and draw ordering; use the locked NumPy 2.2.6 runtime
and retain the implementation for that version. Reject unknown versions
rather than silently replaying with new rules. Bump the simulator version
when random generation, transition, sampling, or timing semantics change.

## Paired evaluation and privacy

Time begins at zero. At timestep `t`, `Scenario.open_pack()` uses slot
`(t, opening_index)`, starting from index zero. The same scenario shares the
same latent outcome for matching slots across strategies. Extra openings
use extra slots, and skipped openings do not shift future timesteps.
`Scenario.advance()` consumes one market transition and resets the opening
index. At `t == horizon` the market snapshot is terminal; both further
openings and advancing raise an error.

Market and pack entropy are SHA-256 hashes of compact JSON arrays containing
the simulator version, scenario ID, and distinct `"market"`/`"pack"` domains.
Pack entropy additionally includes timestep and opening index. Neither the
market stream nor later opening slots depend on how many packs were opened
earlier. No future pack outcomes are precomputed or exposed.

The scenario object is **trusted simulator state**, not a policy observation:
pass policies only the current observation and already-realized results.
Do not pass the scenario object, manifests, seed, or replay recipe to a policy.
The public scenario API has only current timestep, current market, terminal
status, `open_pack()`, and `advance()`; RNGs and replay data are private.
Python attribute privacy is not protection against malicious introspection.
The gym environment is still a placeholder; its implementation must enforce
this observation boundary, slot accounting, and action timing.