# PF-17 learning check: agent sanity checks and diagnostic fixtures

**Purpose:** separate learning bugs in the DQN/training code from a difficult
or genuinely unprofitable environment, before trusting any return measured on
the frozen market. This note documents the diagnostic fixtures, the fixed
smoke budget used against them, what was observed, and the one behavior worth
knowing about when reading action-frequency diagnostics. It is not a
substitute for final experiments on the frozen market (`configs/environment.json`
and the manifests under `configs/splits/`).

## Diagnostic fixtures

`configs/diagnostics/` has two tiny, hand-picked environments. Both fix the
market regime so it can never transition (the transition matrix is the
identity), use a single pack outcome (probability one), and run for a
4-step horizon, so the only thing being tested is whether the agent learns
the known best action — not whether it can also handle market or pack
randomness.

| Fixture | Files | Known best behavior | Why |
| --- | --- | --- | --- |
| Profitable opening | `profitable_opening_environment.json`, `profitable_opening_pack.json` | Always `BUY_PACK` then `OPEN_AND_SELL` | The pack ask is 10.0 and the fee is 5%, but the single outcome's bundle value is 50.0: opening nets about 37.5 profit per buy/open cycle, versus 0 for `HOLD`. |
| Costly trading | `costly_trading_environment.json`, `costly_trading_pack.json` | Always `HOLD` | The fee is 50% and the bundle value is only 2.0, so both `SELL_PACK` (resold at the 10.0 ask, for 5.0 after fee) and `OPEN_AND_SELL` (1.0 after fee) lose money against the 10.0 purchase price; `HOLD` is the only action with zero cost. |

`configs/diagnostics/smoke_agent.yaml` is the fixed smoke budget used against
both fixtures: 1000 environment steps, a compact 16x16 network, and the same
replay/target-network/exploration machinery as full training (`gamma = 1.0`,
per PF-16). Training is fully seeded, so a given fixture and this agent
config reproduce identical metrics and weights every run (see
`tests/test_train.py::test_identical_training_seeds_reproduce_metrics_and_weights`).

## How to run

```powershell
.\.venv\Scripts\python.exe -c "from packfolio.train import train; print(train('configs/diagnostics/profitable_opening_environment.json', 'configs/diagnostics/smoke_agent.yaml', 'runs/diagnostics'))"
.\.venv\Scripts\python.exe -m packfolio.diagnostics --checkpoint <run_directory>/checkpoint.pt --environment-config configs/diagnostics/profitable_opening_environment.json
```

`packfolio.diagnostics.run_greedy_rollout` (used by both the CLI and
`tests/test_agent_diagnostics.py`) rolls out a trained checkpoint greedily,
and raises immediately if the network or environment ever produces a
non-finite value, so "no NaN/Inf in the smoke run" is enforced automatically
rather than only checked after the fact.

## Observed results

With the fixed smoke budget above (seed `20261007`, 20 greedy rollout
episodes at seed `123`):

- **Profitable opening:** the agent always executes `BUY_PACK` then
  `OPEN_AND_SELL`, never `HOLD`s, and reaches the fixture's best possible
  mean return of `0.75` per episode. Q-values stay small (well under 10 in
  magnitude), matching the small reward scale of this fixture.
- **Costly trading:** the agent never executes `BUY_PACK`; its mean return is
  exactly `0.0`, matching the "do nothing" baseline (`HOLD` is as good as it
  gets here).

No learning bug was found in either fixture at this smoke budget: both match
their known best behavior exactly and reproducibly.

## Documented behavior: requested vs. executed action counts

In the costly-trading rollout, the agent's **requested** action is
`SELL_PACK`, not `HOLD` — yet its **executed** action is always `HOLD`, and
the return is still exactly `0.0`. This is not a bug:
`packfolio.env.PackfolioEnv` silently executes `HOLD` whenever the requested
action is infeasible (`SELL_PACK`/`OPEN_AND_SELL` need a sealed pack, which
the agent correctly never buys here), so the environment and the agent agree
on behavior even though the raw requested-action histogram looks like it is
still trying to sell. `packfolio.diagnostics.RolloutDiagnostics` reports both
`requested_action_counts` and `executed_action_counts` for this reason:
**always read the executed distribution when judging what the agent actually
did**, and treat a requested/executed mismatch as expected once
`sealed_count` is zero rather than as a failure.
`tests/test_agent_diagnostics.py::test_smoke_training_learns_the_clearly_costly_trading_fixture`
is the reproducible test case for this behavior.

## Terminal transitions and the horizon

`packfolio.train._bootstrap_target` computes the one-step TD target as
`reward + gamma * next_value * (1 - terminated)`. `terminated` is `1.0`
exactly on the transition that reaches the configured horizon (see
`packfolio.env.PackfolioEnv.step`), so that transition's target excludes the
next-state value regardless of its magnitude — it does not bootstrap beyond
the task horizon. This is covered directly by
`tests/test_agent_diagnostics.py::test_bootstrap_target_excludes_next_value_on_terminated_transitions`
(a terminal row with a deliberately huge next-state value still produces a
target equal to its reward alone) and indirectly by
`tests/test_agent_diagnostics.py::test_diagnostic_episodes_terminate_exactly_at_the_configured_horizon`,
which checks that every rollout episode has exactly `horizon` steps.

## Acceptance criteria

- [x] Terminal transitions do not bootstrap beyond the task horizon — unit
  test on `_bootstrap_target` plus the horizon-length rollout check above.
- [x] No NaN or infinite network outputs appear in the smoke run —
  `run_greedy_rollout` raises `FloatingPointError` on any non-finite Q-value
  or reward, and the smoke tests exercise both fixtures end to end.
- [x] Observed failures have a documented diagnosis or reproducible test
  case — no learning bug was found; the one notable behavior (requested vs.
  executed action counts) is documented above with its test case.
