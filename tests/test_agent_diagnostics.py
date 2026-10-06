"""PF-17 agent sanity checks: diagnostic fixtures, smoke training, and traces.

These tests separate learning bugs from a difficult or genuinely unprofitable
environment by training on tiny fixtures with a known best action (see
``configs/diagnostics``) for a fixed, small step budget, then inspecting the
resulting action frequencies, reward scale, and Q-values. A poor return on
the full frozen market is not evidence of a bug by itself; these fixtures
are the comparison point. See ``docs/learning_check.md`` for the written
diagnosis of what was observed when these fixtures were last run.
"""

import math
from pathlib import Path

import pytest

from packfolio.config import load_environment_config
from packfolio.diagnostics import run_greedy_rollout
from packfolio.train import train
from packfolio.types import Action


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS_DIR = REPOSITORY_ROOT / "configs" / "diagnostics"
SMOKE_AGENT_CONFIG = DIAGNOSTICS_DIR / "smoke_agent.yaml"


def test_bootstrap_target_excludes_next_value_on_terminated_transitions():
    """A terminal transition's TD target must not bootstrap past the horizon.

    The next-state value is huge (1e6) on both rows; for the terminated row
    it must be fully excluded regardless of that magnitude, while the
    non-terminated row still bootstraps normally.
    """
    torch = pytest.importorskip("torch")
    from packfolio.train import _bootstrap_target

    reward_tensor = torch.tensor([1.0, 2.0])
    next_values = torch.tensor([1.0e6, 1.0e6])
    terminated_tensor = torch.tensor([1.0, 0.0])

    target = _bootstrap_target(
        reward_tensor=reward_tensor,
        next_values=next_values,
        terminated_tensor=terminated_tensor,
        gamma=1.0,
    )

    assert torch.isfinite(target).all()
    assert target[0].item() == pytest.approx(1.0)
    assert target[1].item() == pytest.approx(2.0 + 1.0e6)


def _train_smoke_checkpoint(fixture_name: str, tmp_path: Path) -> Path:
    environment_config_path = DIAGNOSTICS_DIR / f"{fixture_name}_environment.json"
    run_directory = train(
        environment_config_path, SMOKE_AGENT_CONFIG, tmp_path / fixture_name
    )
    return run_directory / "checkpoint.pt"


def test_smoke_training_learns_the_clearly_profitable_opening_fixture(tmp_path):
    """Buying then opening is always profitable here; HOLD is clearly wrong.

    A fixed smoke budget (configs/diagnostics/smoke_agent.yaml, 1000 steps)
    reliably learns to alternate BUY_PACK/OPEN_AND_SELL and never holds.
    Training is fully seeded, so this run (and its exact metrics) replay.
    """
    pytest.importorskip("torch")
    environment_config = load_environment_config(
        DIAGNOSTICS_DIR / "profitable_opening_environment.json"
    )
    checkpoint_path = _train_smoke_checkpoint("profitable_opening", tmp_path)

    diagnostics = run_greedy_rollout(
        checkpoint_path, environment_config, episodes=20, seed=123
    )

    assert all(
        math.isfinite(value)
        for episode in diagnostics.episodes
        for row in episode.q_values
        for value in row
    )
    # Reward scale sanity: this fixture's best possible per-episode return is
    # 0.75 (see configs/diagnostics/profitable_opening_pack.json), so sane
    # Q-values for a 4-step horizon must stay small, not explode.
    assert abs(diagnostics.min_q_value) < 10.0
    assert abs(diagnostics.max_q_value) < 10.0
    assert diagnostics.executed_action_frequency(Action.HOLD) == 0.0
    assert set(diagnostics.executed_action_counts) == {
        Action.BUY_PACK,
        Action.OPEN_AND_SELL,
    }
    assert diagnostics.mean_episode_return == pytest.approx(0.75)


def test_smoke_training_learns_the_clearly_costly_trading_fixture(tmp_path):
    """Any trade loses money here; HOLD strictly dominates every trading action.

    The agent correctly learns to never buy, so sealed_count stays at zero
    for the whole episode. At that point SELL_PACK/OPEN_AND_SELL/HOLD are all
    infeasible-or-equivalent requests: the environment silently executes
    HOLD when an action is infeasible (see packfolio.env.PackfolioEnv), so
    the requested action can diverge from the executed one even though the
    agent has behaved correctly. This is a documented quirk of reading
    "requested" action frequencies in isolation, not a learning bug.
    """
    pytest.importorskip("torch")
    environment_config = load_environment_config(
        DIAGNOSTICS_DIR / "costly_trading_environment.json"
    )
    checkpoint_path = _train_smoke_checkpoint("costly_trading", tmp_path)

    diagnostics = run_greedy_rollout(
        checkpoint_path, environment_config, episodes=20, seed=123
    )

    assert diagnostics.executed_action_frequency(Action.BUY_PACK) == 0.0
    assert set(diagnostics.executed_action_counts) == {Action.HOLD}
    assert diagnostics.mean_episode_return == pytest.approx(0.0)
    # Documents the requested-vs-executed divergence described above.
    assert set(diagnostics.requested_action_counts) != set(
        diagnostics.executed_action_counts
    )


def test_diagnostic_episodes_terminate_exactly_at_the_configured_horizon(tmp_path):
    """Terminal transitions must land exactly on the horizon, never short or long."""
    pytest.importorskip("torch")
    environment_config = load_environment_config(
        DIAGNOSTICS_DIR / "profitable_opening_environment.json"
    )
    checkpoint_path = _train_smoke_checkpoint("profitable_opening", tmp_path)

    diagnostics = run_greedy_rollout(
        checkpoint_path, environment_config, episodes=5, seed=7
    )

    for episode in diagnostics.episodes:
        assert len(episode.rewards) == environment_config.horizon
        assert len(episode.executed_actions) == environment_config.horizon
