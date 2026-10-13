"""Independent PF-22 metric and paired-bootstrap calculations."""

from dataclasses import replace
from statistics import mean

import numpy as np
import pytest

from packfolio.held_out import bootstrap_mean_interval, summarize_rows
from packfolio.types import EvaluationRow, TerminationReason


def row(value=100.0, *, seed=1):
    return EvaluationRow(simulator_version="0.1.0", policy_id="DQN", model_id="fixture",
        training_seed=20261013, scenario_id=f"{'a' * 64}:{seed}", scenario_seed=seed,
        config_hash="a" * 64, git_commit="fixture", episode_steps=100,
        initial_portfolio_value=100, final_portfolio_value=value, cumulative_reward=value / 100 - 1,
        requested_hold_count=80, requested_buy_pack_count=20, requested_open_and_sell_count=0,
        requested_sell_pack_count=0, executed_hold_count=85, executed_buy_pack_count=15,
        executed_open_and_sell_count=0, executed_sell_pack_count=0, infeasible_action_count=5,
        terminated=True, truncated=False, termination_reason=TerminationReason.HORIZON)


def test_hand_calculated_final_value_downside_and_action_metrics():
    metrics = summarize_rows([row(80, seed=1), row(100, seed=2), row(120, seed=3)])
    assert metrics["scenario_count"] == 3
    assert metrics["mean_final_value"] == metrics["median_final_value"] == 100
    assert metrics["sample_std_final_value"] == 20
    assert metrics["loss_probability"] == pytest.approx(1 / 3)
    assert metrics["fifth_percentile_final_value"] == 82
    assert metrics["requested_buy_pack_rate"] == 0.2
    assert metrics["executed_buy_pack_rate"] == 0.15
    assert metrics["executed_hold_rate"] == 0.85
    assert metrics["infeasible_action_rate"] == 0.05


def test_loss_tolerance_distinguishes_roundoff_from_loss():
    metrics = summarize_rows([row(100 - 1e-10, seed=1), row(100 - 1e-8, seed=2)])
    assert metrics["loss_probability"] == 0.5


def test_single_scenario_has_undefined_sample_standard_deviation():
    assert summarize_rows([row()])["sample_std_final_value"] is None


@pytest.mark.parametrize("records", [[], [row(), row()], [row(), replace(row(seed=2), model_id="other")]])
def test_summaries_reject_empty_duplicate_and_mixed_models(records):
    with pytest.raises(ValueError):
        summarize_rows(records)


def test_paired_constant_advantage_has_zero_width_interval():
    assert bootstrap_mean_interval([10, 20, 30], [8, 18, 28], seed=7, resamples=100) == (2, 2)


def test_bootstrap_matches_independent_scalar_resampling():
    values = [2.0, 4.0, 9.0]
    rng = np.random.Generator(np.random.PCG64(7))
    expected_means = [mean(values[i] for i in rng.integers(3, size=3)) for _ in range(40)]
    expected = np.quantile(expected_means, [0.1, 0.9], method="linear")
    actual = bootstrap_mean_interval(values, seed=7, resamples=40, confidence=0.8)
    assert actual == pytest.approx(expected)


def test_bootstrap_is_repeatable_and_isolated_from_global_numpy_rng():
    state = np.random.get_state()
    first = bootstrap_mean_interval([1, 4, 9, 16], seed=8)
    assert first == bootstrap_mean_interval([1, 4, 9, 16], seed=8)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


@pytest.mark.parametrize("values", [[], [1], [1, float("nan")], [[1, 2]], [1, float("inf")]])
def test_bootstrap_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        bootstrap_mean_interval(values)


@pytest.mark.parametrize("settings", [
    {"seed": -1}, {"seed": True}, {"resamples": 0}, {"resamples": True},
    {"confidence": 0}, {"confidence": 1}, {"confidence": float("nan")},
])
def test_bootstrap_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        bootstrap_mean_interval([1, 2], **settings)


def test_bootstrap_rejects_unmatched_baseline():
    with pytest.raises(ValueError, match="paired"):
        bootstrap_mean_interval([1, 2], [1, 2, 3])
