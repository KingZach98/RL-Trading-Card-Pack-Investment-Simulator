from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from packfolio.config import MarketQuote, load_environment_config
from packfolio.env import PackfolioEnv
from packfolio.ev_baseline import EVOneStepPolicy, TIE_ORDER, evaluate_ev_episode
from packfolio.evaluate import (
    EvaluationMetadata, evaluate_episode, read_evaluation_rows_jsonl, read_step_traces_jsonl,
)
from packfolio.scenarios import ScenarioSpec, SIMULATOR_VERSION
from packfolio.types import Action, MarketRegime, StepInfo, TerminationReason


@pytest.fixture
def config():
    base = load_environment_config(Path(__file__).resolve().parents[1] / "configs/environment.json")
    outcome = replace(base.pack.outcomes[1], probability=1.0)
    return replace(
        base, horizon=4,
        market=replace(
            base.market, initial_regime=MarketRegime.LOW,
            quotes={regime: MarketQuote(8.0 if regime is MarketRegime.LOW else 10.0, 1.0)
                    for regime in MarketRegime},
            transition_matrix={MarketRegime.LOW: (0.0, 1.0, 0.0),
                               MarketRegime.NORMAL: (1.0, 0.0, 0.0),
                               MarketRegime.HIGH: (1.0, 0.0, 0.0)},
        ),
        pack=replace(base.pack, outcomes=(outcome,)),
    )


def metadata(config):
    spec = ScenarioSpec(config.config_hash, 1001)
    return EvaluationMetadata(
        simulator_version=SIMULATOR_VERSION, policy_id="EV_ONE_STEP",
        model_id=None, training_seed=None, scenario_id=spec.scenario_id,
        scenario_seed=spec.seed, config_hash=config.config_hash, git_commit="fixture",
    )


def environment(config):
    return PackfolioEnv(config, inventory_capacity=2, reference_price=20.0)


@pytest.mark.parametrize("horizon, actions, final_value", [
    (1, [Action.BUY_PACK], 101.5),
    (3, [Action.BUY_PACK, Action.OPEN_AND_SELL, Action.BUY_PACK], 107.75),
    (4, [Action.BUY_PACK, Action.OPEN_AND_SELL] * 2, 112.5),
])
def test_real_traces_and_terminal_liquidation_match_hand_calculations(config, horizon, actions, final_value):
    config = replace(config, horizon=horizon)
    policy = EVOneStepPolicy(config, inventory_capacity=2, reference_price=20.0)
    env = environment(config)
    result = evaluate_episode(policy, env, metadata(config))
    assert [step.requested_action for step in result.step_traces] == actions
    assert [step.executed_action for step in result.step_traces] == actions
    assert result.row.infeasible_action_count == 0
    assert result.row.final_portfolio_value == pytest.approx(final_value)
    assert result.row.cumulative_reward == pytest.approx((final_value - 100.0) / 100.0)
    assert result.row.episode_steps == horizon
    assert result.row.terminated and not result.row.truncated
    assert result.step_traces[-1].sealed_count_after == 0
    assert result.step_traces[-1].termination_reason is TerminationReason.HORIZON
    expected_fees = sum(0.75 for action in actions if action is Action.OPEN_AND_SELL)
    if actions[-1] is Action.BUY_PACK:
        expected_fees += 0.5
    assert sum(step.fee_paid for step in result.step_traces) == pytest.approx(expected_fees)
    assert result == evaluate_episode(policy, environment(config), metadata(config))
    with pytest.raises(RuntimeError, match="ended"):
        env.step(Action.HOLD)


def test_no_lookahead_to_buy_for_a_profitable_future_opening(config):
    market = replace(config.market, initial_regime=MarketRegime.NORMAL,
                     quotes={regime: MarketQuote(10.0, 1.0) for regime in MarketRegime},
                     transition_matrix={regime: (0.0, 1.0, 0.0) for regime in MarketRegime})
    config = replace(config, market=market)
    policy = EVOneStepPolicy(config)
    result = evaluate_episode(policy, PackfolioEnv(config, inventory_capacity=10, reference_price=10.0),
                              metadata(config))
    assert all(step.requested_action is Action.HOLD for step in result.step_traces)
    assert result.row.final_portfolio_value == 100.0


def test_unaffordable_state_holds_without_borrowing(config):
    config = replace(config, initial_cash=5.0)
    policy = EVOneStepPolicy(config, inventory_capacity=2, reference_price=20.0)
    result = evaluate_episode(policy, environment(config), metadata(config))
    assert all(step.requested_action is Action.HOLD for step in result.step_traces)
    assert result.row.final_portfolio_value == 5.0
    assert result.row.infeasible_action_count == 0


def test_known_float32_boundary_keeps_strict_accounting_authoritative(config):
    config = replace(config, initial_cash=float(np.nextafter(8.0, 0.0)), horizon=1)
    env = environment(config)
    obs, _ = env.reset(seed=1001)
    policy = EVOneStepPolicy(config, inventory_capacity=2, reference_price=20.0)
    assert Action.BUY_PACK in policy.score_actions(obs)
    assert policy.choose_action(obs) is Action.BUY_PACK
    step = StepInfo.from_dict(env.step(policy.choose_action(obs))[4])
    assert step.action_was_infeasible
    assert step.executed_action is Action.HOLD
    assert step.cash_after == config.initial_cash
    assert step.sealed_count_after == 0


@pytest.mark.parametrize("action", TIE_ORDER)
def test_scores_equal_weighted_actual_final_step_values(config, action):
    # After a prefix BUY at 8: C=92, N=1, current NORMAL ask=10.
    # Next quotes 8/12 have probabilities .25/.75. Opening values 5/15
    # have probabilities .25/.75. Enumerate real ledger outcomes, not EV formulas.
    quotes = {MarketRegime.LOW: MarketQuote(8.0, 1.0),
              MarketRegime.NORMAL: MarketQuote(10.0, 1.0),
              MarketRegime.HIGH: MarketQuote(12.0, 1.0)}
    outcomes = (replace(config.pack.outcomes[0], outcome_id="base_bundle",
                        probability=0.25, base_bundle_value=5.0),
                replace(config.pack.outcomes[0], outcome_id="rookie_bundle",
                        probability=0.75, base_bundle_value=15.0))
    config = replace(config, horizon=2, pack=replace(config.pack, outcomes=outcomes),
                     market=replace(config.market, quotes=quotes,
                                    transition_matrix={MarketRegime.LOW: (0.0, 1.0, 0.0),
                                                       MarketRegime.NORMAL: (0.25, 0.0, 0.75),
                                                       MarketRegime.HIGH: (0.0, 1.0, 0.0)}))
    env = PackfolioEnv(config, inventory_capacity=3, reference_price=20.0)
    env.reset(seed=1001)
    obs = env.step(Action.BUY_PACK)[0]
    score = EVOneStepPolicy(config, inventory_capacity=3, reference_price=20.0).score_actions(obs)[action]
    actual_expected = 0.0
    for target, probability in ((MarketRegime.LOW, 0.25), (MarketRegime.HIGH, 0.75)):
        row = (1.0, 0.0, 0.0) if target is MarketRegime.LOW else (0.0, 0.0, 1.0)
        for outcome, pack_probability in zip(outcomes, (0.25, 0.75), strict=True):
            matrix = dict(config.market.transition_matrix)
            matrix[MarketRegime.NORMAL] = row
            variant = replace(config, market=replace(config.market, transition_matrix=matrix),
                              pack=replace(config.pack, outcomes=(replace(outcome, probability=1.0),)))
            trial = PackfolioEnv(variant, inventory_capacity=3, reference_price=20.0)
            trial.reset(seed=1001)
            trial.step(Action.BUY_PACK)
            _, _, terminated, truncated, info = trial.step(action)
            step = StepInfo.from_dict(info)
            assert terminated and not truncated and not step.action_was_infeasible
            assert step.sealed_count_after == 0
            actual_expected += probability * pack_probability * step.portfolio_value_after
    assert score == pytest.approx(actual_expected, abs=1e-5, rel=0)


@pytest.mark.parametrize("seed", [1001, 1234, 5678])
def test_repeated_scoring_cannot_change_environment_or_latent_streams(seed):
    config = load_environment_config(Path(__file__).resolve().parents[1] / "configs/environment.json")
    config = replace(config, horizon=8)
    env, control = environment(config), environment(config)
    obs, _ = env.reset(seed=seed)
    control.reset(seed=seed)
    policy = EVOneStepPolicy(config, inventory_capacity=2, reference_price=20.0)
    for action in (Action.BUY_PACK, Action.OPEN_AND_SELL) * 4:
        before = obs.copy()
        for _ in range(5):
            policy.score_actions(obs)
            policy.choose_action(obs)
        np.testing.assert_array_equal(obs, before)
        actual = env.step(action)
        expected = control.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]
        obs = actual[0]


def test_adapter_uses_common_runner_and_labels_outputs_without_schema_changes(config, tmp_path, monkeypatch):
    policy = EVOneStepPolicy(config, inventory_capacity=2, reference_price=20.0)
    calls = []

    def runner(*args):
        calls.append(args)
        return evaluate_episode(*args)

    monkeypatch.setattr("packfolio.evaluate.evaluate_episode", runner)
    env = environment(config)
    meta = metadata(config)
    output = tmp_path / "evaluation"
    result = evaluate_ev_episode(policy, env, meta, output_directory=output)
    assert calls == [(policy, env, meta)]
    assert read_evaluation_rows_jsonl(output / "evaluation_rows.jsonl") == (result.row,)
    assert read_step_traces_jsonl(output / "step_traces.jsonl") == result.step_traces
    classification = json.loads((output / "policy_metadata.json").read_text(encoding="utf-8"))
    assert classification["policy_id"] == result.row.policy_id == "EV_ONE_STEP"
    assert classification["information_class"] == "model-informed"
    assert classification["config_hash"] == result.row.config_hash == config.config_hash
    assert classification["scenario_id"] == result.row.scenario_id
    assert classification["inventory_capacity"] == 2
    assert classification["observation_reference_price"] == 20.0
    assert classification["tie_order"] == ["HOLD", "SELL_PACK", "OPEN_AND_SELL", "BUY_PACK"]
    assert result.row.model_id is None and result.row.training_seed is None
    replay = evaluate_ev_episode(policy, environment(config), meta, output_directory=tmp_path / "replay")
    assert replay == result
    for file in output.iterdir():
        assert file.read_bytes() == (tmp_path / "replay" / file.name).read_bytes()


@pytest.mark.parametrize("field, value, message", [
    ("policy_id", "DQN", "policy_id"), ("config_hash", "0" * 64, "config_hash"),
    ("model_id", "dqn-model", "trained model"), ("training_seed", 42, "training seed"),
])
def test_bad_evaluation_identity_fails_before_reset_or_output(config, tmp_path, field, value, message):
    env = environment(config)
    policy = EVOneStepPolicy(config, inventory_capacity=2, reference_price=20.0)
    output = tmp_path / "evaluation"
    with pytest.raises(ValueError, match=message):
        evaluate_ev_episode(policy, env, replace(metadata(config), **{field: value}), output_directory=output)
    assert not output.exists()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.HOLD)
