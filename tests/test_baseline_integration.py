from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from packfolio.baselines import AlwaysOpenPolicy, BuyAndHoldPolicy, CashOnlyPolicy
from packfolio.config import MarketQuote, load_environment_config
from packfolio.env import PackfolioEnv
from packfolio.evaluate import EvaluationMetadata, evaluate_episode
from packfolio.scenarios import ScenarioSpec
from packfolio.types import Action, MarketRegime, StepInfo, TerminationReason


@pytest.fixture
def config():
    base = load_environment_config(Path(__file__).resolve().parents[1] / "configs/environment.json")
    rookie = next(outcome for outcome in base.pack.outcomes if outcome.outcome_id == "rookie_bundle")
    market = replace(
        base.market,
        quotes={regime: MarketQuote(10.0, 1.0) for regime in MarketRegime},
        transition_matrix={regime: (0.0, 1.0, 0.0) for regime in MarketRegime},
    )
    return replace(base, initial_cash=30.0, horizon=4, market=market,
                   pack=replace(base.pack, outcomes=(replace(rookie, probability=1.0),)))


def run_policy(config, policy_type, *, capacity=2):
    env = PackfolioEnv(config, inventory_capacity=capacity, reference_price=20.0)
    policy = policy_type() if policy_type is CashOnlyPolicy else policy_type(
        initial_cash=config.initial_cash, reference_price=20.0
    )
    spec = ScenarioSpec(config.config_hash, 1001)
    metadata = EvaluationMetadata(
        simulator_version="0.1.0", policy_id=policy.policy_id, model_id=None, training_seed=None,
        scenario_id=spec.scenario_id, scenario_seed=spec.seed, config_hash=config.config_hash,
        git_commit="fixture",
    )
    return env, evaluate_episode(policy, env, metadata)


@pytest.mark.parametrize("fee", [0.0, 0.05])
@pytest.mark.parametrize("policy_type, actions", [
    (CashOnlyPolicy, [Action.HOLD] * 4),
    (BuyAndHoldPolicy, [Action.BUY_PACK, Action.BUY_PACK, Action.HOLD, Action.HOLD]),
    (AlwaysOpenPolicy, [Action.BUY_PACK, Action.OPEN_AND_SELL] * 2),
])
def test_baseline_traces_and_terminal_accounting(config, policy_type, actions, fee):
    config = replace(config, selling_fee_rate=fee)
    env, result = run_policy(config, policy_type)
    expected = {
        CashOnlyPolicy: 30.0,
        BuyAndHoldPolicy: 10.0 + 2 * 10.0 * (1 - fee),
        AlwaysOpenPolicy: 30.0 - 20.0 + 2 * 15.0 * (1 - fee),
    }[policy_type]
    assert [step.requested_action for step in result.step_traces] == actions
    assert [step.executed_action for step in result.step_traces] == actions
    assert result.row.infeasible_action_count == 0
    assert result.row.final_portfolio_value == pytest.approx(expected)
    assert result.row.cumulative_reward == pytest.approx((expected - 30.0) / 30.0)
    assert sum(step.fee_paid for step in result.step_traces) == pytest.approx(
        {CashOnlyPolicy: 0.0, BuyAndHoldPolicy: 20.0 * fee, AlwaysOpenPolicy: 30.0 * fee}[policy_type]
    )
    assert [step.step_index for step in result.step_traces] == list(range(4))
    assert result.step_traces[-1].sealed_count_after == 0
    assert result.step_traces[-1].termination_reason is TerminationReason.HORIZON
    assert result.row.terminated and not result.row.truncated
    assert result == run_policy(config, policy_type)[1]
    with pytest.raises(RuntimeError, match="ended"):
        env.step(Action.HOLD)


def test_opening_baseline_leaves_terminal_liquidation_to_environment(config):
    config = replace(config, horizon=3)
    _, result = run_policy(config, AlwaysOpenPolicy)
    assert [step.requested_action for step in result.step_traces] == [
        Action.BUY_PACK, Action.OPEN_AND_SELL, Action.BUY_PACK
    ]
    assert result.row.final_portfolio_value == pytest.approx(33.75)
    assert result.step_traces[-1].fee_paid == pytest.approx(0.5)
    assert result.step_traces[-1].sealed_count_after == 0


@pytest.mark.parametrize("policy_type", [BuyAndHoldPolicy, AlwaysOpenPolicy])
def test_cash_scale_mismatch_does_not_request_unaffordable_buy(config, policy_type):
    _, result = run_policy(replace(config, initial_cash=5.0), policy_type)
    assert all(step.requested_action is Action.HOLD for step in result.step_traces)
    assert result.row.final_portfolio_value == 5.0
    assert result.row.infeasible_action_count == 0


def test_exact_cash_after_a_price_change_is_not_lost_to_float32_decoding(config):
    market = replace(
        config.market,
        quotes={MarketRegime.NORMAL: MarketQuote(92.0, 1.0),
                MarketRegime.LOW: MarketQuote(8.0, 1.0), MarketRegime.HIGH: MarketQuote(12.0, 1.0)},
        transition_matrix={regime: (1.0, 0.0, 0.0) for regime in MarketRegime},
    )
    _, result = run_policy(replace(config, initial_cash=100.0, horizon=2, market=market), BuyAndHoldPolicy)
    assert [step.executed_action for step in result.step_traces] == [Action.BUY_PACK] * 2
    assert result.step_traces[1].cash_before == 8.0
    assert result.row.final_portfolio_value == pytest.approx(15.2)
    assert result.row.infeasible_action_count == 0


def test_indistinguishable_boundary_is_still_enforced_by_strict_ledger(config):
    config = replace(config, initial_cash=float(np.nextafter(10.0, 0.0)), horizon=1)
    env = PackfolioEnv(config, inventory_capacity=2, reference_price=20.0)
    observation, _ = env.reset(seed=1001)
    policy = BuyAndHoldPolicy(initial_cash=config.initial_cash, reference_price=20.0)
    action = policy.choose_action(observation)
    assert action is Action.BUY_PACK
    step = StepInfo.from_dict(env.step(action)[4])
    assert step.action_was_infeasible
    assert step.executed_action is Action.HOLD
    assert step.cash_after == config.initial_cash
    assert step.sealed_count_after == 0


def test_terminal_inventory_is_sold_at_the_next_market_quote(config):
    market = replace(config.market, quotes={regime: MarketQuote(12.0 if regime is MarketRegime.HIGH else 10.0, 1.0)
                                         for regime in MarketRegime},
                     transition_matrix={regime: (0.0, 0.0, 1.0) for regime in MarketRegime})
    _, result = run_policy(replace(config, horizon=1, market=market), BuyAndHoldPolicy)
    assert result.row.final_portfolio_value == pytest.approx(31.4)
    assert result.step_traces[0].fee_paid == pytest.approx(0.6)
    assert result.step_traces[0].regime_after is MarketRegime.HIGH
