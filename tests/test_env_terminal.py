from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import packfolio.env as env_module
from packfolio.config import load_environment_config
from packfolio.env import PackfolioEnv
from packfolio.scenarios import Scenario
from packfolio.types import Action, MarketRegime, PortfolioSnapshot, StepInfo, TerminationReason


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "environment.json"
    config = load_environment_config(path)
    market = replace(config.market, initial_regime=MarketRegime.HIGH, transition_matrix={MarketRegime.LOW: (1.0, 0.0, 0.0),MarketRegime.NORMAL: (1.0, 0.0, 0.0),
                                                                                         MarketRegime.HIGH: (0.0, 1.0, 0.0),})
    rookie = next(outcome for outcome in config.pack.outcomes if outcome.outcome_id == "rookie_bundle")
    pack = replace(config.pack, outcomes=(replace(rookie, probability=1.0),))
    return replace(config, horizon=2, market=market, pack=pack)


@pytest.fixture
def calls(monkeypatch):
    calls = []
    original_open = Scenario.open_pack
    original_action = env_module.apply_action
    original_advance = Scenario.advance
    original_liquidate = env_module.liquidate

    def open_pack(scenario):
        calls.append("open")
        return original_open(scenario)

    def apply_action(portfolio, action, market, **kwargs):
        calls.append(action)
        return original_action(portfolio, action, market, **kwargs)

    def advance(scenario):
        calls.append("advance")
        return original_advance(scenario)

    def liquidate(portfolio, market, **kwargs):
        calls.append("liquidate")
        assert market.regime is MarketRegime.LOW
        before = env_module.liquidation_value(portfolio, market, **kwargs)
        update = original_liquidate(portfolio, market, **kwargs)
        after = env_module.liquidation_value(update.portfolio, market, **kwargs)
        assert update.portfolio.sealed_count == 0
        assert update.portfolio.cash == pytest.approx(before)
        assert after == pytest.approx(before)
        return update

    monkeypatch.setattr(Scenario, "open_pack", open_pack)
    monkeypatch.setattr(env_module, "apply_action", apply_action)
    monkeypatch.setattr(Scenario, "advance", advance)
    monkeypatch.setattr(env_module, "liquidate", liquidate)
    return calls


def assert_terminal_state(env, result, cash, fee):
    observation, reward, terminated, truncated, info = result
    step = StepInfo.from_dict(info)
    assert terminated is True
    assert truncated is False
    assert env._scenario.timestep == env._config.horizon
    assert env._portfolio.cash == step.cash_after == pytest.approx(cash)
    assert env._portfolio.sealed_count == step.sealed_count_after == 0
    assert step.portfolio_value_after == pytest.approx(cash)
    assert step.fee_paid == pytest.approx(fee)
    assert step.termination_reason is TerminationReason.HORIZON
    assert reward == step.reward == pytest.approx(
        (cash - step.portfolio_value_before) / env._config.initial_cash)
    np.testing.assert_array_equal(observation, np.array(
        [cash / env._config.initial_cash, 0, 0.8, 0.75, 0, 1, 0, 0], dtype=np.float32))
    assert env.observation_space.contains(observation)
    return step


@pytest.mark.parametrize("action, cash, fee", [
    (Action.HOLD, 95.6, 0.4),
    (Action.BUY_PACK, 93.2, 0.8),
    (Action.OPEN_AND_SELL, 102.25, 0.75),
    (Action.SELL_PACK, 97.5, 0.5),
])
def test_final_actions_use_current_prices_then_liquidate_at_the_next_price(config, calls, action, cash, fee):
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    first_reward = env.step(Action.BUY_PACK)[1]
    calls.clear()
    step = assert_terminal_state(env, env.step(action), cash, fee)
    assert calls == (["open"] if action is Action.OPEN_AND_SELL else []) + [action, "advance", "liquidate"]
    assert step.step_index == 1
    assert step.requested_action is step.executed_action is action
    assert not step.action_was_infeasible
    assert step.regime_before is MarketRegime.NORMAL
    assert step.regime_after is MarketRegime.LOW
    assert step.cash_before == 88.0
    assert step.sealed_count_before == 1
    assert step.portfolio_value_before == 97.5
    if action is Action.OPEN_AND_SELL:
        assert step.pack_outcome_id.value == "rookie_bundle"
        assert step.gross_opened_value == 15.0
    else:
        assert step.pack_outcome_id is step.gross_opened_value is None
    assert first_reward + step.reward == pytest.approx((cash - config.initial_cash) / config.initial_cash)


@pytest.mark.parametrize("action, cash, fee", [
    (Action.SELL_PACK, 117.1, 0.9),
    (Action.OPEN_AND_SELL, 121.85, 1.15),
])
def test_final_sale_records_both_the_action_fee_and_remaining_inventory_fee(config, calls, action, cash, fee):
    config = replace(config, horizon=1, market=replace(config.market, initial_regime=MarketRegime.NORMAL))
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    # Isolate the final ledger with two owned packs; reset cash still defines V0.
    env._portfolio = PortfolioSnapshot(cash=100.0, sealed_count=2)
    step = assert_terminal_state(env, env.step(action), cash, fee)
    assert step.requested_action is step.executed_action is action
    assert step.sealed_count_before == 2
    assert step.portfolio_value_before == 119.0
    assert calls == (["open"] if action is Action.OPEN_AND_SELL else []) + [action, "advance", "liquidate"]
    if action is Action.OPEN_AND_SELL:
        assert step.pack_outcome_id.value == "rookie_bundle"
        assert step.gross_opened_value == 15.0
    else:
        assert step.pack_outcome_id is step.gross_opened_value is None


@pytest.mark.parametrize("count", [0, 1, 3])
def test_terminal_hold_liquidates_empty_one_and_multiple_pack_portfolios(config, calls, count):
    config = replace(config, horizon=1, market=replace(config.market, initial_regime=MarketRegime.NORMAL))
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    env._portfolio = PortfolioSnapshot(cash=100.0, sealed_count=count)
    step = assert_terminal_state(env, env.step(Action.HOLD), 100 + 7.6 * count, 0.4 * count)
    assert step.portfolio_value_before == pytest.approx(100 + 9.5 * count)
    assert step.pack_outcome_id is step.gross_opened_value is None
    assert calls == [Action.HOLD, "advance", "liquidate"]


@pytest.mark.parametrize("fee, cash, terminal_fee", [(0.05, 97.6, 0.4), (0.0, 98.0, 0.0)])
def test_a_one_step_buy_also_liquidates_the_new_pack(config, calls, fee, cash, terminal_fee):
    config = replace(config, horizon=1, selling_fee_rate=fee,
                     market=replace(config.market, initial_regime=MarketRegime.NORMAL))
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    step = assert_terminal_state(env, env.step(Action.BUY_PACK), cash, terminal_fee)
    assert step.portfolio_value_before == config.initial_cash
    assert step.sealed_count_before == 0
    assert step.executed_action is Action.BUY_PACK
    assert step.pack_outcome_id is step.gross_opened_value is None
    assert calls == [Action.BUY_PACK, "advance", "liquidate"]


@pytest.mark.parametrize("initial_cash, capacity", [(100.0, 1), (12.0, 10)])
def test_an_infeasible_final_buy_still_liquidates_owned_packs(config, calls, initial_cash, capacity):
    config = replace(config, initial_cash=initial_cash)
    env = PackfolioEnv(config, inventory_capacity=capacity, reference_price=10.0)
    env.reset(seed=1001)
    first_reward = env.step(Action.BUY_PACK)[1]
    calls.clear()
    final_cash = initial_cash - 12 + 7.6
    step = assert_terminal_state(env, env.step(Action.BUY_PACK), final_cash, 0.4)
    assert step.requested_action is Action.BUY_PACK
    assert step.executed_action is Action.HOLD
    assert step.action_was_infeasible
    assert step.pack_outcome_id is step.gross_opened_value is None
    assert calls == [Action.HOLD, "advance", "liquidate"]
    assert first_reward + step.reward == pytest.approx((final_cash - initial_cash) / initial_cash)


@pytest.mark.parametrize("action", [Action.OPEN_AND_SELL, Action.SELL_PACK])
def test_an_infeasible_final_sale_does_not_draw_a_pack_or_pay_a_fee(config, calls, action):
    config = replace(config, horizon=1, market=replace(config.market, initial_regime=MarketRegime.NORMAL))
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    step = assert_terminal_state(env, env.step(action), 100.0, 0.0)
    assert step.requested_action is action
    assert step.executed_action is Action.HOLD
    assert step.action_was_infeasible
    assert step.pack_outcome_id is step.gross_opened_value is None
    assert calls == [Action.HOLD, "advance", "liquidate"]


def test_a_finished_episode_cannot_trade_or_liquidate_twice_and_can_reset(config, calls):
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    env.step(Action.BUY_PACK)
    assert_terminal_state(env, env.step(Action.HOLD), 95.6, 0.4)
    portfolio = env._portfolio
    market = env._scenario.current_market
    completed_calls = list(calls)
    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(Action.OPEN_AND_SELL)
    assert env._portfolio is portfolio
    assert env._scenario.current_market == market
    assert env._scenario.timestep == config.horizon
    assert calls == completed_calls
    calls.clear()
    env.reset(seed=1001)
    assert env.step(Action.BUY_PACK)[2] is False
    assert env._portfolio == PortfolioSnapshot(cash=88.0, sealed_count=1)
    assert calls == [Action.BUY_PACK, "advance"]
