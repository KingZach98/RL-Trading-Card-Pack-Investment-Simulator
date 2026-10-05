from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from packfolio.config import load_environment_config
from packfolio.env import PackfolioEnv
from packfolio.types import Action, MarketRegime, ObservationIndex


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "environment.json"
    return load_environment_config(path)


@pytest.fixture
def steady_config(config):
    market = replace(config.market, initial_regime=MarketRegime.NORMAL, transition_matrix={regime: (0.0, 1.0, 0.0) for regime in MarketRegime})
    return replace(config, horizon=6, market=market)


def run_episode(env, config, actions, *, seed, capacity):
    observation, _ = env.reset(seed=seed)
    assert env.observation_space.contains(observation)
    cash, count = config.initial_cash, 0
    regime = config.market.initial_regime
    trace = []
    fee_rate = config.selling_fee_rate
    for index, action in enumerate(actions):
        quote = config.market.quotes[regime]
        value_before = cash + count * quote.pack_ask * (1 - fee_rate)
        observation, reward, terminated, truncated, info = env.step(action)
        assert info["step_index"] == index
        assert info["cash_before"] == pytest.approx(cash)
        assert info["sealed_count_before"] == count
        assert info["regime_before"] == regime.value
        assert info["requested_action"] == int(action)

        feasible = True
        if action is Action.BUY_PACK:
            feasible = cash >= quote.pack_ask and count < capacity
        elif action in (Action.OPEN_AND_SELL, Action.SELL_PACK):
            feasible = count > 0
        executed = action if feasible else Action.HOLD
        assert info["executed_action"] == int(executed)
        assert info["action_was_infeasible"] is (not feasible)

        fee = 0.0
        if executed is Action.BUY_PACK:
            cash -= quote.pack_ask
            count += 1
        elif executed in (Action.OPEN_AND_SELL, Action.SELL_PACK):
            gross = quote.pack_ask
            if executed is Action.OPEN_AND_SELL:
                outcome = next(item for item in config.pack.outcomes
                               if item.outcome_id == info["pack_outcome_id"])
                gross = outcome.base_bundle_value * quote.card_value_multiplier
                assert info["gross_opened_value"] == pytest.approx(gross)
            cash += gross * (1 - fee_rate)
            fee = gross * fee_rate
            count -= 1
        if executed is not Action.OPEN_AND_SELL:
            assert info["pack_outcome_id"] is info["gross_opened_value"] is None

        regime = MarketRegime(info["regime_after"])
        next_quote = config.market.quotes[regime]
        last_step = index == config.horizon - 1
        if last_step:
            cash += count * next_quote.pack_ask * (1 - fee_rate)
            fee += count * next_quote.pack_ask * fee_rate
            count = 0
        value_after = cash + count * next_quote.pack_ask * (1 - fee_rate)
        assert info["cash_after"] == pytest.approx(cash)
        assert info["sealed_count_after"] == count
        assert np.isfinite(info["cash_after"]) and info["cash_after"] >= 0
        assert 0 <= info["sealed_count_after"] <= capacity
        assert info["fee_paid"] == pytest.approx(fee)
        assert info["portfolio_value_before"] == pytest.approx(value_before)
        assert info["portfolio_value_after"] == pytest.approx(value_after)
        assert reward == info["reward"] == pytest.approx(
            (value_after - value_before) / config.initial_cash)
        assert terminated is last_step
        assert truncated is False
        assert info["termination_reason"] == ("HORIZON" if last_step else None)
        assert np.isfinite(observation).all() and np.isfinite(reward)
        assert env.observation_space.contains(observation)
        assert observation[ObservationIndex.CASH_RATIO] == pytest.approx(cash / config.initial_cash)
        assert observation[ObservationIndex.SEALED_COUNT_RATIO] == pytest.approx(count / capacity)
        assert observation[ObservationIndex.REMAINING_STEPS_RATIO] == pytest.approx(
            (config.horizon - index - 1) / config.horizon)
        trace.append(info)
    assert len(trace) == config.horizon
    assert trace[-1]["sealed_count_after"] == 0
    assert sum(step["reward"] for step in trace) == pytest.approx(
        (cash - config.initial_cash) / config.initial_cash)
    return trace


@pytest.mark.parametrize("seed", [0, 1, 42, 1001])
@pytest.mark.parametrize("capacity", [1, 10])
@pytest.mark.parametrize("fee", [0.0, 0.05])
def test_random_actions_keep_the_ledger_balanced_and_replay(config, seed, capacity, fee):
    config = replace(config, selling_fee_rate=fee)
    env = PackfolioEnv(config, inventory_capacity=capacity, reference_price=10.0)
    # Pick actions with a separate RNG, not the simulator's random streams.
    rng = np.random.default_rng(seed + 10_000)
    actions = [Action(int(value)) for value in rng.integers(0, len(Action), size=config.horizon)]
    first = run_episode(env, config, actions, seed=seed, capacity=capacity)
    replay = run_episode(env, config, actions, seed=seed, capacity=capacity)
    assert replay == first


@pytest.mark.parametrize("seed", [1001, 1002])
@pytest.mark.parametrize("action, initial_cash, capacity, setup", [
    (Action.OPEN_AND_SELL, 100.0, 10, []),
    (Action.SELL_PACK, 100.0, 10, []),
    (Action.BUY_PACK, 10.0, 10, [Action.BUY_PACK]),
    (Action.BUY_PACK, 100.0, 1, [Action.BUY_PACK]),
], ids=["empty-open", "empty-sell", "short-cash", "full-inventory"])
def test_unavailable_actions_match_hold_and_later_draws(config, seed, action, initial_cash, capacity, setup):
    config = replace(config, initial_cash=initial_cash)
    active = PackfolioEnv(config, inventory_capacity=capacity, reference_price=10.0)
    holding = PackfolioEnv(config, inventory_capacity=capacity, reference_price=10.0)
    active.reset(seed=seed)
    holding.reset(seed=seed)
    followup = [Action.BUY_PACK, Action.OPEN_AND_SELL, Action.BUY_PACK,
                Action.SELL_PACK, Action.BUY_PACK, Action.OPEN_AND_SELL]
    active_actions = [*setup, action, *followup]
    hold_actions = [*setup, Action.HOLD, *followup]
    opened = False
    for index, (requested, control) in enumerate(zip(active_actions, hold_actions)):
        active_obs, *active_result, active_info = active.step(requested)
        hold_obs, *hold_result, hold_info = holding.step(control)
        np.testing.assert_array_equal(active_obs, hold_obs)
        assert active_result == hold_result
        opened |= active_info["executed_action"] == int(Action.OPEN_AND_SELL)
        if index == len(setup):
            assert active_info.pop("requested_action") == int(action)
            assert active_info.pop("action_was_infeasible") is True
            assert hold_info.pop("requested_action") == int(Action.HOLD)
            assert hold_info.pop("action_was_infeasible") is False
        assert active_info == hold_info
    assert opened


@pytest.mark.parametrize("fee", [0.0, 0.05])
def test_holding_owned_packs_at_fixed_prices_keeps_the_value(steady_config, fee):
    config = replace(steady_config, selling_fee_rate=fee)
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    _, _, _, _, buy = env.step(Action.BUY_PACK)
    price = config.market.quotes[MarketRegime.NORMAL].pack_ask
    expected_value = config.initial_cash - price * fee
    assert buy["portfolio_value_after"] == pytest.approx(expected_value)
    for index in range(1, config.horizon):
        _, reward, terminated, truncated, info = env.step(Action.HOLD)
        assert info["executed_action"] == int(Action.HOLD)
        assert info["portfolio_value_before"] == pytest.approx(expected_value)
        assert info["portfolio_value_after"] == pytest.approx(expected_value)
        assert reward == 0.0
        assert terminated is (index == config.horizon - 1)
        assert truncated is False
        assert info["sealed_count_after"] == (0 if terminated else 1)
        assert info["cash_after"] == pytest.approx(
            expected_value if terminated else config.initial_cash - price)
        assert info["fee_paid"] == pytest.approx(price * fee if terminated else 0.0)


@pytest.mark.parametrize("initial_cash", [10.0, 100.0])
def test_zero_fee_buy_and_sell_at_fixed_prices_returns_the_starting_cash(steady_config, initial_cash):
    config = replace(steady_config, horizon=2, initial_cash=initial_cash, selling_fee_rate=0.0)
    env = PackfolioEnv(config, inventory_capacity=1, reference_price=10.0)
    env.reset(seed=1001)
    price = config.market.quotes[MarketRegime.NORMAL].pack_ask
    _, buy_reward, terminated, truncated, buy = env.step(Action.BUY_PACK)
    assert buy["executed_action"] == int(Action.BUY_PACK)
    assert buy["cash_after"] == pytest.approx(initial_cash - price)
    assert buy["sealed_count_after"] == 1
    assert buy_reward == buy["fee_paid"] == 0.0
    assert terminated is truncated is False
    _, sell_reward, terminated, truncated, sell = env.step(Action.SELL_PACK)
    assert sell["executed_action"] == int(Action.SELL_PACK)
    assert sell["cash_after"] == sell["portfolio_value_after"] == pytest.approx(initial_cash)
    assert sell["sealed_count_after"] == 0
    assert sell_reward == sell["fee_paid"] == 0.0
    assert terminated is True
    assert truncated is False
