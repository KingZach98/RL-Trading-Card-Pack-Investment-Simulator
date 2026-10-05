from dataclasses import replace
from pathlib import Path

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
    market = replace(config.market, transition_matrix={regime: (0.0, 1.0, 0.0) for regime in MarketRegime})
    return replace(config, horizon=6, market=market)


@pytest.fixture
def env(steady_config):
    env = PackfolioEnv(steady_config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    return env


def run_episode(env, actions, seed):
    env.reset(seed=seed)
    trace = []
    previous_value = env._config.initial_cash
    for index in range(env._config.horizon):
        observation, reward, terminated, truncated, info = env.step(actions[index % len(actions)])
        assert reward == info["reward"]
        assert info["portfolio_value_before"] == pytest.approx(previous_value)
        assert terminated is (index == env._config.horizon - 1)
        assert truncated is False
        previous_value = info["portfolio_value_after"]
        trace.append(info)
    initial_value = env._config.initial_cash
    assert trace[-1]["sealed_count_after"] == 0
    assert trace[-1]["cash_after"] == pytest.approx(trace[-1]["portfolio_value_after"])
    assert observation[ObservationIndex.SEALED_COUNT_RATIO] == 0
    assert observation[ObservationIndex.REMAINING_STEPS_RATIO] == 0
    assert sum(step["reward"] for step in trace) == pytest.approx((trace[-1]["cash_after"] - initial_value) / initial_value)
    return trace


def test_buy_accounts_for_the_sale_fee_and_sell_is_not_new_profit(env):
    _, buy_reward, _, _, buy = env.step(Action.BUY_PACK)
    assert buy["cash_after"] == 90.0
    assert buy["portfolio_value_before"] == 100.0
    assert buy["portfolio_value_after"] == 99.5
    assert buy_reward == buy["reward"] == pytest.approx(-0.005)

    _, sell_reward, _, _, sell = env.step(Action.SELL_PACK)
    assert sell["cash_before"] == 90.0
    assert sell["cash_after"] == 99.5
    assert sell["portfolio_value_before"] == sell["portfolio_value_after"] == 99.5
    assert sell_reward == sell["reward"] == 0.0


@pytest.mark.parametrize("action", [Action.HOLD, Action.OPEN_AND_SELL, Action.SELL_PACK])
def test_cash_only_actions_have_no_reward(config, action):
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    for _ in range(5):
        _, reward, _, _, info = env.step(action)
        assert info["portfolio_value_before"] == info["portfolio_value_after"] == 100.0
        assert reward == info["reward"] == 0.0


@pytest.mark.parametrize("start, target, expected", [(MarketRegime.LOW, MarketRegime.HIGH, 0.019),(MarketRegime.HIGH, MarketRegime.LOW, -0.019),])
@pytest.mark.parametrize("action", [Action.HOLD, Action.BUY_PACK])
def test_hold_and_infeasible_buy_receive_the_market_value_change(config, start, target, expected, action):
    normal_row = (0.0, 0.0, 1.0) if target is MarketRegime.HIGH else (1.0, 0.0, 0.0)
    market = replace(config.market, initial_regime=start, transition_matrix={MarketRegime.LOW: (0.0, 1.0, 0.0), MarketRegime.NORMAL: normal_row,
                                                                             MarketRegime.HIGH: (0.0, 1.0, 0.0),})
    env = PackfolioEnv(replace(config, market=market), inventory_capacity=1, reference_price=10.0)
    env.reset(seed=1001)
    env.step(Action.BUY_PACK)
    _, reward, _, _, info = env.step(action)
    assert info["regime_before"] == MarketRegime.NORMAL.value
    assert info["regime_after"] == target.value
    assert info["executed_action"] == int(Action.HOLD)
    assert info["action_was_infeasible"] is (action is Action.BUY_PACK)
    assert info["sealed_count_before"] == info["sealed_count_after"] == 1
    assert info["cash_before"] == info["cash_after"]
    assert info["fee_paid"] == 0.0
    assert reward == info["reward"] == pytest.approx(expected)


@pytest.mark.parametrize("initial_cash, expected", [(50.0, -0.01), (200.0, -0.0025)])
def test_reward_uses_the_configured_initial_cash(steady_config, initial_cash, expected):
    config = replace(steady_config, initial_cash=initial_cash)
    env = PackfolioEnv(config, inventory_capacity=2, reference_price=20.0)
    env.reset(seed=1001)
    _, reward, _, _, info = env.step(Action.BUY_PACK)
    assert info["portfolio_value_before"] == initial_cash
    assert info["portfolio_value_after"] == initial_cash - 0.5
    assert reward == info["reward"] == pytest.approx(expected)


@pytest.mark.parametrize("fee, buy_reward, open_reward", [(0.05, -0.005, 0.0475), (0.0, 0.0, 0.05)])
def test_open_reward_subtracts_the_owned_pack_value(steady_config, fee, buy_reward, open_reward):
    rookie = next(outcome for outcome in steady_config.pack.outcomes if outcome.outcome_id == "rookie_bundle")
    pack = replace(steady_config.pack, outcomes=(replace(rookie, probability=1.0),))
    config = replace(steady_config, pack=pack, selling_fee_rate=fee)
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    assert env.step(Action.BUY_PACK)[1] == pytest.approx(buy_reward)
    _, reward, _, _, info = env.step(Action.OPEN_AND_SELL)
    assert info["gross_opened_value"] == 15.0
    assert info["cash_after"] == pytest.approx(90 + 15 * (1 - fee))
    assert info["sealed_count_after"] == 0
    assert reward == info["reward"] == pytest.approx(open_reward)


def test_a_fixed_price_episode_has_the_expected_reward_sum(steady_config):
    rookie = next(outcome for outcome in steady_config.pack.outcomes if outcome.outcome_id == "rookie_bundle")
    pack = replace(steady_config.pack, outcomes=(replace(rookie, probability=1.0),))
    config = replace(steady_config, pack=pack)
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    actions = [Action.BUY_PACK, Action.HOLD, Action.OPEN_AND_SELL,
               Action.BUY_PACK, Action.SELL_PACK, Action.HOLD]
    trace = run_episode(env, actions, seed=1001)
    assert [step["reward"] for step in trace] == pytest.approx([-0.005, 0, 0.0475, -0.005, 0, 0])
    assert trace[-1]["portfolio_value_after"] == pytest.approx(103.75)
    assert sum(step["reward"] for step in trace) == pytest.approx(0.0375)


@pytest.mark.parametrize("seed", [1001, 1002])
@pytest.mark.parametrize("actions", [[Action.HOLD], [Action.BUY_PACK, Action.HOLD], [Action.BUY_PACK, Action.OPEN_AND_SELL, Action.BUY_PACK, Action.SELL_PACK],], ids=["cash-only", "keep-sealed", "trade-and-open"])
def test_rewards_telescope_with_the_shipped_config(config, seed, actions):
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    trace = run_episode(env, actions, seed)
    assert len(trace) == 100
    if actions == [Action.BUY_PACK, Action.HOLD]:
        assert trace[-1]["sealed_count_before"] > 0


def test_reset_replays_the_same_rewards_after_another_episode(config):
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    actions = [Action.BUY_PACK, Action.OPEN_AND_SELL, Action.BUY_PACK, Action.SELL_PACK]
    first = run_episode(env, actions, seed=1001)
    run_episode(env, actions, seed=1002)
    replay = run_episode(env, actions, seed=1001)
    assert replay == first
