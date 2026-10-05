from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import packfolio.env as env_module
from packfolio.config import load_environment_config
from packfolio.env import PackfolioEnv
from packfolio.scenarios import Scenario
from packfolio.types import Action, MarketRegime, ObservationIndex, PortfolioSnapshot, StepInfo, TerminationReason


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "environment.json"
    return load_environment_config(path)


@pytest.fixture
def action_config(config):
    market = replace(config.market, initial_regime=MarketRegime.HIGH,
                     transition_matrix={regime: (1.0, 0.0, 0.0) for regime in MarketRegime})
    rookie = next(outcome for outcome in config.pack.outcomes if outcome.outcome_id == "rookie_bundle")
    pack = replace(config.pack, outcomes=(replace(rookie, probability=1.0),))
    return replace(config, horizon=3, market=market, pack=pack)


@pytest.fixture
def env(action_config):
    env = PackfolioEnv(action_config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    env._portfolio = PortfolioSnapshot(cash=100.0, sealed_count=2)
    return env


@pytest.fixture
def calls(monkeypatch):
    calls = []
    original_open = Scenario.open_pack
    original_action = env_module.apply_action
    original_advance = Scenario.advance

    def open_pack(scenario):
        calls.append("open")
        return original_open(scenario)

    def apply_action(portfolio, action, market, **kwargs):
        calls.append(action)
        if action is Action.OPEN_AND_SELL:
            assert kwargs["pack_outcome"].base_gross_value == 15.0
            assert kwargs["pack_outcome"].gross_value == 24.0
        return original_action(portfolio, action, market, **kwargs)

    def advance(scenario):
        calls.append("advance")
        return original_advance(scenario)

    monkeypatch.setattr(Scenario, "open_pack", open_pack)
    monkeypatch.setattr(env_module, "apply_action", apply_action)
    monkeypatch.setattr(Scenario, "advance", advance)
    return calls


@pytest.mark.parametrize("action, cash, count, fee, value", [
    (Action.HOLD, 100.0, 2, 0.0, 115.2),
    (Action.BUY_PACK, 88.0, 3, 0.0, 110.8),
    (Action.OPEN_AND_SELL, 122.8, 1, 1.2, 130.4),
    (Action.SELL_PACK, 111.4, 1, 0.6, 119.0),
])
def test_actions_use_current_quotes_then_advance_once(env, calls, action, cash, count, fee, value):
    observation, reward, terminated, truncated, info = env.step(action)
    step = StepInfo.from_dict(info)
    assert calls == (["open"] if action is Action.OPEN_AND_SELL else []) + [action, "advance"]
    assert env._scenario.timestep == 1
    assert env._portfolio.cash == pytest.approx(cash)
    assert env._portfolio.sealed_count == count
    assert step.step_index == 0
    assert step.requested_action is step.executed_action is action
    assert not step.action_was_infeasible
    assert step.regime_before is MarketRegime.HIGH
    assert step.regime_after is MarketRegime.LOW
    assert step.cash_before == 100.0
    assert step.cash_after == pytest.approx(cash)
    assert step.sealed_count_before == 2
    assert step.sealed_count_after == count
    assert step.fee_paid == pytest.approx(fee)
    assert step.portfolio_value_before == pytest.approx(122.8)
    assert step.portfolio_value_after == pytest.approx(value)
    assert step.termination_reason is None
    assert info == step.to_dict()
    assert type(info["requested_action"]) is int
    assert type(info["executed_action"]) is int
    if action is Action.OPEN_AND_SELL:
        assert info["pack_outcome_id"] == "rookie_bundle"
        assert step.gross_opened_value == 24.0
    else:
        assert step.pack_outcome_id is step.gross_opened_value is None
    np.testing.assert_array_equal(observation, np.array(
        [cash / 100, count / 10, 0.8, 0.75, 2 / 3, 1, 0, 0], dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert reward == step.reward == 0.0  # PF-10 will supply the reward rule.
    assert terminated is truncated is False


@pytest.mark.parametrize("action, cash, count", [
    (Action.BUY_PACK, 11.0, 0),
    (Action.BUY_PACK, 100.0, 10),
    (Action.OPEN_AND_SELL, 100.0, 0),
    (Action.SELL_PACK, 100.0, 0),
])
def test_infeasible_actions_hold_without_drawing_a_pack(env, calls, action, cash, count):
    before = PortfolioSnapshot(cash=cash, sealed_count=count)
    env._portfolio = before
    observation, reward, terminated, truncated, info = env.step(action)
    step = StepInfo.from_dict(info)
    assert calls == [Action.HOLD, "advance"]
    assert env._portfolio == before
    assert env._scenario.timestep == 1
    assert env._scenario.current_market.regime is MarketRegime.LOW
    assert step.requested_action is action
    assert step.executed_action is Action.HOLD
    assert step.action_was_infeasible
    assert step.fee_paid == 0.0
    assert step.pack_outcome_id is step.gross_opened_value is None
    assert env.observation_space.contains(observation)
    assert reward == 0.0
    assert terminated is truncated is False


def test_buy_accepts_exact_cash(env):
    env._portfolio = PortfolioSnapshot(cash=12.0, sealed_count=0)
    _, _, _, _, info = env.step(Action.BUY_PACK)
    assert env._portfolio == PortfolioSnapshot(cash=0.0, sealed_count=1)
    assert info["executed_action"] == int(Action.BUY_PACK)
    assert not info["action_was_infeasible"]


@pytest.mark.parametrize("action, error", [
    (-1, ValueError), (4, ValueError), (np.int64(-1), ValueError),
    (np.uint64(2**64 - 1), ValueError),
    (True, TypeError), (False, TypeError), (np.bool_(True), TypeError),
    (1.0, TypeError), (np.float32(1), TypeError), (None, TypeError),
    ("1", TypeError), (np.array(1), TypeError), (np.array([1]), TypeError), ([1], TypeError),
])
def test_invalid_actions_leave_the_episode_unchanged(env, calls, action, error):
    portfolio = env._portfolio
    market = env._scenario.current_market
    rng_state = deepcopy(env.np_random.bit_generator.state)
    with pytest.raises(error, match="action"):
        env.step(action)
    assert calls == []
    assert env._portfolio is portfolio
    assert env._scenario.current_market == market
    assert env._scenario.timestep == 0
    assert env.np_random.bit_generator.state == rng_state


@pytest.mark.parametrize("action", [Action.HOLD, 0, np.int8(0), np.int64(0), np.uint64(0)])
def test_integer_scalar_actions_are_accepted(env, action):
    _, _, _, _, info = env.step(action)
    assert info["requested_action"] == int(Action.HOLD)
    assert info["executed_action"] == int(Action.HOLD)
    assert env._scenario.timestep == 1


def test_step_requires_reset(action_config, calls):
    env = PackfolioEnv(action_config, inventory_capacity=10, reference_price=10.0)
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(Action.HOLD)
    assert calls == []


def test_the_last_transition_ends_the_episode_and_blocks_more_steps(env, calls):
    for index in range(3):
        observation, _, terminated, truncated, info = env.step(Action.HOLD)
        assert info["step_index"] == index
        assert env._scenario.timestep == index + 1
        assert terminated is (index == 2)
        assert truncated is False
    assert calls == [Action.HOLD, "advance"] * 3
    assert info["termination_reason"] == TerminationReason.HORIZON.value
    assert observation[ObservationIndex.REMAINING_STEPS_RATIO] == 0
    portfolio = env._portfolio
    market = env._scenario.current_market
    with pytest.raises(RuntimeError, match="episode has ended"):
        env.step(Action.HOLD)
    assert calls == [Action.HOLD, "advance"] * 3
    assert env._portfolio is portfolio
    assert env._scenario.current_market == market
    assert env._scenario.timestep == 3
    env.reset(seed=1001)
    assert env.step(Action.HOLD)[2] is False


def test_zero_cash_and_empty_inventory_do_not_end_the_episode_early(env):
    env._portfolio = PortfolioSnapshot(cash=0.0, sealed_count=0)
    for index in range(3):
        observation, _, terminated, truncated, _ = env.step(Action.HOLD)
        assert env._portfolio == PortfolioSnapshot(cash=0.0, sealed_count=0)
        assert env.observation_space.contains(observation)
        assert terminated is (index == 2)
        assert truncated is False


def test_actions_preserve_the_existing_seeded_market_trace(config):
    active = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    holding = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    active.reset(seed=1001)
    holding.reset(seed=1001)
    actions = [Action.BUY_PACK, Action.OPEN_AND_SELL, Action.BUY_PACK, Action.SELL_PACK,
               Action.BUY_PACK, Action.HOLD, Action.OPEN_AND_SELL, Action.HOLD]
    regimes = ["NORMAL", "NORMAL", "HIGH", "HIGH", "HIGH", "HIGH", "NORMAL", "NORMAL"]
    for index, (action, regime) in enumerate(zip(actions, regimes)):
        _, _, _, _, active_info = active.step(action)
        _, _, _, _, holding_info = holding.step(Action.HOLD)
        assert active_info["step_index"] == holding_info["step_index"] == index
        assert active_info["regime_after"] == holding_info["regime_after"] == regime
        assert active._scenario.current_market == holding._scenario.current_market
        if action is Action.OPEN_AND_SELL:
            expected_pack = ("autograph_bundle", 75.0) if index == 1 else ("base_bundle", 8.0)
            assert (active_info["pack_outcome_id"], active_info["gross_opened_value"]) == expected_pack


def test_gymnasium_environment_checks_pass(config):
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    with pytest.warns(UserWarning, match="A Box observation space maximum value is infinity") as warnings:
        check_env(env, skip_render_check=True)
    assert len(warnings) == 1
