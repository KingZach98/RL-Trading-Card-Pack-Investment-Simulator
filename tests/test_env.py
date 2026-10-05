from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
from gymnasium.utils.env_checker import check_reset_options, check_reset_seed_determinism
import numpy as np
import pytest

from packfolio.config import load_environment_config
from packfolio.env import PackfolioEnv
from packfolio.scenarios import Scenario, ScenarioSpec
from packfolio.types import Action, MarketRegime, ObservationIndex, PortfolioSnapshot


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "environment.json"
    return load_environment_config(path)


@pytest.fixture
def env(config):
    return PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)


def test_reset_returns_the_starting_observation(env, config):
    observation, info = env.reset(seed=1001)
    assert isinstance(env, gym.Env)
    assert observation.shape == (8,)
    assert observation.dtype == np.float32
    np.testing.assert_array_equal(observation, [1, 0, 1, 1, 1, 0, 1, 0])
    assert env.observation_space.contains(observation)
    assert info == {}
    assert env._portfolio == PortfolioSnapshot(cash=config.initial_cash, sealed_count=0)
    assert env._scenario.timestep == 0


def test_spaces_match_the_contract(env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 4
    assert all(env.action_space.contains(int(action)) for action in Action)
    assert not env.action_space.contains(4)
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (8,)
    assert env.observation_space.dtype == np.float32
    np.testing.assert_array_equal(env.observation_space.low, np.zeros(8))
    np.testing.assert_array_equal(env.observation_space.high, [np.inf, 1, np.inf, np.inf, 1, 1, 1, 1])


@pytest.mark.parametrize("regime, expected", [(MarketRegime.LOW, [1, 0, 0.8, 0.75, 1, 1, 0, 0]),(MarketRegime.NORMAL, [1, 0, 1, 1, 1, 0, 1, 0]),
                                              (MarketRegime.HIGH, [1, 0, 1.2, 1.6, 1, 0, 0, 1]),])
def test_reset_uses_the_configured_starting_regime(config, regime, expected):
    config = replace(config, market=replace(config.market, initial_regime=regime))
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    observation, _ = env.reset(seed=1001)
    np.testing.assert_array_equal(observation, np.array(expected, dtype=np.float32))
    assert env.observation_space.contains(observation)


def test_observation_uses_the_supplied_settings_without_clipping(config):
    config = replace(config, initial_cash=200.0, horizon=4)
    env = PackfolioEnv(config, inventory_capacity=4, reference_price=20.0)
    env.reset(seed=1001)
    env._portfolio = PortfolioSnapshot(cash=300.0, sealed_count=2)
    env._scenario.advance()
    market = env._scenario.current_market
    expected = [1.5, 0.5, market.pack_ask / 20, market.card_value_multiplier, 0.75,float(market.regime is MarketRegime.LOW),
                float(market.regime is MarketRegime.NORMAL),
                float(market.regime is MarketRegime.HIGH)]
    observation = env._get_observation()
    np.testing.assert_array_equal(observation, np.array(expected, dtype=np.float32))
    assert env.observation_space.contains(observation)


def test_observation_has_no_steps_remaining_at_the_horizon(config):
    config = replace(config, horizon=2)
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    for _ in range(config.horizon):
        env._scenario.advance()
    observation = env._get_observation()
    assert observation[ObservationIndex.REMAINING_STEPS_RATIO] == 0
    assert env.observation_space.contains(observation)


def test_reset_restores_cash_inventory_time_and_market(env, config):
    env.reset(seed=1001)
    previous_scenario = env._scenario
    env._portfolio = PortfolioSnapshot(cash=20.0, sealed_count=3)
    env._scenario.open_pack()
    env._scenario.advance()
    observation, info = env.reset(seed=1001)
    assert env._scenario is not previous_scenario
    assert env._scenario.timestep == 0
    assert env._scenario.current_market.regime is config.market.initial_regime
    assert env._portfolio == PortfolioSnapshot(cash=config.initial_cash, sealed_count=0)
    np.testing.assert_array_equal(observation, [1, 0, 1, 1, 1, 0, 1, 0])
    assert info == {}


def test_observation_arrays_do_not_share_state(env):
    observation, _ = env.reset(seed=1001)
    observation[:] = -999
    current = env._get_observation()
    np.testing.assert_array_equal(current, [1, 0, 1, 1, 1, 0, 1, 0])
    assert not np.shares_memory(observation, current)


def test_observation_requires_a_reset(env):
    with pytest.raises(RuntimeError, match="call reset"):
        env._get_observation()


@pytest.mark.parametrize("seed", [0, 1001, 1002, 2**64 - 1])
def test_explicit_seed_replays_the_existing_scenario(env, config, seed):
    expected = Scenario(config, ScenarioSpec(config.config_hash, seed))
    for _ in range(2):
        env.reset(seed=seed)
        for _ in range(8):
            assert env._scenario.current_market == expected.current_market
            assert env._scenario.open_pack() == expected.open_pack()
            assert env._scenario.advance() == expected.advance()
        expected = Scenario(config, ScenarioSpec(config.config_hash, seed))


def test_seedless_resets_keep_the_rng_and_match_across_environments(env, config):
    replay = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    replay.reset(seed=1001)
    rng = env.np_random
    expected_rng = np.random.default_rng(1001)
    seeds = []
    for _ in range(3):
        seed = int(expected_rng.integers(2**64, dtype=np.uint64))
        seeds.append(seed)
        expected = Scenario(config, ScenarioSpec(config.config_hash, seed))
        env.reset()
        replay.reset()
        assert env.np_random is rng
        assert env.np_random_seed == 1001
        for _ in range(8):
            assert env._scenario.current_market == replay._scenario.current_market == expected.current_market
            assert env._scenario.open_pack() == replay._scenario.open_pack() == expected.open_pack()
            assert env._scenario.advance() == replay._scenario.advance() == expected.advance()
    assert len(set(seeds)) == 3


def test_first_reset_can_choose_a_seed(env):
    observation, info = env.reset()
    assert env.observation_space.contains(observation)
    assert env._scenario.timestep == 0
    assert info == {}


@pytest.mark.parametrize("seed", [-1, True, False, 1.5, "1001", np.int64(1001), 2**64])
def test_invalid_seeds_do_not_change_episode_state(env, seed):
    env.reset(seed=1001)
    scenario = env._scenario
    portfolio = env._portfolio
    rng = env.np_random
    rng_state = deepcopy(rng.bit_generator.state)
    with pytest.raises((TypeError, ValueError), match="seed"):
        env.reset(seed=seed)
    assert env._scenario is scenario
    assert env._portfolio is portfolio
    assert env.np_random is rng
    assert env.np_random.bit_generator.state == rng_state


def test_empty_reset_options_work_but_other_options_fail(env):
    observation, _ = env.reset(seed=1001, options={})
    assert env.observation_space.contains(observation)
    with pytest.raises(ValueError, match="options are not supported"):
        env.reset(options={"cash": 1})
    with pytest.raises(TypeError, match="options must be a dict"):
        env.reset(options=[])


@pytest.mark.parametrize("capacity, error", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError), ("10", TypeError),])
def test_invalid_capacity_is_rejected(config, capacity, error):
    with pytest.raises(error, match="inventory_capacity"):
        PackfolioEnv(config, inventory_capacity=capacity, reference_price=10.0)


@pytest.mark.parametrize("reference_price, error", [(0, ValueError), (-1, ValueError), (float("inf"), ValueError), (float("nan"), ValueError),
                                                    (True, TypeError), ("10", TypeError),])
def test_invalid_reference_price_is_rejected(config, reference_price, error):
    with pytest.raises(error, match="reference_price"):
        PackfolioEnv(config, inventory_capacity=10, reference_price=reference_price)


def test_config_must_be_the_validated_type():
    with pytest.raises(TypeError, match="config must be EnvironmentConfig"):
        PackfolioEnv({}, inventory_capacity=10, reference_price=10.0)


def test_fee_must_match_the_portfolio_rules(config):
    with pytest.raises(ValueError, match="selling_fee_rate must be less than 1"):
        PackfolioEnv(replace(config, selling_fee_rate=1.0), inventory_capacity=10, reference_price=10.0)


def test_unsupported_pack_outcome_is_rejected(config):
    outcomes = (replace(config.pack.outcomes[0], outcome_id="unknown_bundle"), *config.pack.outcomes[1:])
    config = replace(config, pack=replace(config.pack, outcomes=outcomes))
    with pytest.raises(ValueError, match="unknown_bundle"):
        PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)


def test_a_supported_single_outcome_pack_is_allowed(config):
    outcome = replace(config.pack.outcomes[0], probability=1.0)
    config = replace(config, pack=replace(config.pack, outcomes=(outcome,)))
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=10.0)
    env.reset(seed=1001)
    assert env._scenario.open_pack().outcome_id == outcome.outcome_id


def test_float32_overflow_fails_instead_of_returning_infinity(config):
    env = PackfolioEnv(config, inventory_capacity=10, reference_price=1e-308)
    with pytest.raises(ValueError, match="finite as float32"):
        env.reset(seed=1001)


def test_gymnasium_reset_checks_pass(env):
    check_reset_seed_determinism(env)
    check_reset_options(env)
