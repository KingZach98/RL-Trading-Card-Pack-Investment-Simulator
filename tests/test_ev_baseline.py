from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from packfolio.baselines import BuyAndHoldPolicy
from packfolio.config import MarketQuote, REFERENCE_MARKET_CONFIG, load_environment_config
from packfolio.ev_baseline import EVOneStepPolicy, TIE_ORDER
from packfolio.types import Action, MarketRegime, ObservationIndex


@pytest.fixture
def config():
    return load_environment_config(Path(__file__).resolve().parents[1] / "configs/environment.json")


def observation(config, *, cash=100.0, count=0, regime=MarketRegime.NORMAL,
                capacity=10, reference_price=None, remaining=1.0):
    reference_price = reference_price or config.market.quotes[MarketRegime.NORMAL].pack_ask
    quote = config.market.quotes[regime]
    return np.array([
        cash / config.initial_cash, count / capacity, quote.pack_ask / reference_price,
        quote.card_value_multiplier, remaining,
        *(float(regime is item) for item in MarketRegime),
    ], dtype=np.float32)


def fixed_model(config, *, ask=10.0, value=15.0, fee=0.05):
    outcome = replace(config.pack.outcomes[0], probability=1.0, base_bundle_value=value)
    return replace(
        config, selling_fee_rate=fee,
        market=replace(config.market,
                       quotes={regime: MarketQuote(ask, 1.0) for regime in MarketRegime},
                       transition_matrix={regime: (0.0, 1.0, 0.0) for regime in MarketRegime}),
        pack=replace(config.pack, outcomes=(outcome,)),
    )


@pytest.mark.parametrize("regime, expected, action", [
    (MarketRegime.LOW, (6653.0, 6586.5, 6389.375, 6679.5), Action.BUY_PACK),
    (MarketRegime.NORMAL, (6900.0, 6900.0, 6700.5, 6850.0), Action.HOLD),
    (MarketRegime.HIGH, (7147.0, 7213.5, 7274.3, 7020.5), Action.OPEN_AND_SELL),
])
def test_bible_reference_scores_match_independent_hand_calculations(config, regime, expected, action):
    # Bible fixture: E[gross]=790, net openings=562.875/750.5/1200.8;
    # next net sealed quotes=826.5/950/1073.5, with C=5000 and N=2.
    outcomes = tuple(replace(item, probability=p, base_bundle_value=v) for item, p, v in zip(
        (config.pack.outcomes[0], config.pack.outcomes[1], config.pack.outcomes[3]),
        (0.70, 0.25, 0.05), (200.0, 1000.0, 8000.0), strict=True,
    ))
    config = replace(config, initial_cash=10000.0, market=REFERENCE_MARKET_CONFIG,
                     pack=replace(config.pack, outcomes=outcomes))
    policy = EVOneStepPolicy(config)
    obs = observation(config, cash=5000.0, count=2, regime=regime)
    scores = policy.score_actions(obs)
    assert tuple(scores) == TIE_ORDER
    assert tuple(scores.values()) == pytest.approx(expected)
    assert policy.choose_action(obs) is action


@pytest.mark.parametrize("regime, mu, opening", [
    (MarketRegime.LOW, 8.265, 11.221875),
    (MarketRegime.NORMAL, 9.5, 14.9625),
    (MarketRegime.HIGH, 10.735, 23.94),
])
def test_shipped_model_uses_configured_four_outcome_ev(config, regime, mu, opening):
    policy = EVOneStepPolicy(config)
    scores = policy.score_actions(observation(config, count=1, regime=regime))
    assert scores[Action.HOLD] == pytest.approx(100.0 + mu)
    assert scores[Action.OPEN_AND_SELL] == pytest.approx(100.0 + opening)


@pytest.mark.parametrize("fee", [0.0, 0.05, 0.2])
def test_one_fee_and_current_opening_multiplier(config, fee):
    config = fixed_model(config, fee=fee)
    scores = EVOneStepPolicy(config).score_actions(observation(config, count=2))
    assert scores == pytest.approx({
        Action.HOLD: 100.0 + 20.0 * (1 - fee),
        Action.BUY_PACK: 90.0 + 30.0 * (1 - fee),
        Action.SELL_PACK: 100.0 + 20.0 * (1 - fee),
        Action.OPEN_AND_SELL: 100.0 + 25.0 * (1 - fee),
    })


def test_all_four_equal_scores_choose_hold(config):
    config = fixed_model(config, value=10.0, fee=0.0)
    policy = EVOneStepPolicy(config)
    scores = policy.score_actions(observation(config, count=1))
    assert len(set(scores.values())) == 1
    assert policy.choose_action(observation(config, count=1)) is Action.HOLD


def test_sell_precedes_open_on_a_winning_tie(config):
    config = fixed_model(config, value=12.0, fee=0.0)
    quotes = dict(config.market.quotes)
    quotes[MarketRegime.HIGH] = MarketQuote(12.0, 1.0)
    config = replace(config, market=replace(config.market, quotes=quotes))
    policy = EVOneStepPolicy(config)
    obs = observation(config, count=1, regime=MarketRegime.HIGH)
    assert policy.score_actions(obs)[Action.SELL_PACK] == policy.score_actions(obs)[Action.OPEN_AND_SELL]
    assert policy.choose_action(obs) is Action.SELL_PACK


def test_open_precedes_buy_on_a_winning_tie(config):
    config = fixed_model(config, value=12.0, fee=0.0)
    quotes = dict(config.market.quotes)
    quotes[MarketRegime.LOW] = MarketQuote(8.0, 1.0)
    config = replace(config, market=replace(config.market, quotes=quotes))
    policy = EVOneStepPolicy(config)
    obs = observation(config, count=1, regime=MarketRegime.LOW)
    scores = policy.score_actions(obs)
    assert scores[Action.OPEN_AND_SELL] == scores[Action.BUY_PACK]
    assert policy.choose_action(obs) is Action.OPEN_AND_SELL


@pytest.mark.parametrize("count, cash, expected", [
    (0, 100.0, {Action.HOLD, Action.BUY_PACK}),
    (0, 1.0, {Action.HOLD}),
    (10, 100.0, {Action.HOLD, Action.SELL_PACK, Action.OPEN_AND_SELL}),
    (1, 1.0, {Action.HOLD, Action.SELL_PACK, Action.OPEN_AND_SELL}),
])
def test_only_observationally_feasible_actions_are_scored(config, count, cash, expected):
    assert set(EVOneStepPolicy(config).score_actions(observation(config, cash=cash, count=count))) == expected


@pytest.mark.parametrize("capacity", [1, 3, 10, 25])
@pytest.mark.parametrize("initial_cash, reference_price", [(100.0, 20.0), (200.0, 10.0)])
def test_configured_observation_scales_and_discrete_inventory(config, capacity, initial_cash, reference_price):
    config = replace(config, initial_cash=initial_cash)
    policy = EVOneStepPolicy(config, inventory_capacity=capacity, reference_price=reference_price)
    obs = observation(config, cash=initial_cash, count=capacity, capacity=capacity,
                      reference_price=reference_price)
    scores = policy.score_actions(obs)
    assert Action.BUY_PACK not in scores
    assert scores[Action.HOLD] == pytest.approx(initial_cash + capacity * 9.5)


@pytest.mark.parametrize("cash", [8.0, float(np.nextafter(8.0, 0.0)), 7.9999])
def test_affordability_exactly_reuses_pf12_convention(config, cash):
    policy = EVOneStepPolicy(config, reference_price=20.0)
    obs = observation(config, cash=cash, regime=MarketRegime.LOW, reference_price=20.0)
    baseline = BuyAndHoldPolicy(initial_cash=config.initial_cash, reference_price=20.0)
    assert (Action.BUY_PACK in policy.score_actions(obs)) == (baseline.choose_action(obs) is Action.BUY_PACK)
    assert policy.choose_action(obs) is baseline.choose_action(obs)


def test_float32_adjacent_cash_values_follow_pf12(config):
    policy = EVOneStepPolicy(config, reference_price=20.0)
    obs = observation(config, cash=8.0, regime=MarketRegime.LOW, reference_price=20.0)
    assert policy.choose_action(obs) is Action.BUY_PACK
    obs[ObservationIndex.CASH_RATIO] = np.nextafter(obs[0], np.float32(0.0))
    assert policy.choose_action(obs) is Action.HOLD


def test_ranking_does_not_lose_small_advantages_in_large_cash_balance(config):
    config = replace(fixed_model(config), initial_cash=1e20)
    policy = EVOneStepPolicy(config)
    obs = observation(config, cash=1e20, count=1)
    assert len(set(policy.score_actions(obs).values())) == 1
    assert policy.choose_action(obs) is Action.OPEN_AND_SELL


def test_final_decision_uses_the_same_equations(config):
    policy = EVOneStepPolicy(config)
    obs = observation(config, count=1)
    expected = policy.score_actions(obs)
    obs[ObservationIndex.REMAINING_STEPS_RATIO] = 1 / config.horizon
    assert policy.score_actions(obs) == expected


def test_scoring_and_construction_never_sample_or_mutate(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("EV scoring attempted to sample a realized outcome")

    monkeypatch.setattr("packfolio.packs.sample_pack", forbidden)
    monkeypatch.setattr("packfolio.market.advance_market", forbidden)
    monkeypatch.setattr("packfolio.scenarios._generator", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    rng_before = np.random.get_state()
    config_before = config.to_dict()
    policy = EVOneStepPolicy(config)
    obs = observation(config, count=1)
    before = obs.copy()
    for _ in range(4):
        policy.score_actions(obs)
        assert policy.choose_action(obs) is Action.OPEN_AND_SELL
    np.testing.assert_array_equal(obs, before)
    assert config.to_dict() == config_before
    rng_after = np.random.get_state()
    assert rng_before[0] == rng_after[0]
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    with pytest.raises(FrozenInstanceError):
        policy.inventory_capacity = 2
    scores = policy.score_actions(obs)
    scores.clear()
    assert policy.choose_action(obs) is Action.OPEN_AND_SELL


@pytest.mark.parametrize("field, value", [
    ("inventory_capacity", 0), ("inventory_capacity", True),
    ("reference_price", 0), ("reference_price", float("nan")),
])
def test_invalid_settings_are_rejected(config, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        EVOneStepPolicy(config, **{field: value})


def test_incompatible_fee_is_rejected_without_changing_config_contract(config):
    with pytest.raises(ValueError, match="selling_fee_rate"):
        EVOneStepPolicy(replace(config, selling_fee_rate=1.0))


@pytest.mark.parametrize("index, value, message", [
    (0, float("nan"), "finite"), (0, -1.0, "nonnegative"),
    (1, 1.1, "ratios"), (1, 0.15, "sealed-pack count"),
    (4, 1.1, "ratios"), (5, 1.0, "one-hot"),
    (2, 2.0, "market features"), (3, 2.0, "market features"),
])
def test_bad_observations_fail_clearly(config, index, value, message):
    obs = observation(config)
    obs[index] = value
    with pytest.raises(ValueError, match=message):
        EVOneStepPolicy(config).choose_action(obs)


@pytest.mark.parametrize("bad", [np.zeros(7, dtype=np.float32), np.zeros(8, dtype=np.float64), [0] * 8])
def test_observation_contract_is_enforced(config, bad):
    with pytest.raises((TypeError, ValueError)):
        EVOneStepPolicy(config).choose_action(bad)
