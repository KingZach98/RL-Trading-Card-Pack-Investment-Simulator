import numpy as np
import pytest

from packfolio.types import Action, MarketRegime, MarketSnapshot
from packfolio.config import (
    REFERENCE_MARKET_CONFIG,
    REGIME_ORDER,
    MarketConfig,
    MarketQuote,
)
from packfolio.market import (
    advance_market,
    build_market_path,
    regimes_from_path,
    snapshot_for,
)


def market_path_for_actions(
    actions: list[Action],
    rng: np.random.Generator,
) -> tuple[MarketSnapshot, ...]:
    current_regime = REFERENCE_MARKET_CONFIG.initial_regime
    snapshots = [snapshot_for(current_regime)]
    for _action in actions:
        next_snapshot = advance_market(current_regime, rng)
        snapshots.append(next_snapshot)
        current_regime = next_snapshot.regime
    return tuple(snapshots)


def test_reference_transition_rows_are_probability_distributions():
    for row in REFERENCE_MARKET_CONFIG.transition_matrix.values():
        assert len(row) == len(REGIME_ORDER)
        assert all(probability >= 0.0 for probability in row)
        assert sum(row) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "matrix, message",
    [
        (
            {
                MarketRegime.LOW: (0.70, 0.25, 0.10),
                MarketRegime.NORMAL: (0.15, 0.70, 0.15),
                MarketRegime.HIGH: (0.05, 0.25, 0.70),
            },
            "must sum to 1.0",
        ),
        (
            {
                MarketRegime.LOW: (0.70, -0.25, 0.55),
                MarketRegime.NORMAL: (0.15, 0.70, 0.15),
                MarketRegime.HIGH: (0.05, 0.25, 0.70),
            },
            "must not be negative",
        ),
        (
            {
                MarketRegime.LOW: (0.70, 0.30),
                MarketRegime.NORMAL: (0.15, 0.70, 0.15),
                MarketRegime.HIGH: (0.05, 0.25, 0.70),
            },
            "must have 3 probabilities",
        ),
    ],
)
def test_market_config_rejects_invalid_transition_rows(matrix, message):
    with pytest.raises(ValueError, match=message):
        MarketConfig(
            initial_regime=MarketRegime.NORMAL,
            quotes=REFERENCE_MARKET_CONFIG.quotes,
            transition_matrix=matrix,
        )


def test_market_config_rejects_nonpositive_quotes():
    with pytest.raises(ValueError, match="pack_ask must be positive"):
        MarketQuote(pack_ask=0.0, card_value_multiplier=1.0)

    with pytest.raises(ValueError, match="card_value_multiplier must be positive"):
        MarketQuote(pack_ask=1_000.0, card_value_multiplier=0.0)


def test_reference_quotes_are_positive():
    for quote in REFERENCE_MARKET_CONFIG.quotes.values():
        assert quote.pack_ask > 0.0
        assert quote.card_value_multiplier > 0.0


def test_snapshot_exposes_current_regime_and_quote():
    snapshot = snapshot_for(MarketRegime.HIGH)

    assert isinstance(snapshot, MarketSnapshot)
    assert snapshot.regime is MarketRegime.HIGH
    assert snapshot.pack_ask == 1_200.0
    assert snapshot.card_value_multiplier == 1.6


def test_snapshot_lookup_does_not_advance_or_mutate_market():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    first = snapshot_for(MarketRegime.NORMAL)
    second = snapshot_for(MarketRegime.NORMAL)
    after_lookup = advance_market(MarketRegime.NORMAL, rng_a)
    without_lookup = advance_market(MarketRegime.NORMAL, rng_b)

    assert first == second
    assert after_lookup == without_lookup


def test_advance_market_consumes_exactly_one_transition_draw():
    current_regime = MarketRegime.NORMAL
    expected_rng = np.random.default_rng(999)
    actual_rng = np.random.default_rng(999)

    expected_index = expected_rng.choice(
        len(REGIME_ORDER),
        p=REFERENCE_MARKET_CONFIG.transition_matrix[current_regime],
    )
    snapshot = advance_market(current_regime, actual_rng)

    assert snapshot.regime is REGIME_ORDER[int(expected_index)]
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state


def test_market_path_is_reproducible_for_same_seed():
    path_a = build_market_path(
        REFERENCE_MARKET_CONFIG.initial_regime,
        steps=12,
        rng=np.random.default_rng(42),
    )
    path_b = build_market_path(
        REFERENCE_MARKET_CONFIG.initial_regime,
        steps=12,
        rng=np.random.default_rng(42),
    )

    assert path_a == path_b
    assert len(path_a) == 13


def test_different_action_sequences_cannot_affect_market_path():
    actions_a = [
        Action.HOLD,
        Action.BUY_PACK,
        Action.OPEN_AND_SELL,
        Action.SELL_PACK,
        Action.HOLD,
        Action.BUY_PACK,
    ]
    actions_b = [
        Action.SELL_PACK,
        Action.SELL_PACK,
        Action.HOLD,
        Action.OPEN_AND_SELL,
        Action.BUY_PACK,
        Action.HOLD,
    ]

    path_a = market_path_for_actions(actions_a, np.random.default_rng(7))
    path_b = market_path_for_actions(actions_b, np.random.default_rng(7))

    assert len(actions_a) == len(actions_b)
    assert regimes_from_path(path_a) == regimes_from_path(path_b)
