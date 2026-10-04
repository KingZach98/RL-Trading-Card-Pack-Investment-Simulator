from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from packfolio.config import MarketConfig, REFERENCE_MARKET_CONFIG, REGIME_ORDER
from packfolio.types import MarketRegime, MarketSnapshot


def snapshot_for(
    regime: MarketRegime,
    config: MarketConfig = REFERENCE_MARKET_CONFIG,
) -> MarketSnapshot:
    """Return the public market state for a regime without consuming randomness."""
    if not isinstance(regime, MarketRegime):
        raise TypeError("regime must be MarketRegime")
    quote = config.quotes[regime]
    return MarketSnapshot(
        regime=regime,
        pack_ask=quote.pack_ask,
        card_value_multiplier=quote.card_value_multiplier,
    )


def advance_market(
    current_regime: MarketRegime,
    rng: np.random.Generator,
    config: MarketConfig = REFERENCE_MARKET_CONFIG,
) -> MarketSnapshot:
    """Sample one regime transition and return the next public snapshot."""
    if not isinstance(current_regime, MarketRegime):
        raise TypeError("current_regime must be MarketRegime")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    next_index = rng.choice(
        len(REGIME_ORDER),
        p=config.transition_matrix[current_regime],
    )
    next_regime = REGIME_ORDER[int(next_index)]
    return snapshot_for(next_regime, config)


def build_market_path(
    initial_regime: MarketRegime,
    steps: int,
    rng: np.random.Generator,
    config: MarketConfig = REFERENCE_MARKET_CONFIG,
) -> tuple[MarketSnapshot, ...]:
    """Build a deterministic path when called with the same RNG state."""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps must not be negative")

    snapshots: list[MarketSnapshot] = [snapshot_for(initial_regime, config)]
    current_regime = initial_regime
    for _ in range(steps):
        next_snapshot = advance_market(current_regime, rng, config)
        snapshots.append(next_snapshot)
        current_regime = next_snapshot.regime
    return tuple(snapshots)


def regimes_from_path(path: Sequence[MarketSnapshot]) -> tuple[MarketRegime, ...]:
    return tuple(snapshot.regime for snapshot in path)


__all__ = [
    "advance_market",
    "build_market_path",
    "regimes_from_path",
    "snapshot_for",
]
