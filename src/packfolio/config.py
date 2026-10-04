from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType

from packfolio.types import MarketRegime


REGIME_ORDER = (MarketRegime.LOW, MarketRegime.NORMAL, MarketRegime.HIGH)
PROBABILITY_TOLERANCE = 1e-12


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _check_regime_mapping_keys(
    values: Mapping[MarketRegime, object],
    name: str,
) -> None:
    expected = set(REGIME_ORDER)
    actual = set(values)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        details = []
        if missing:
            details.append(
                "missing regimes: "
                + ", ".join(sorted(regime.value for regime in missing))
            )
        if extra:
            details.append(
                "extra regimes: "
                + ", ".join(sorted(str(regime) for regime in extra))
            )
        raise ValueError(f"invalid {name} ({'; '.join(details)})")


@dataclass(frozen=True, slots=True)
class MarketQuote:
    pack_ask: float
    card_value_multiplier: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pack_ask",
            _positive_float(self.pack_ask, "pack_ask"),
        )
        object.__setattr__(
            self,
            "card_value_multiplier",
            _positive_float(
                self.card_value_multiplier,
                "card_value_multiplier",
            ),
        )


@dataclass(frozen=True, slots=True)
class MarketConfig:
    initial_regime: MarketRegime
    quotes: Mapping[MarketRegime, MarketQuote]
    transition_matrix: Mapping[MarketRegime, tuple[float, float, float]]

    def __post_init__(self) -> None:
        if not isinstance(self.initial_regime, MarketRegime):
            raise TypeError("initial_regime must be MarketRegime")

        if not isinstance(self.quotes, Mapping):
            raise TypeError("quotes must be a mapping")
        quote_values = dict(self.quotes)
        _check_regime_mapping_keys(quote_values, "quotes")
        if not all(isinstance(item, MarketQuote) for item in quote_values.values()):
            raise TypeError("quotes must contain MarketQuote values")

        if not isinstance(self.transition_matrix, Mapping):
            raise TypeError("transition_matrix must be a mapping")
        matrix_values = dict(self.transition_matrix)
        _check_regime_mapping_keys(matrix_values, "transition_matrix")

        validated_rows: dict[MarketRegime, tuple[float, float, float]] = {}
        for regime, row in matrix_values.items():
            validated_rows[regime] = self._validated_transition_row(regime, row)

        object.__setattr__(self, "quotes", MappingProxyType(quote_values))
        object.__setattr__(
            self,
            "transition_matrix",
            MappingProxyType(validated_rows),
        )

    @staticmethod
    def _validated_transition_row(
        regime: MarketRegime,
        row: Sequence[object],
    ) -> tuple[float, float, float]:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"transition row for {regime.value} must be a sequence")
        if len(row) != len(REGIME_ORDER):
            raise ValueError(
                f"transition row for {regime.value} must have "
                f"{len(REGIME_ORDER)} probabilities"
            )
        probabilities = tuple(
            _probability(value, f"transition probability for {regime.value}")
            for value in row
        )
        if abs(sum(probabilities) - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError(
                f"transition row for {regime.value} must sum to 1.0"
            )
        return probabilities


REFERENCE_MARKET_CONFIG = MarketConfig(
    initial_regime=MarketRegime.NORMAL,
    quotes={
        MarketRegime.LOW: MarketQuote(pack_ask=800.0, card_value_multiplier=0.75),
        MarketRegime.NORMAL: MarketQuote(
            pack_ask=1_000.0,
            card_value_multiplier=1.0,
        ),
        MarketRegime.HIGH: MarketQuote(pack_ask=1_200.0, card_value_multiplier=1.6),
    },
    transition_matrix={
        MarketRegime.LOW: (0.70, 0.25, 0.05),
        MarketRegime.NORMAL: (0.15, 0.70, 0.15),
        MarketRegime.HIGH: (0.05, 0.25, 0.70),
    },
)


__all__ = [
    "MarketConfig",
    "MarketQuote",
    "PROBABILITY_TOLERANCE",
    "REFERENCE_MARKET_CONFIG",
    "REGIME_ORDER",
]
