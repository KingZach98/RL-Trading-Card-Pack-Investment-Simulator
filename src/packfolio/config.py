from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from types import MappingProxyType

from packfolio.packs import PackConfig, load_pack_config
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


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _json_record(data: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise TypeError(f"{name} must be a JSON object")
    if set(data) != keys:
        raise ValueError(f"{name} must have exactly these fields: {', '.join(sorted(keys))}")
    return data


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    """Validated, immutable experiment settings; no seed or random state."""

    horizon: int
    initial_cash: float
    selling_fee_rate: float
    market: MarketConfig
    pack: PackConfig

    def __post_init__(self) -> None:
        _integer(self.horizon, "horizon", minimum=1)
        object.__setattr__(self, "initial_cash", _positive_float(self.initial_cash, "initial_cash"))
        fee = _probability(self.selling_fee_rate, "selling_fee_rate")
        if fee > 1:
            raise ValueError("selling_fee_rate must not exceed 1")
        object.__setattr__(self, "selling_fee_rate", fee)
        if not isinstance(self.market, MarketConfig):
            raise TypeError("market must be MarketConfig")
        if not isinstance(self.pack, PackConfig):
            raise TypeError("pack must be PackConfig")
        for quote in self.market.quotes.values():
            for outcome in self.pack.outcomes:
                if not isfinite(outcome.base_bundle_value * quote.card_value_multiplier):
                    raise ValueError("market-scaled pack values must be finite")

    def to_dict(self) -> dict[str, object]:
        """Canonical content, including resolved pack data rather than its path."""
        return {
            "horizon": self.horizon,
            "initial_cash": self.initial_cash,
            "selling_fee_rate": self.selling_fee_rate,
            "market": {
                "initial_regime": self.market.initial_regime.value,
                "quotes": {
                    regime.value: {
                        "pack_ask": self.market.quotes[regime].pack_ask,
                        "card_value_multiplier": self.market.quotes[regime].card_value_multiplier,
                    }
                    for regime in REGIME_ORDER
                },
                "transition_matrix": {
                    regime.value: list(self.market.transition_matrix[regime])
                    for regime in REGIME_ORDER
                },
            },
            "pack": {
                "pack_id": self.pack.pack_id,
                "cards_per_pack": self.pack.cards_per_pack,
                "outcomes": [
                    {
                        "outcome_id": outcome.outcome_id,
                        "probability": float(outcome.probability),
                        "base_bundle_value": float(outcome.base_bundle_value),
                        "contents": list(outcome.contents),
                    }
                    for outcome in self.pack.outcomes
                ],
            },
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    """Load strict JSON or YAML settings; pack_config is relative to this file.

    YAML is parsed with ``yaml.safe_load``, which produces the same native
    dict/list/str/int/float/bool/None structure as ``json.load``, so a frozen
    ``.yaml``/``.yml`` configuration (see PF-20, ``configs/env_frozen.yaml``)
    validates identically to the equivalent JSON file and shares its
    ``config_hash`` when the resolved settings match. PyYAML is only imported
    when a YAML file is actually loaded, so JSON-only callers do not need it.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as file:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            raw = yaml.safe_load(file)
        else:
            raw = json.load(file)
        data = _json_record(
            raw,
            {"horizon", "initial_cash", "selling_fee_rate", "market", "pack_config"},
            "environment configuration",
        )
    market = _json_record(
        data["market"], {"initial_regime", "quotes", "transition_matrix"}, "market"
    )
    quotes = _json_record(market["quotes"], {regime.value for regime in REGIME_ORDER}, "quotes")
    matrix = _json_record(
        market["transition_matrix"], {regime.value for regime in REGIME_ORDER}, "transition_matrix"
    )
    parsed_quotes = {}
    for regime in REGIME_ORDER:
        quote = _json_record(
            quotes[regime.value], {"pack_ask", "card_value_multiplier"}, f"{regime.value} quote"
        )
        parsed_quotes[regime] = MarketQuote(
            pack_ask=_positive_float(quote["pack_ask"], "pack_ask"),
            card_value_multiplier=_positive_float(quote["card_value_multiplier"], "card_value_multiplier"),
        )
    if not isinstance(market["initial_regime"], str):
        raise TypeError("initial_regime must be a string")
    parsed_matrix = {}
    for regime in REGIME_ORDER:
        row = matrix[regime.value]
        if not isinstance(row, list):
            raise TypeError("transition rows must be JSON arrays")
        parsed_matrix[regime] = MarketConfig._validated_transition_row(regime, row)
    pack_path = data["pack_config"]
    if not isinstance(pack_path, str) or not pack_path.strip():
        raise ValueError("pack_config must be a nonempty path string")
    return EnvironmentConfig(
        horizon=_integer(data["horizon"], "horizon", minimum=1),
        initial_cash=_positive_float(data["initial_cash"], "initial_cash"),
        selling_fee_rate=_probability(data["selling_fee_rate"], "selling_fee_rate"),
        market=MarketConfig(
            initial_regime=MarketRegime(market["initial_regime"]),
            quotes=parsed_quotes,
            transition_matrix=parsed_matrix,
        ),
        pack=load_pack_config(path.parent / pack_path),
    )


__all__ = [
    "EnvironmentConfig",
    "load_environment_config",
    "MarketConfig",
    "MarketQuote",
    "PROBABILITY_TOLERANCE",
    "REFERENCE_MARKET_CONFIG",
    "REGIME_ORDER",
]
