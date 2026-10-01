"""Shared data contracts for the Packfolio simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from math import isfinite
from numbers import Integral, Real


INTERFACE_VERSION = "1.0"
OBSERVATION_SCHEMA_VERSION = "1.0"
STEP_INFO_SCHEMA_VERSION = "1.0"
EVALUATION_ROW_SCHEMA_VERSION = "1.0"


class Action(IntEnum):
    """Actions accepted by the environment."""

    HOLD = 0
    BUY_PACK = 1
    OPEN_AND_SELL = 2
    SELL_PACK = 3


class ObservationIndex(IntEnum):
    """Positions in the Version 1 observation array."""

    CASH_RATIO = 0
    SEALED_COUNT_RATIO = 1
    PACK_ASK_RATIO = 2
    CARD_VALUE_MULTIPLIER = 3
    REMAINING_STEPS_RATIO = 4
    REGIME_LOW = 5
    REGIME_NORMAL = 6
    REGIME_HIGH = 7


OBSERVATION_SIZE = len(ObservationIndex)


class MarketRegime(StrEnum):
    """Visible market regimes."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


class PackOutcomeId(StrEnum):
    """Whole-pack outcome groups."""

    LOW_VALUE = "LOW_VALUE"
    MEDIUM_VALUE = "MEDIUM_VALUE"
    HIGH_VALUE = "HIGH_VALUE"


class TerminationReason(StrEnum):
    """Reasons a Version 1 episode can end."""

    HORIZON = "HORIZON"


def _as_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number")

    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _as_int(value: object, name: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")

    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _expect_type(value: object, expected_type: type, name: str) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__}")


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Visible values for one market regime."""

    regime: MarketRegime
    pack_ask: float
    card_value_multiplier: float

    def __post_init__(self) -> None:
        _expect_type(self.regime, MarketRegime, "regime")
        object.__setattr__(
            self,
            "pack_ask",
            _as_float(self.pack_ask, "pack_ask", nonnegative=True),
        )
        object.__setattr__(
            self,
            "card_value_multiplier",
            _as_float(
                self.card_value_multiplier,
                "card_value_multiplier",
                nonnegative=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class PackOutcome:
    """Gross result of opening one pack."""

    outcome_id: PackOutcomeId
    base_gross_value: float
    gross_value: float

    def __post_init__(self) -> None:
        _expect_type(self.outcome_id, PackOutcomeId, "outcome_id")
        object.__setattr__(
            self,
            "base_gross_value",
            _as_float(
                self.base_gross_value,
                "base_gross_value",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "gross_value",
            _as_float(self.gross_value, "gross_value", nonnegative=True),
        )


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """Cash and sealed inventory at one point in time."""

    cash: float
    sealed_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cash",
            _as_float(self.cash, "cash", nonnegative=True),
        )
        object.__setattr__(
            self,
            "sealed_count",
            _as_int(self.sealed_count, "sealed_count", nonnegative=True),
        )


@dataclass(frozen=True, slots=True)
class PortfolioUpdate:
    """Portfolio state and fee produced by one accounting operation."""

    portfolio: PortfolioSnapshot
    fee_paid: float

    def __post_init__(self) -> None:
        _expect_type(self.portfolio, PortfolioSnapshot, "portfolio")
        object.__setattr__(
            self,
            "fee_paid",
            _as_float(self.fee_paid, "fee_paid", nonnegative=True),
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    """Private market path and pack outcomes for one episode."""

    market_path: tuple[MarketSnapshot, ...]
    pack_outcomes: tuple[PackOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.market_path, tuple):
            raise TypeError("market_path must be a tuple")
        if not isinstance(self.pack_outcomes, tuple):
            raise TypeError("pack_outcomes must be a tuple")
        if not all(isinstance(item, MarketSnapshot) for item in self.market_path):
            raise TypeError("market_path must contain MarketSnapshot values")
        if not all(isinstance(item, PackOutcome) for item in self.pack_outcomes):
            raise TypeError("pack_outcomes must contain PackOutcome values")
        if len(self.market_path) != len(self.pack_outcomes) + 1:
            raise ValueError(
                "market_path must have one more item than pack_outcomes"
            )


__all__ = [
    "Action",
    "EVALUATION_ROW_SCHEMA_VERSION",
    "INTERFACE_VERSION",
    "MarketRegime",
    "MarketSnapshot",
    "OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_SIZE",
    "ObservationIndex",
    "PackOutcome",
    "PackOutcomeId",
    "PortfolioSnapshot",
    "PortfolioUpdate",
    "STEP_INFO_SCHEMA_VERSION",
    "Scenario",
    "TerminationReason",
]
