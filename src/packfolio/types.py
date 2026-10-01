from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import IntEnum, StrEnum
from math import isfinite
from numbers import Integral, Real
from typing import Self, TypeVar


INTERFACE_VERSION = "1.0"
OBSERVATION_SCHEMA_VERSION = "1.0"
STEP_INFO_SCHEMA_VERSION = "1.0"
EVALUATION_ROW_SCHEMA_VERSION = "1.0"


class Action(IntEnum):
    HOLD = 0
    BUY_PACK = 1
    OPEN_AND_SELL = 2
    SELL_PACK = 3


class ObservationIndex(IntEnum):
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
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"

class PackOutcomeId(StrEnum):
    LOW_VALUE = "LOW_VALUE"
    MEDIUM_VALUE = "MEDIUM_VALUE"
    HIGH_VALUE = "HIGH_VALUE"

class TerminationReason(StrEnum):
    HORIZON = "HORIZON"

EnumType = TypeVar("EnumType", bound=StrEnum)


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


def _read_enum(value: object,enum_type: type[EnumType], name: str,) -> EnumType:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{name} has unknown value {value!r}") from error


def _record_data(data: object,expected_fields: tuple[str, ...], record_name: str,) -> dict[str, object]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{record_name} data must be a mapping")
    expected = set(expected_fields)
    actual = set(data)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if extra:
            names = ", ".join(sorted(str(item) for item in extra))
            details.append(f"extra fields: {names}")
        raise ValueError(f"invalid {record_name} data ({'; '.join(details)})")
    return dict(data)


def _field_names(record: object) -> tuple[str, ...]:
    return tuple(item.name for item in fields(record))


def _serialized_value(value: object) -> object:
    if isinstance(value, Action):
        return int(value)
    if isinstance(value, StrEnum):
        return value.value
    return value


class _SerializableRecord:
    __slots__ = ()
    
    def to_dict(self) -> dict[str, object]:
        return {name: _serialized_value(getattr(self, name))
                for name in _field_names(self)
                }


@dataclass(frozen=True, slots=True)
class MarketSnapshot(_SerializableRecord):
    regime: MarketRegime
    pack_ask: float
    card_value_multiplier: float

    def __post_init__(self) -> None:
        _expect_type(self.regime, MarketRegime, "regime")
        object.__setattr__(self, "pack_ask",_as_float(self.pack_ask, "pack_ask", nonnegative=True),)
        object.__setattr__(self,"card_value_multiplier",_as_float(self.card_value_multiplier,"card_value_multiplier", nonnegative=True,),)

    @classmethod
    def from_dict(cls, data: object) -> Self:
        values = _record_data(data, _field_names(cls), cls.__name__)
        return cls(
            regime=_read_enum(values["regime"], MarketRegime, "regime"),
            pack_ask=values["pack_ask"],
            card_value_multiplier=values["card_value_multiplier"],
        )


@dataclass(frozen=True, slots=True)
class PackOutcome(_SerializableRecord):
    outcome_id: PackOutcomeId
    base_gross_value: float
    gross_value: float

    def __post_init__(self) -> None:
        _expect_type(self.outcome_id, PackOutcomeId, "outcome_id")
        object.__setattr__(self,"base_gross_value",_as_float(self.base_gross_value,"base_gross_value",nonnegative=True,),)
        object.__setattr__(self,"gross_value",_as_float(self.gross_value, "gross_value", nonnegative=True),)

    @classmethod
    def from_dict(cls, data: object) -> Self:
        values = _record_data(data, _field_names(cls), cls.__name__)
        return cls(outcome_id=_read_enum(values["outcome_id"],PackOutcomeId,"outcome_id",),
            base_gross_value=values["base_gross_value"],
            gross_value=values["gross_value"],)


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    cash: float
    sealed_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cash", _as_float(self.cash, "cash", nonnegative=True),)
        object.__setattr__(self, "sealed_count", _as_int(self.sealed_count, "sealed_count", nonnegative=True),)


@dataclass(frozen=True, slots=True)
class PortfolioUpdate:
    portfolio: PortfolioSnapshot
    fee_paid: float

    def __post_init__(self) -> None:
        _expect_type(self.portfolio, PortfolioSnapshot, "portfolio")
        object.__setattr__(self,"fee_paid", _as_float(self.fee_paid, "fee_paid", nonnegative=True),)


@dataclass(frozen=True, slots=True)
class Scenario:
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
                "market_path must have one more item than pack_outcomes")


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
