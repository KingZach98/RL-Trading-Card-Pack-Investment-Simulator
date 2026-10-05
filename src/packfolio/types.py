from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from enum import IntEnum, StrEnum
from math import isfinite
from numbers import Integral, Real
from typing import Self, TypeVar


INTERFACE_VERSION = "2.0"
OBSERVATION_SCHEMA_VERSION = "1.0"
STEP_INFO_SCHEMA_VERSION = "2.0"
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
    BASE_BUNDLE = "base_bundle"
    ROOKIE_BUNDLE = "rookie_bundle"
    AUTOGRAPH_BUNDLE = "autograph_bundle"
    PREMIUM_BUNDLE = "premium_bundle"

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


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _as_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _as_optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, name)


def _as_optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, name, nonnegative=True)


def _read_action(value: object, name: str) -> Action:
    action_id = _as_int(value, name)
    try:
        return Action(action_id)
    except ValueError as error:
        raise ValueError(f"{name} has unknown action ID {action_id}") from error


def _read_enum(value: object,enum_type: type[EnumType], name: str,) -> EnumType:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{name} has unknown value {value!r}") from error


def _read_optional_enum(value: object,enum_type: type[EnumType],name: str,) -> EnumType | None:
    if value is None:
        return None
    return _read_enum(value, enum_type, name)


def _check_version(value: object, expected: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value != expected:
        raise ValueError(f"{name} must be {expected!r}, got {value!r}")


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


@dataclass(frozen=True, slots=True, kw_only=True)
class StepInfo(_SerializableRecord):
    schema_version: str = field(init=False, default=STEP_INFO_SCHEMA_VERSION)
    step_index: int
    requested_action: Action
    executed_action: Action
    action_was_infeasible: bool
    regime_before: MarketRegime
    regime_after: MarketRegime
    cash_before: float
    cash_after: float
    sealed_count_before: int
    sealed_count_after: int
    pack_outcome_id: PackOutcomeId | None
    gross_opened_value: float | None
    fee_paid: float
    portfolio_value_before: float
    portfolio_value_after: float
    reward: float
    termination_reason: TerminationReason | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "step_index",
            _as_int(self.step_index, "step_index", nonnegative=True),)
        _expect_type(self.requested_action, Action, "requested_action")
        _expect_type(self.executed_action, Action, "executed_action")
        _as_bool(self.action_was_infeasible, "action_was_infeasible")
        _expect_type(self.regime_before, MarketRegime, "regime_before")
        _expect_type(self.regime_after, MarketRegime, "regime_after")

        for name in ("cash_before", "cash_after", "fee_paid", "portfolio_value_before", "portfolio_value_after",):
            object.__setattr__(self,name, _as_float(getattr(self, name), name, nonnegative=True),)

        for name in ("sealed_count_before", "sealed_count_after"):
            object.__setattr__(self,name, _as_int(getattr(self, name), name, nonnegative=True),)

        if self.pack_outcome_id is not None:
            _expect_type(self.pack_outcome_id, PackOutcomeId, "pack_outcome_id")
        if self.gross_opened_value is not None:
            object.__setattr__(self,"gross_opened_value",_as_float(self.gross_opened_value,"gross_opened_value",nonnegative=True,),)

        object.__setattr__(self, "reward", _as_float(self.reward, "reward"))
        if self.termination_reason is not None: _expect_type(self.termination_reason,TerminationReason, "termination_reason",)
        self._check_action_result()
        self._check_pack_result()

    def _check_action_result(self) -> None:
        if self.action_was_infeasible:
            if self.requested_action is Action.HOLD:
                raise ValueError("HOLD cannot be infeasible")
            if self.executed_action is not Action.HOLD:
                raise ValueError("an infeasible action must execute HOLD")
        elif self.requested_action is not self.executed_action:
            raise ValueError("a feasible requested action must execute unchanged")

    def _check_pack_result(self) -> None:
        has_outcome = self.pack_outcome_id is not None
        has_value = self.gross_opened_value is not None
        if has_outcome != has_value:
            raise ValueError("pack outcome ID and gross value must appear together")

        opened_pack = self.executed_action is Action.OPEN_AND_SELL
        if opened_pack != has_outcome:
            raise ValueError("pack result must appear only for OPEN_AND_SELL")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        values = _record_data(data, _field_names(cls), cls.__name__)
        _check_version(values.pop("schema_version"),STEP_INFO_SCHEMA_VERSION, "schema_version",)
        values["requested_action"] = _read_action(values["requested_action"],"requested_action",)
        values["executed_action"] = _read_action(values["executed_action"],"executed_action",)
        values["regime_before"] = _read_enum(values["regime_before"],MarketRegime,"regime_before",)
        values["regime_after"] = _read_enum( values["regime_after"],MarketRegime,"regime_after",)
        values["pack_outcome_id"] = _read_optional_enum(values["pack_outcome_id"], PackOutcomeId,"pack_outcome_id",)
        values["termination_reason"] = _read_optional_enum(values["termination_reason"], TerminationReason,"termination_reason",)
        return cls(**values)


_REQUESTED_ACTION_COUNTS = (
    "requested_hold_count",
    "requested_buy_pack_count",
    "requested_open_and_sell_count",
    "requested_sell_pack_count",
)

_EXECUTED_ACTION_COUNTS = (
    "executed_hold_count",
    "executed_buy_pack_count",
    "executed_open_and_sell_count",
    "executed_sell_pack_count",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationRow(_SerializableRecord):
    schema_version: str = field(init=False,default=EVALUATION_ROW_SCHEMA_VERSION,)
    interface_version: str = field(init=False, default=INTERFACE_VERSION)
    simulator_version: str
    observation_schema_version: str = field(init=False,default=OBSERVATION_SCHEMA_VERSION,)
    policy_id: str
    model_id: str | None
    training_seed: int | None
    scenario_id: str
    scenario_seed: int
    config_hash: str
    git_commit: str
    episode_steps: int
    initial_portfolio_value: float
    final_portfolio_value: float
    cumulative_reward: float
    requested_hold_count: int
    requested_buy_pack_count: int
    requested_open_and_sell_count: int
    requested_sell_pack_count: int
    executed_hold_count: int
    executed_buy_pack_count: int
    executed_open_and_sell_count: int
    executed_sell_pack_count: int
    infeasible_action_count: int
    terminated: bool
    truncated: bool
    termination_reason: TerminationReason

    def __post_init__(self) -> None:
        for name in ("simulator_version","policy_id","scenario_id","config_hash","git_commit",):
            object.__setattr__(self, name, _as_string(getattr(self, name), name))

        object.__setattr__(self,"model_id",_as_optional_string(self.model_id, "model_id"),)
        object.__setattr__(self,"training_seed",_as_optional_int(self.training_seed, "training_seed"),)
        object.__setattr__(self,"scenario_seed", _as_int(self.scenario_seed, "scenario_seed", nonnegative=True),)
        object.__setattr__( self,"episode_steps",_as_int(self.episode_steps, "episode_steps", nonnegative=True), )

        for name in ("initial_portfolio_value", "final_portfolio_value"):
            object.__setattr__(self,name,_as_float(getattr(self, name), name, nonnegative=True),)
        object.__setattr__(self,"cumulative_reward",_as_float(self.cumulative_reward, "cumulative_reward"),)

        for name in (*_REQUESTED_ACTION_COUNTS, *_EXECUTED_ACTION_COUNTS,"infeasible_action_count",):
            object.__setattr__(self, name,_as_int(getattr(self, name), name, nonnegative=True),)

        _as_bool(self.terminated, "terminated")
        _as_bool(self.truncated, "truncated")
        _expect_type( self.termination_reason, TerminationReason, "termination_reason",)

        self._check_action_counts()
        self._check_end_state()

    def _check_action_counts(self) -> None:
        requested_total = sum(
            getattr(self, name) for name in _REQUESTED_ACTION_COUNTS)
        if requested_total != self.episode_steps:
            raise ValueError("requested action counts must sum to episode_steps")
        executed_total = sum(
            getattr(self, name) for name in _EXECUTED_ACTION_COUNTS)
        if executed_total != self.episode_steps:
            raise ValueError("executed action counts must sum to episode_steps")
        if self.infeasible_action_count > self.episode_steps:
            raise ValueError("infeasible_action_count cannot exceed episode_steps")

        expected_holds = self.requested_hold_count + self.infeasible_action_count
        if self.executed_hold_count != expected_holds:
            raise ValueError("executed_hold_count must equal requested holds plus " "infeasible actions")

        for action_name in ("buy_pack", "open_and_sell", "sell_pack"):
            requested = getattr(self, f"requested_{action_name}_count")
            executed = getattr(self, f"executed_{action_name}_count")
            if executed > requested:
                raise ValueError(f"executed_{action_name}_count cannot exceed its request count")

    def _check_end_state(self) -> None:
        if self.episode_steps == 0:
            raise ValueError("a completed evaluation row must contain a step")
        if not self.terminated:
            raise ValueError("a Version 1 evaluation row must be terminated")
        if self.truncated:
            raise ValueError("a Version 1 evaluation row cannot be truncated")
        if self.termination_reason is not TerminationReason.HORIZON:
            raise ValueError("a Version 1 evaluation row must end at HORIZON")

    @classmethod
    def from_dict(cls, data: object) -> Self:
        values = _record_data(data, _field_names(cls), cls.__name__)
        _check_version(values.pop("schema_version"),EVALUATION_ROW_SCHEMA_VERSION,"schema_version",)
        _check_version( values.pop("interface_version"), INTERFACE_VERSION, "interface_version",)
        _check_version(values.pop("observation_schema_version"),OBSERVATION_SCHEMA_VERSION,"observation_schema_version",)
        values["termination_reason"] = _read_enum(values["termination_reason"],TerminationReason,"termination_reason",)
        return cls(**values)

__all__ = [
    "Action",
    "EVALUATION_ROW_SCHEMA_VERSION",
    "EvaluationRow",
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
    "StepInfo",
    "TerminationReason",
]
