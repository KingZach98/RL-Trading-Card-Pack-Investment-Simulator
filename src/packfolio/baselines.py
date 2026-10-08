from __future__ import annotations

import numpy as np

from packfolio.config import _positive_float
from packfolio.types import Action, OBSERVATION_SIZE, ObservationIndex


# Reference-fixture defaults retained for existing constructors and exports.
# Real environments must supply their own normalization settings.
INITIAL_CASH = 10_000.0
INVENTORY_CAPACITY = 10
REFERENCE_PRICE = 1_000.0
AFFORDABILITY_TOLERANCE = 1e-3
INVENTORY_TOLERANCE = 1e-6


def _check_observation(observation: np.ndarray) -> None:
    if not isinstance(observation, np.ndarray):
        raise TypeError("observation must be a numpy array")
    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError("observation has the wrong shape")
    if observation.dtype != np.float32:
        raise TypeError("observation must use float32")


def _can_buy(
    observation: np.ndarray, *, initial_cash: float, reference_price: float
) -> bool:
    cash_ratio = observation[ObservationIndex.CASH_RATIO]
    ask_ratio = observation[ObservationIndex.PACK_ASK_RATIO]
    # Both features were rounded independently. Overlapping rounding intervals
    # count as equality; the environment still enforces float64 affordability.
    with np.errstate(over="ignore"):
        cash_upper = (
            float(cash_ratio) + float(np.nextafter(cash_ratio, np.float32(np.inf)))
        ) / 2
        ask_lower = (
            float(ask_ratio) + float(np.nextafter(ask_ratio, np.float32(-np.inf)))
        ) / 2
    required_cash_ratio = ask_lower * reference_price / initial_cash
    return (
        cash_upper >= required_cash_ratio
        and observation[ObservationIndex.SEALED_COUNT_RATIO] < 1.0
    )


def _has_sealed_pack(observation: np.ndarray) -> bool:
    return observation[ObservationIndex.SEALED_COUNT_RATIO] > 0


class CashOnlyPolicy:
    policy_id = "CASH_ONLY"

    def choose_action(self, observation: np.ndarray) -> Action:
        _check_observation(observation)
        return Action.HOLD


class BuyAndHoldPolicy:
    policy_id = "BUY_AND_HOLD"

    def __init__(
        self, *, initial_cash: float = INITIAL_CASH, reference_price: float = REFERENCE_PRICE
    ) -> None:
        self._initial_cash = _positive_float(initial_cash, "initial_cash")
        self._reference_price = _positive_float(reference_price, "reference_price")

    def choose_action(self, observation: np.ndarray) -> Action:
        _check_observation(observation)
        if _can_buy(
            observation, initial_cash=self._initial_cash, reference_price=self._reference_price
        ):
            return Action.BUY_PACK
        return Action.HOLD


class AlwaysOpenPolicy:
    policy_id = "ALWAYS_OPEN"

    def __init__(
        self, *, initial_cash: float = INITIAL_CASH, reference_price: float = REFERENCE_PRICE
    ) -> None:
        self._initial_cash = _positive_float(initial_cash, "initial_cash")
        self._reference_price = _positive_float(reference_price, "reference_price")

    def choose_action(self, observation: np.ndarray) -> Action:
        _check_observation(observation)
        if _has_sealed_pack(observation):
            return Action.OPEN_AND_SELL
        if _can_buy(
            observation, initial_cash=self._initial_cash, reference_price=self._reference_price
        ):
            return Action.BUY_PACK
        return Action.HOLD


CASH_ONLY = CashOnlyPolicy()
BUY_AND_HOLD = BuyAndHoldPolicy()
ALWAYS_OPEN = AlwaysOpenPolicy()


__all__ = [
    "ALWAYS_OPEN",
    "AFFORDABILITY_TOLERANCE",
    "BUY_AND_HOLD",
    "CASH_ONLY",
    "INITIAL_CASH",
    "INVENTORY_CAPACITY",
    "INVENTORY_TOLERANCE",
    "REFERENCE_PRICE",
    "AlwaysOpenPolicy",
    "BuyAndHoldPolicy",
    "CashOnlyPolicy",
]
