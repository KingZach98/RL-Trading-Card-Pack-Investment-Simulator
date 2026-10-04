from __future__ import annotations

import numpy as np

from packfolio.types import Action, OBSERVATION_SIZE, ObservationIndex


# These match the current project specification. Replace them with shared
# validated config values when the configuration system is implemented.
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


def _can_buy(observation: np.ndarray) -> bool:
    cash = float(observation[ObservationIndex.CASH_RATIO]) * INITIAL_CASH
    sealed_count = (
        float(observation[ObservationIndex.SEALED_COUNT_RATIO])
        * INVENTORY_CAPACITY
    )
    pack_ask = float(observation[ObservationIndex.PACK_ASK_RATIO]) * REFERENCE_PRICE
    return (
        cash + AFFORDABILITY_TOLERANCE >= pack_ask
        and sealed_count < INVENTORY_CAPACITY - INVENTORY_TOLERANCE
    )


def _has_sealed_pack(observation: np.ndarray) -> bool:
    sealed_count = (
        float(observation[ObservationIndex.SEALED_COUNT_RATIO])
        * INVENTORY_CAPACITY
    )
    return sealed_count > 0


class CashOnlyPolicy:
    policy_id = "CASH_ONLY"

    def choose_action(self, observation: np.ndarray) -> Action:
        _check_observation(observation)
        return Action.HOLD


class BuyAndHoldPolicy:
    policy_id = "BUY_AND_HOLD"

    def choose_action(self, observation: np.ndarray) -> Action:
        _check_observation(observation)
        if _can_buy(observation):
            return Action.BUY_PACK
        return Action.HOLD


class AlwaysOpenPolicy:
    policy_id = "ALWAYS_OPEN"

    def choose_action(self, observation: np.ndarray) -> Action:
        _check_observation(observation)
        if _has_sealed_pack(observation):
            return Action.OPEN_AND_SELL
        if _can_buy(observation):
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
