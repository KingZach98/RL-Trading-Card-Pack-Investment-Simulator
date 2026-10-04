import numpy as np
import pytest

from packfolio.baselines import (
    ALWAYS_OPEN,
    BUY_AND_HOLD,
    CASH_ONLY,
    INITIAL_CASH,
    INVENTORY_CAPACITY,
    REFERENCE_PRICE,
    AlwaysOpenPolicy,
    BuyAndHoldPolicy,
    CashOnlyPolicy,
)
from packfolio.types import Action, OBSERVATION_SIZE, ObservationIndex


def make_observation(
    *,
    cash: float = INITIAL_CASH,
    sealed_count: int = 0,
    pack_ask: float = REFERENCE_PRICE,
) -> np.ndarray:
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    observation[ObservationIndex.CASH_RATIO] = cash / INITIAL_CASH
    observation[ObservationIndex.SEALED_COUNT_RATIO] = (
        sealed_count / INVENTORY_CAPACITY
    )
    observation[ObservationIndex.PACK_ASK_RATIO] = pack_ask / REFERENCE_PRICE
    observation[ObservationIndex.CARD_VALUE_MULTIPLIER] = 1.0
    observation[ObservationIndex.REMAINING_STEPS_RATIO] = 1.0
    observation[ObservationIndex.REGIME_NORMAL] = 1.0
    return observation


@pytest.mark.parametrize(
    "policy",
    [CASH_ONLY, CashOnlyPolicy()],
)
def test_cash_only_always_holds(policy):
    assert policy.choose_action(make_observation(cash=INITIAL_CASH)) is Action.HOLD
    assert policy.choose_action(make_observation(cash=0.0)) is Action.HOLD
    assert policy.choose_action(make_observation(sealed_count=1)) is Action.HOLD


@pytest.mark.parametrize(
    "policy",
    [BUY_AND_HOLD, BuyAndHoldPolicy()],
)
def test_buy_and_hold_buys_when_affordable_and_capacity_available(policy):
    observation = make_observation(
        cash=REFERENCE_PRICE,
        sealed_count=INVENTORY_CAPACITY - 1,
        pack_ask=REFERENCE_PRICE,
    )

    assert policy.choose_action(observation) is Action.BUY_PACK


@pytest.mark.parametrize("pack_ask", [800.0, 1_000.0, 1_200.0])
def test_buy_and_hold_buys_when_cash_exactly_matches_pack_ask(pack_ask):
    observation = make_observation(
        cash=pack_ask,
        sealed_count=0,
        pack_ask=pack_ask,
    )

    assert BUY_AND_HOLD.choose_action(observation) is Action.BUY_PACK


def test_buy_and_hold_holds_when_cash_is_insufficient():
    observation = make_observation(
        cash=REFERENCE_PRICE - 1.0,
        pack_ask=REFERENCE_PRICE,
    )

    assert BUY_AND_HOLD.choose_action(observation) is Action.HOLD


def test_buy_and_hold_holds_when_cash_is_slightly_insufficient():
    observation = make_observation(
        cash=799.0,
        sealed_count=0,
        pack_ask=800.0,
    )

    assert BUY_AND_HOLD.choose_action(observation) is Action.HOLD


def test_buy_and_hold_holds_when_inventory_is_at_capacity():
    observation = make_observation(
        cash=INITIAL_CASH,
        sealed_count=INVENTORY_CAPACITY,
    )

    assert BUY_AND_HOLD.choose_action(observation) is Action.HOLD


def test_buy_and_hold_buys_when_inventory_is_one_below_capacity():
    observation = make_observation(
        cash=INITIAL_CASH,
        sealed_count=INVENTORY_CAPACITY - 1,
    )

    assert BUY_AND_HOLD.choose_action(observation) is Action.BUY_PACK


def test_always_open_opens_when_a_sealed_pack_exists():
    observation = make_observation(
        cash=INITIAL_CASH,
        sealed_count=1,
        pack_ask=REFERENCE_PRICE,
    )

    assert ALWAYS_OPEN.choose_action(observation) is Action.OPEN_AND_SELL


@pytest.mark.parametrize(
    "policy",
    [ALWAYS_OPEN, AlwaysOpenPolicy()],
)
def test_always_open_buys_when_no_sealed_pack_exists_and_buying_is_feasible(policy):
    observation = make_observation(
        cash=REFERENCE_PRICE,
        sealed_count=0,
        pack_ask=REFERENCE_PRICE,
    )

    assert policy.choose_action(observation) is Action.BUY_PACK


def test_always_open_holds_when_neither_opening_nor_buying_is_feasible():
    observation = make_observation(
        cash=REFERENCE_PRICE - 1.0,
        sealed_count=0,
        pack_ask=REFERENCE_PRICE,
    )

    assert ALWAYS_OPEN.choose_action(observation) is Action.HOLD


@pytest.mark.parametrize("policy", [CASH_ONLY, BUY_AND_HOLD, ALWAYS_OPEN])
def test_policies_return_action_enum_values(policy):
    action = policy.choose_action(make_observation())

    assert isinstance(action, Action)


@pytest.mark.parametrize("policy", [CASH_ONLY, BUY_AND_HOLD, ALWAYS_OPEN])
def test_policies_do_not_mutate_the_supplied_observation(policy):
    observation = make_observation()
    original = observation.copy()

    policy.choose_action(observation)

    np.testing.assert_array_equal(observation, original)


@pytest.mark.parametrize("policy", [CASH_ONLY, BUY_AND_HOLD, ALWAYS_OPEN])
def test_repeated_calls_on_same_observation_produce_same_action(policy):
    observation = make_observation(cash=INITIAL_CASH, sealed_count=0)

    first_action = policy.choose_action(observation)
    second_action = policy.choose_action(observation)

    assert first_action is second_action
