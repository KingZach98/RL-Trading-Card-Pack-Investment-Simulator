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
    initial_cash: float = INITIAL_CASH,
    reference_price: float = REFERENCE_PRICE,
    inventory_capacity: int = INVENTORY_CAPACITY,
) -> np.ndarray:
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    observation[ObservationIndex.CASH_RATIO] = cash / initial_cash
    observation[ObservationIndex.SEALED_COUNT_RATIO] = (
        sealed_count / inventory_capacity
    )
    observation[ObservationIndex.PACK_ASK_RATIO] = pack_ask / reference_price
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


@pytest.mark.parametrize("policy_type", [BuyAndHoldPolicy, AlwaysOpenPolicy])
@pytest.mark.parametrize("initial_cash, reference_price", [(5.0, 10.0), (100.0, 20.0), (200.0, 10.0)])
@pytest.mark.parametrize("cash, expected", [(5.0, Action.HOLD), (10.0, Action.BUY_PACK)])
def test_buying_uses_supplied_normalization(policy_type, initial_cash, reference_price, cash, expected):
    policy = policy_type(initial_cash=initial_cash, reference_price=reference_price)
    observation = make_observation(
        cash=cash, pack_ask=10.0, initial_cash=initial_cash, reference_price=reference_price
    )
    assert policy.choose_action(observation) is expected


@pytest.mark.parametrize("policy_type", [BuyAndHoldPolicy, AlwaysOpenPolicy])
def test_no_monetary_allowance_for_a_distinguishably_unaffordable_buy(policy_type):
    observation = make_observation(cash=999.9995, pack_ask=1000.0)
    assert policy_type().choose_action(observation) is Action.HOLD


@pytest.mark.parametrize("policy_type", [BuyAndHoldPolicy, AlwaysOpenPolicy])
def test_float32_equality_and_adjacent_cash_ratios(policy_type):
    policy = policy_type(initial_cash=100.0, reference_price=20.0)
    observation = make_observation(cash=8.0, pack_ask=8.0, initial_cash=100.0, reference_price=20.0)
    assert float(observation[ObservationIndex.CASH_RATIO]) * 100.0 < 8.0
    assert policy.choose_action(observation) is Action.BUY_PACK
    observation[ObservationIndex.CASH_RATIO] = np.nextafter(
        observation[ObservationIndex.CASH_RATIO], np.float32(0.0)
    )
    assert policy.choose_action(observation) is Action.HOLD
    observation[ObservationIndex.CASH_RATIO] = np.nextafter(np.float32(0.08), np.float32(np.inf))
    assert policy.choose_action(observation) is Action.BUY_PACK


def test_indistinguishable_float64_cash_states_have_the_same_decision():
    affordable = make_observation(cash=10.0, pack_ask=10.0, initial_cash=100.0, reference_price=10.0)
    below = make_observation(
        cash=float(np.nextafter(10.0, 0.0)), pack_ask=10.0, initial_cash=100.0, reference_price=10.0
    )
    np.testing.assert_array_equal(affordable, below)
    policy = BuyAndHoldPolicy(initial_cash=100.0, reference_price=10.0)
    assert policy.choose_action(affordable) is policy.choose_action(below) is Action.BUY_PACK


@pytest.mark.parametrize("capacity", [1, 4, 25])
def test_inventory_rules_use_normalized_ownership_and_capacity(capacity):
    assert BUY_AND_HOLD.choose_action(make_observation(sealed_count=capacity, inventory_capacity=capacity)) is Action.HOLD
    assert BUY_AND_HOLD.choose_action(make_observation(sealed_count=capacity - 1, inventory_capacity=capacity)) is Action.BUY_PACK
    assert ALWAYS_OPEN.choose_action(make_observation(sealed_count=1, inventory_capacity=capacity)) is Action.OPEN_AND_SELL


@pytest.mark.parametrize("policy_type", [BuyAndHoldPolicy, AlwaysOpenPolicy])
@pytest.mark.parametrize("field, value", [("initial_cash", 0), ("reference_price", -1), ("initial_cash", float("nan")), ("reference_price", True)])
def test_invalid_normalization_is_rejected(policy_type, field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        policy_type(**{field: value})
