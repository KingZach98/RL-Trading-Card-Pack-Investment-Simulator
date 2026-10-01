from dataclasses import FrozenInstanceError

import pytest

from packfolio.types import (
    EVALUATION_ROW_SCHEMA_VERSION,
    INTERFACE_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    STEP_INFO_SCHEMA_VERSION,
    Action,
    MarketRegime,
    MarketSnapshot,
    ObservationIndex,
    PackOutcome,
    PackOutcomeId,
    PortfolioSnapshot,
    PortfolioUpdate,
    Scenario,
    TerminationReason,
)


def make_market() -> MarketSnapshot:
    return MarketSnapshot(regime=MarketRegime.NORMAL,pack_ask=1_000.0,card_value_multiplier=1.0,)


def make_pack_outcome() -> PackOutcome:
    return PackOutcome(outcome_id=PackOutcomeId.MEDIUM_VALUE,base_gross_value=1_000.0,gross_value=1_000.0,)


def test_contract_ids_and_versions_are_fixed():
    assert INTERFACE_VERSION == "1.0"
    assert OBSERVATION_SCHEMA_VERSION == "1.0"
    assert STEP_INFO_SCHEMA_VERSION == "1.0"
    assert EVALUATION_ROW_SCHEMA_VERSION == "1.0"
    assert OBSERVATION_SIZE == 8

    assert {item.name: item.value for item in Action} == {"HOLD": 0,"BUY_PACK": 1,"OPEN_AND_SELL": 2,"SELL_PACK": 3,}
    assert {item.name: item.value for item in ObservationIndex} == {"CASH_RATIO": 0,"SEALED_COUNT_RATIO": 1,"PACK_ASK_RATIO": 2,
        "CARD_VALUE_MULTIPLIER": 3,
        "REMAINING_STEPS_RATIO": 4,                                                           
        "REGIME_LOW": 5,
        "REGIME_NORMAL": 6,
        "REGIME_HIGH": 7,}
    assert [item.value for item in MarketRegime] == ["LOW", "NORMAL", "HIGH"]
    assert [item.value for item in PackOutcomeId] == ["LOW_VALUE","MEDIUM_VALUE","HIGH_VALUE",]
    assert [item.value for item in TerminationReason] == ["HORIZON"]


def test_shared_records_are_immutable_and_use_slots():
    market = make_market()
    with pytest.raises(FrozenInstanceError):
        market.pack_ask = 900.0
    assert not hasattr(market, "__dict__")


@pytest.mark.parametrize(
    "build_record, expected_error",
    [(lambda: MarketSnapshot(regime=MarketRegime.LOW,pack_ask=-1, card_value_multiplier=0.75,),ValueError,),
        (lambda: MarketSnapshot(regime=MarketRegime.LOW, pack_ask=float("inf"), card_value_multiplier=0.75,),ValueError,),
        (lambda: PackOutcome(outcome_id=PackOutcomeId.LOW_VALUE, base_gross_value=200, gross_value=float("nan"),),ValueError,),
        (lambda: PortfolioSnapshot(cash=10_000, sealed_count=True), TypeError,),
        (lambda: PortfolioUpdate(portfolio=PortfolioSnapshot(cash=10_000, sealed_count=0),fee_paid=-1,), ValueError,),],)

def test_shared_records_reject_invalid_numbers(build_record, expected_error):
    with pytest.raises(expected_error):
        build_record()


def test_scenario_checks_tuple_types_and_lengths():
    market = make_market()
    outcome = make_pack_outcome()
    scenario = Scenario(market_path=(market, market),pack_outcomes=(outcome,),)
    assert len(scenario.market_path) == len(scenario.pack_outcomes) + 1
    with pytest.raises(TypeError, match="market_path must be a tuple"):
        Scenario(market_path=[market, market], pack_outcomes=(outcome,))
    with pytest.raises(ValueError, match="one more item"):
        Scenario(market_path=(market,), pack_outcomes=(outcome,))


def test_amounts_are_stored_as_floats():
    market = MarketSnapshot(regime=MarketRegime.NORMAL,pack_ask=1_000,card_value_multiplier=1,)
    assert market.pack_ask == 1_000.0
    assert type(market.pack_ask) is float
    assert type(market.card_value_multiplier) is float


def test_records_require_contract_enum_types():
    with pytest.raises(TypeError, match="regime must be MarketRegime"):
        MarketSnapshot(regime="NORMAL",pack_ask=1_000,card_value_multiplier=1,)

    with pytest.raises(TypeError, match="outcome_id must be PackOutcomeId"):
        PackOutcome(outcome_id="MEDIUM_VALUE",base_gross_value=1_000,gross_value=1_000,)
