from dataclasses import FrozenInstanceError

import pytest

from packfolio.types import (
    EVALUATION_ROW_SCHEMA_VERSION,
    INTERFACE_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    STEP_INFO_SCHEMA_VERSION,
    Action,
    EvaluationRow,
    MarketRegime,
    MarketSnapshot,
    ObservationIndex,
    PackOutcome,
    PackOutcomeId,
    PortfolioSnapshot,
    PortfolioUpdate,
    Scenario,
    StepInfo,
    TerminationReason,
)


def make_market() -> MarketSnapshot:
    return MarketSnapshot(regime=MarketRegime.NORMAL,pack_ask=1_000.0,card_value_multiplier=1.0,)


def make_pack_outcome() -> PackOutcome:
    return PackOutcome(outcome_id=PackOutcomeId.ROOKIE_BUNDLE,base_gross_value=1_000.0,gross_value=1_000.0,)


def make_step_info(**changes: object) -> StepInfo:
    values = {
        "step_index": 0,
        "requested_action": Action.HOLD,
        "executed_action": Action.HOLD,
        "action_was_infeasible": False,
        "regime_before": MarketRegime.NORMAL,
        "regime_after": MarketRegime.NORMAL,
        "cash_before": 10_000.0,
        "cash_after": 10_000.0,
        "sealed_count_before": 0,
        "sealed_count_after": 0,
        "pack_outcome_id": None,
        "gross_opened_value": None,
        "fee_paid": 0.0,
        "portfolio_value_before": 10_000.0,
        "portfolio_value_after": 10_000.0,
        "reward": 0.0,
        "termination_reason": None,
    }
    values.update(changes)
    return StepInfo(**values)


def make_evaluation_row(**changes: object) -> EvaluationRow:
    values = {
        "simulator_version": "0.1.0",
        "policy_id": "CASH_ONLY",
        "model_id": None,
        "training_seed": None,
        "scenario_id": "test-001",
        "scenario_seed": 123,
        "config_hash": "a" * 64,
        "git_commit": "b" * 40,
        "episode_steps": 1,
        "initial_portfolio_value": 10_000.0,
        "final_portfolio_value": 10_000.0,
        "cumulative_reward": 0.0,
        "requested_hold_count": 1,
        "requested_buy_pack_count": 0,
        "requested_open_and_sell_count": 0,
        "requested_sell_pack_count": 0,
        "executed_hold_count": 1,
        "executed_buy_pack_count": 0,
        "executed_open_and_sell_count": 0,
        "executed_sell_pack_count": 0,
        "infeasible_action_count": 0,
        "terminated": True,
        "truncated": False,
        "termination_reason": TerminationReason.HORIZON,
    }
    values.update(changes)
    return EvaluationRow(**values)


def test_contract_ids_and_versions_are_fixed():
    assert INTERFACE_VERSION == "2.0"
    assert OBSERVATION_SCHEMA_VERSION == "1.0"
    assert STEP_INFO_SCHEMA_VERSION == "2.0"
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
    assert [item.value for item in PackOutcomeId] == ["base_bundle", "rookie_bundle", "autograph_bundle", "premium_bundle"]
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
        (lambda: PackOutcome(outcome_id=PackOutcomeId.BASE_BUNDLE, base_gross_value=200, gross_value=float("nan"),),ValueError,),
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
        PackOutcome(outcome_id="rookie_bundle",base_gross_value=1_000,gross_value=1_000,)


def test_step_info_round_trip_uses_public_values():
    step = make_step_info()
    data = step.to_dict()
    assert data["schema_version"] == "2.0"
    assert data["requested_action"] == 0
    assert data["regime_before"] == "NORMAL"
    assert data["pack_outcome_id"] is None
    assert StepInfo.from_dict(data) == step


def test_step_info_accepts_an_infeasible_action():
    step = make_step_info(requested_action=Action.BUY_PACK, executed_action=Action.HOLD,action_was_infeasible=True,)
    assert step.action_was_infeasible is True


@pytest.mark.parametrize("outcome_id", list(PackOutcomeId))
def test_step_info_records_an_opened_pack(outcome_id):
    step = make_step_info(
        requested_action=Action.OPEN_AND_SELL,
        executed_action=Action.OPEN_AND_SELL,
        sealed_count_before=1,
        pack_outcome_id=outcome_id,
        gross_opened_value=1_000,
        fee_paid=50,
    )

    assert step.gross_opened_value == 1_000.0
    assert step.to_dict()["pack_outcome_id"] == outcome_id.value
    assert StepInfo.from_dict(step.to_dict()) == step


@pytest.mark.parametrize("outcome_id", ["LOW_VALUE", "MEDIUM_VALUE", "HIGH_VALUE"])
def test_pack_outcome_loader_rejects_legacy_ids(outcome_id):
    data = make_pack_outcome().to_dict()
    data["outcome_id"] = outcome_id
    with pytest.raises(ValueError, match="outcome_id has unknown value"):
        PackOutcome.from_dict(data)


@pytest.mark.parametrize("outcome_id", ["LOW_VALUE", "MEDIUM_VALUE", "HIGH_VALUE"])
def test_step_info_loader_rejects_legacy_pack_ids(outcome_id):
    data = make_step_info(
        requested_action=Action.OPEN_AND_SELL,
        executed_action=Action.OPEN_AND_SELL,
        sealed_count_before=1,
        pack_outcome_id=PackOutcomeId.ROOKIE_BUNDLE,
        gross_opened_value=15.0,
    ).to_dict()
    data["pack_outcome_id"] = outcome_id
    with pytest.raises(ValueError, match="pack_outcome_id has unknown value"):
        StepInfo.from_dict(data)


@pytest.mark.parametrize(
        "changes, message",
        [
            ({"requested_action": Action.BUY_PACK, "executed_action": Action.HOLD,},"must execute unchanged",),
            ({"action_was_infeasible": True},"HOLD cannot be infeasible",),
            ({"requested_action": Action.BUY_PACK,"executed_action": Action.SELL_PACK,"action_was_infeasible": True,},"must execute HOLD",),
            ({"requested_action": Action.OPEN_AND_SELL, "executed_action": Action.OPEN_AND_SELL, },"pack result must appear",),
            ({"pack_outcome_id": PackOutcomeId.BASE_BUNDLE,"gross_opened_value": 200, },"pack result must appear only",),
        ],)

def test_step_info_rejects_impossible_action_results(changes, message):
    with pytest.raises(ValueError, match=message):
        make_step_info(**changes)


def test_step_info_rejects_invalid_field_values():
    with pytest.raises(ValueError, match="fee_paid must not be negative"):
        make_step_info(fee_paid=-1)

    with pytest.raises(TypeError, match="action_was_infeasible must be a boolean"):
        make_step_info(action_was_infeasible=1)

    with pytest.raises(TypeError, match="gross_opened_value must be a number"):
        make_step_info(gross_opened_value=True)


def test_step_info_loader_checks_versions_and_action_ids():
    wrong_version = make_step_info().to_dict()
    wrong_version["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="schema_version must be '2.0'"):
        StepInfo.from_dict(wrong_version)

    boolean_action = make_step_info().to_dict()
    boolean_action["requested_action"] = True
    with pytest.raises(TypeError, match="requested_action must be an integer"):
        StepInfo.from_dict(boolean_action)


def test_evaluation_row_round_trip_uses_schema_versions():
    row = make_evaluation_row()
    data = row.to_dict()
    assert data["schema_version"] == "1.0"
    assert data["interface_version"] == "2.0"
    assert data["observation_schema_version"] == "1.0"
    assert data["model_id"] is None
    assert data["termination_reason"] == "HORIZON"
    assert EvaluationRow.from_dict(data) == row


@pytest.mark.parametrize(
        "changes, message",
        [
            ({"requested_hold_count": 0}, "requested action counts must sum",),
            ({"executed_hold_count": 0},"executed action counts must sum",),
            ({"infeasible_action_count": 2},"infeasible_action_count cannot exceed",),
            ({"infeasible_action_count": 1},"executed_hold_count must equal",),
            ({"requested_hold_count": -1},"requested_hold_count must not be negative",),
        ],)

def test_evaluation_row_rejects_invalid_action_totals(changes, message):
    with pytest.raises(ValueError, match=message):
        make_evaluation_row(**changes)


def test_evaluation_row_rejects_impossible_action_counts():
    with pytest.raises(ValueError, match="executed_sell_pack_count cannot exceed"):
        make_evaluation_row(
            episode_steps=2,
            requested_hold_count=0,
            requested_buy_pack_count=1,
            requested_sell_pack_count=1,
            executed_hold_count=0,
            executed_buy_pack_count=0,
            executed_sell_pack_count=2,
        )


@pytest.mark.parametrize(
        "changes, message",
        [
            ({"episode_steps": 0,"requested_hold_count": 0,"executed_hold_count": 0,},"must contain a step",),
            ({"scenario_seed": -1}, "scenario_seed must not be negative"),
            ({"training_seed": -1}, "training_seed must not be negative"),
            ({"terminated": False}, "must be terminated"),
            ({"truncated": True}, "cannot be truncated"),
        ],)

def test_evaluation_row_rejects_invalid_run_details(changes, message):
    with pytest.raises(ValueError, match=message):
        make_evaluation_row(**changes)


def test_evaluation_row_loader_checks_versions_and_fields():
    wrong_version = make_evaluation_row().to_dict()
    wrong_version["interface_version"] = "1.0"
    with pytest.raises(ValueError, match="interface_version must be '2.0'"):
        EvaluationRow.from_dict(wrong_version)

    missing_field = make_evaluation_row().to_dict()
    missing_field.pop("policy_id")
    with pytest.raises(ValueError, match="missing fields: policy_id"):
        EvaluationRow.from_dict(missing_field)
