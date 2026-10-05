import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from packfolio.packs import load_pack_config
from packfolio.types import (
    INTERFACE_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SIZE,
    Action,
    EvaluationRow,
    MarketRegime,
    MarketSnapshot,
    ObservationIndex,
    PackOutcome,
    PackOutcomeId,
    StepInfo,
)


FIXTURE_DIR = Path(__file__).with_name("fixtures")


def load_fixture(name: str) -> dict[str, object]:
    path = FIXTURE_DIR / name
    with path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


class HoldPolicy:
    policy_id = "CASH_ONLY"

    def choose_action(self, observation: np.ndarray) -> Action:
        if observation.shape != (OBSERVATION_SIZE,):
            raise ValueError("observation has the wrong shape")
        if observation.dtype != np.float32:
            raise TypeError("observation must use float32")
        return Action.HOLD


def test_market_fixture_matches_public_observation_fields():
    fixture = load_fixture("market_snapshot_v2.json")
    assert fixture["interface_version"] == INTERFACE_VERSION

    snapshot = MarketSnapshot.from_dict(fixture["market_snapshot"])
    reference_price = fixture["reference_price"]
    expected = fixture["expected_observation_fields"]

    observed = {
        "pack_ask_ratio": snapshot.pack_ask / reference_price,
        "card_value_multiplier": snapshot.card_value_multiplier,
        "regime_low": float(snapshot.regime is MarketRegime.LOW),
        "regime_normal": float(snapshot.regime is MarketRegime.NORMAL),
        "regime_high": float(snapshot.regime is MarketRegime.HIGH),
    }
    assert observed == expected
    assert {item.name for item in fields(MarketSnapshot)} == {"regime","pack_ask", "card_value_multiplier",}
    assert "cash" not in fixture["market_snapshot"]
    assert "sealed_count" not in fixture["market_snapshot"]


def test_pack_fixture_has_no_portfolio_state():
    fixture = load_fixture("pack_outcome_v2.json")
    assert fixture["interface_version"] == INTERFACE_VERSION

    outcome_data = fixture["pack_outcome"]
    outcome = PackOutcome.from_dict(outcome_data)

    assert outcome.to_dict() == outcome_data
    assert {item.name for item in fields(PackOutcome)} == {"outcome_id", "base_gross_value","gross_value",}
    assert "cash" not in outcome_data
    assert "sealed_count" not in outcome_data


def test_shared_pack_outcomes_match_the_pack_config():
    config = load_pack_config(Path(__file__).resolve().parents[1] / "configs" / "nfl_pack.json")
    assert {outcome.outcome_id for outcome in config.outcomes} == {item.value for item in PackOutcomeId}
    for configured in config.outcomes:
        outcome = PackOutcome(
            outcome_id=PackOutcomeId(configured.outcome_id),
            base_gross_value=configured.base_bundle_value,
            gross_value=configured.base_bundle_value * 1.6,
        )
        assert outcome.to_dict()["outcome_id"] == configured.outcome_id
        assert PackOutcome.from_dict(outcome.to_dict()) == outcome


def test_fixture_records_reject_bad_fields():
    market_data = load_fixture("market_snapshot_v2.json")["market_snapshot"]
    pack_data = load_fixture("pack_outcome_v2.json")["pack_outcome"]

    missing_pack_ask = dict(market_data)
    missing_pack_ask.pop("pack_ask")
    with pytest.raises(ValueError, match="missing fields: pack_ask"):
        MarketSnapshot.from_dict(missing_pack_ask)

    market_with_cash = dict(market_data)
    market_with_cash["cash"] = 10_000.0
    with pytest.raises(ValueError, match="extra fields: cash"):
        MarketSnapshot.from_dict(market_with_cash)

    wrong_pack_ask = dict(market_data)
    wrong_pack_ask["pack_ask"] = True
    with pytest.raises(TypeError, match="pack_ask must be a number"):
        MarketSnapshot.from_dict(wrong_pack_ask)

    pack_with_inventory = dict(pack_data)
    pack_with_inventory["sealed_count"] = 1
    with pytest.raises(ValueError, match="extra fields: sealed_count"):
        PackOutcome.from_dict(pack_with_inventory)


def test_fixture_exchange_between_policy_environment_and_evaluator():
    fixture = load_fixture("contract_exchange_v2.json")
    assert fixture["interface_version"] == INTERFACE_VERSION
    assert fixture["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION

    observation = np.asarray(fixture["observation"], dtype=np.float32)
    policy = HoldPolicy()
    requested_action = policy.choose_action(observation)
    step = StepInfo.from_dict(fixture["step_info"])
    row = EvaluationRow.from_dict(fixture["evaluation_row"])

    assert requested_action == fixture["expected_policy_action"]
    assert requested_action is step.requested_action
    assert row.policy_id == policy.policy_id
    assert row.episode_steps == step.step_index + 1
    assert row.final_portfolio_value == step.portfolio_value_after
    assert row.cumulative_reward == step.reward
    requested_counts = {Action.HOLD: row.requested_hold_count, Action.BUY_PACK: row.requested_buy_pack_count,
                        Action.OPEN_AND_SELL: row.requested_open_and_sell_count,
                        Action.SELL_PACK: row.requested_sell_pack_count,
                        }
    executed_counts = {Action.HOLD: row.executed_hold_count, Action.BUY_PACK: row.executed_buy_pack_count,
                       Action.OPEN_AND_SELL: row.executed_open_and_sell_count,
                       Action.SELL_PACK: row.executed_sell_pack_count,
                       }
    assert requested_counts[step.requested_action] == 1
    assert executed_counts[step.executed_action] == 1
    assert row.infeasible_action_count == int(step.action_was_infeasible)
    assert row.termination_reason is step.termination_reason
    assert observation[ObservationIndex.REMAINING_STEPS_RATIO] == 1.0
