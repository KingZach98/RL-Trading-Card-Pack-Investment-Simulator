import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from packfolio.packs import (
    PROBABILITY_TOLERANCE,
    PackDraw,
    PackOutcome,
    expected_gross_value,
    load_pack_config,
    sample_pack,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "nfl_pack.json"


@pytest.fixture
def config():
    return load_pack_config(CONFIG_PATH)


def test_config_is_a_distribution_of_complete_nfl_packs(config):
    assert config.pack_id == "synthetic_nfl_five_card"
    assert config.cards_per_pack == 5
    assert sum(config.probabilities) == pytest.approx(1.0, abs=1e-15)
    assert all(outcome.probability >= 0 for outcome in config.outcomes)
    assert all(len(outcome.contents) == 5 for outcome in config.outcomes)
    assert all("NFL" in card for outcome in config.outcomes for card in outcome.contents)


def test_restored_generator_state_reproduces_draws(config):
    rng = np.random.default_rng(1234)
    state = copy.deepcopy(rng.bit_generator.state)
    first = [sample_pack(config, 1.5, rng) for _ in range(100)]
    rng.bit_generator.state = state
    assert [sample_pack(config, 1.5, rng) for _ in range(100)] == first
    assert len({draw.outcome_id for draw in first}) > 1


def test_draw_returns_entire_bundle_and_gross_market_value(config):
    rng = np.random.default_rng(42)
    outcomes = {outcome.outcome_id: outcome for outcome in config.outcomes}
    for _ in range(100):
        draw = sample_pack(config, 2.0, rng)
        outcome = outcomes[draw.outcome_id]
        assert isinstance(draw, PackDraw)
        assert draw.contents == outcome.contents
        assert len(draw.contents) == config.cards_per_pack
        assert draw.gross_bundle_value == outcome.base_bundle_value * 2.0


def test_analytical_expected_value(config):
    # 0.7 * 5 + 0.2 * 15 + 0.09 * 75 + 0.01 * 250 = 15.75.
    assert expected_gross_value(config, 1.0) == pytest.approx(15.75)
    assert expected_gross_value(config, 1.2) == pytest.approx(18.9)
    assert expected_gross_value(config, 2.0) == pytest.approx(31.5)


def test_ev_does_not_modify_configuration_or_random_state(config):
    rng = np.random.default_rng(42)
    state = copy.deepcopy(rng.bit_generator.state)
    before = copy.deepcopy(config)
    expected_gross_value(config, 1.0)
    sample_pack(config, 1.0, np.random.default_rng(7))
    assert rng.bit_generator.state == state
    assert config == before


def test_zero_market_multiplier(config):
    assert expected_gross_value(config, 0.0) == 0.0
    assert sample_pack(config, 0.0, np.random.default_rng(4)).gross_bundle_value == 0.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "1"])
def test_invalid_market_multiplier_does_not_advance_rng(config, value):
    rng = np.random.default_rng(42)
    state = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="market_multiplier"):
        sample_pack(config, value, rng)
    assert rng.bit_generator.state == state
    with pytest.raises(ValueError, match="market_multiplier"):
        expected_gross_value(config, value)


@pytest.mark.parametrize("field", ["probability", "base_bundle_value"])
@pytest.mark.parametrize("value", [-1e-12, float("nan"), float("inf"), True, "0.5"])
def test_invalid_outcome_numbers(config, field, value):
    with pytest.raises(ValueError, match=field):
        replace(config.outcomes[0], **{field: value})


@pytest.mark.parametrize("delta", [-2e-9, 2e-9, 1e-8, -0.1, 0.1])
def test_probability_sum_outside_absolute_tolerance_is_rejected(config, delta):
    first = replace(config.outcomes[0], probability=0.7 + delta)
    with pytest.raises(ValueError, match="sum to one"):
        replace(config, outcomes=(first, *config.outcomes[1:]))


@pytest.mark.parametrize("delta", [-0.5e-9, 0.5e-9])
def test_accepted_rounding_error_is_normalized_for_both_apis(config, delta):
    assert PROBABILITY_TOLERANCE == 1e-9
    first = replace(config.outcomes[0], probability=0.7 + delta)
    rounded = replace(config, outcomes=(first, *config.outcomes[1:]))
    expected = sum(
        outcome.probability * outcome.base_bundle_value for outcome in rounded.outcomes
    ) / sum(outcome.probability for outcome in rounded.outcomes)
    assert sum(rounded.probabilities) == pytest.approx(1.0, abs=1e-15)
    assert expected_gross_value(rounded, 1.0) == pytest.approx(expected, rel=1e-14)
    rng = np.random.default_rng(42)
    reference_rng = np.random.default_rng(42)
    for _ in range(20):
        index = int(reference_rng.choice(len(rounded.outcomes), p=rounded.probabilities))
        draw = sample_pack(rounded, 1.0, rng)
        assert draw.outcome_id == rounded.outcomes[index].outcome_id


def test_zero_probability_outcome_is_never_selected(config):
    outcomes = (
        replace(config.outcomes[0], probability=0.0),
        replace(config.outcomes[1], probability=1.0),
    )
    certain = replace(config, outcomes=outcomes)
    rng = np.random.default_rng(42)
    assert all(sample_pack(certain, 1.0, rng).outcome_id == "rookie_bundle" for _ in range(50))
    assert expected_gross_value(certain, 1.0) == 15.0


def test_zero_value_bundle_is_allowed(config):
    outcome = replace(config.outcomes[0], probability=1.0, base_bundle_value=0.0)
    free = replace(config, outcomes=(outcome,))
    assert expected_gross_value(free, 2.0) == 0.0
    assert sample_pack(free, 2.0, np.random.default_rng(42)).gross_bundle_value == 0.0


def test_overflow_is_rejected_before_advancing_rng(config):
    rng = np.random.default_rng(42)
    state = copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError, match="gross bundle value"):
        sample_pack(config, 1e308, rng)
    assert rng.bit_generator.state == state
    with pytest.raises(ValueError, match="gross bundle value"):
        expected_gross_value(config, 1e308)


@pytest.mark.parametrize("cards_per_pack", [0, -1, 5.0, True])
def test_invalid_pack_size(config, cards_per_pack):
    with pytest.raises(ValueError, match="cards_per_pack"):
        replace(config, cards_per_pack=cards_per_pack)


def test_incomplete_pack_is_rejected(config):
    first = replace(config.outcomes[0], contents=("NFL autograph card",))
    with pytest.raises(ValueError, match="exactly cards_per_pack"):
        replace(config, outcomes=(first, *config.outcomes[1:]))


def test_duplicate_outcome_identifiers(config):
    with pytest.raises(ValueError, match="unique"):
        replace(config, outcomes=(config.outcomes[0], config.outcomes[0]))


@pytest.mark.parametrize("contents", [(), ["NFL card"], ("",), (1,)])
def test_invalid_contents(config, contents):
    with pytest.raises(ValueError, match="description"):
        replace(config.outcomes[0], contents=contents)


def test_empty_distribution(config):
    with pytest.raises(ValueError, match="nonempty"):
        replace(config, outcomes=())


def test_all_zero_probabilities(config):
    with pytest.raises(ValueError, match="sum to one"):
        replace(
            config,
            outcomes=tuple(replace(outcome, probability=0.0) for outcome in config.outcomes),
        )


@pytest.mark.parametrize("identifier", ["", " ", 1])
def test_invalid_identifiers(config, identifier):
    with pytest.raises(ValueError, match="pack_id"):
        replace(config, pack_id=identifier)
    with pytest.raises(ValueError, match="outcome_id"):
        replace(config.outcomes[0], outcome_id=identifier)


@pytest.mark.parametrize(
    "data, message",
    [
        ([], "JSON object"),
        ({}, "missing pack_id"),
        ({"pack_id": "NFL"}, "missing cards_per_pack"),
        ({"pack_id": "NFL", "cards_per_pack": 5}, "missing outcomes"),
        ({"pack_id": "NFL", "cards_per_pack": 5, "outcomes": {}}, "JSON array"),
        ({"pack_id": "NFL", "cards_per_pack": 5, "outcomes": [1]}, "JSON object"),
        ({"pack_id": "NFL", "cards_per_pack": 5, "outcomes": [{}]}, "missing outcome_id"),
    ],
)
def test_invalid_json_schema(tmp_path, data, message):
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_pack_config(path)


@pytest.mark.parametrize("field", ["probability", "base_bundle_value", "contents"])
def test_missing_outcome_field(tmp_path, field):
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    del data["outcomes"][0][field]
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=f"missing {field}"):
        load_pack_config(path)


def test_loader_does_not_treat_string_as_card_array(tmp_path):
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["outcomes"][0]["contents"] = "NFL autograph card"
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="contents must be a JSON array"):
        load_pack_config(path)


def test_loader_reports_missing_file_and_malformed_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pack_config(tmp_path / "missing.json")
    path = tmp_path / "pack.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_pack_config(path)


def test_distribution_matches_configured_odds(config):
    rng = np.random.default_rng(42)
    counts = {outcome.outcome_id: 0 for outcome in config.outcomes}
    for _ in range(20_000):
        counts[sample_pack(config, 1.0, rng).outcome_id] += 1
    for outcome in config.outcomes:
        assert counts[outcome.outcome_id] / 20_000 == pytest.approx(
            outcome.probability, abs=0.01
        )
