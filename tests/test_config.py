from dataclasses import replace
import json
from pathlib import Path

import pytest

from packfolio.config import load_environment_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config():
    return load_environment_config(CONFIG_DIR / "environment.json")


def test_starter_configuration(config):
    assert config.horizon == 100
    assert config.initial_cash == 100.0
    assert config.selling_fee_rate == 0.05
    assert config.pack.cards_per_pack == 5
    assert len(config.market.quotes) == 3


@pytest.mark.parametrize(
    "field, value",
    [
        ("horizon", 0), ("horizon", -1), ("horizon", True), ("horizon", 1.5),
        ("initial_cash", 0), ("initial_cash", -1), ("initial_cash", float("nan")),
        ("initial_cash", float("inf")), ("initial_cash", True),
        ("selling_fee_rate", -0.01), ("selling_fee_rate", 1.01),
        ("selling_fee_rate", float("nan")), ("selling_fee_rate", True),
        ("market", None), ("pack", None),
    ],
)
def test_invalid_environment_settings(config, field, value):
    with pytest.raises((ValueError, TypeError), match=field):
        replace(config, **{field: value})


def test_fee_boundaries(config):
    assert replace(config, selling_fee_rate=0).selling_fee_rate == 0.0
    assert replace(config, selling_fee_rate=1).selling_fee_rate == 1.0


def test_config_hash_covers_all_behavioral_settings(config):
    modifications = [
        replace(config, horizon=101),
        replace(config, initial_cash=200),
        replace(config, selling_fee_rate=0.1),
        replace(config, market=replace(config.market, initial_regime=next(iter(config.market.quotes)))),
        replace(config, pack=replace(config.pack, pack_id="another-pack")),
    ]
    for modified in modifications:
        assert modified.config_hash != config.config_hash
    data = config.to_dict()
    data["horizon"] = 999
    assert config.horizon == 100


def test_hash_independent_of_file_location_and_key_order(tmp_path, config):
    data = json.loads((CONFIG_DIR / "environment.json").read_text(encoding="utf-8"))
    data["pack_config"] = str(CONFIG_DIR / "nfl_pack.json")
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(dict(reversed(list(data.items())))), encoding="utf-8")
    assert load_environment_config(path).config_hash == config.config_hash


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data.update(typo=1), "exactly"),
        (lambda data: data.pop("horizon"), "exactly"),
        (lambda data: data.update(pack_config=""), "pack_config"),
        (lambda data: data.update(market=[]), "market"),
        (lambda data: data["market"].update(initial_regime="UNKNOWN"), "UNKNOWN"),
        (lambda data: data["market"]["quotes"].pop("LOW"), "quotes"),
        (lambda data: data["market"]["quotes"]["LOW"].update(pack_ask=-1), "pack_ask"),
        (lambda data: data["market"]["transition_matrix"].update(LOW=[0.1, 0.1, 0.1]), "sum"),
        (lambda data: data["market"]["transition_matrix"].update(LOW="invalid"), "arrays"),
    ],
)
def test_loader_rejects_invalid_configuration(tmp_path, mutation, message):
    data = json.loads((CONFIG_DIR / "environment.json").read_text(encoding="utf-8"))
    data["pack_config"] = str(CONFIG_DIR / "nfl_pack.json")
    mutation(data)
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=message):
        load_environment_config(path)


def test_market_scaled_pack_overflow_is_rejected(config):
    outcomes = tuple(replace(outcome, base_bundle_value=1.7e308) for outcome in config.pack.outcomes)
    with pytest.raises(ValueError, match="finite"):
        replace(config, pack=replace(config.pack, outcomes=outcomes))


def test_io_errors_are_not_hidden(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_environment_config(tmp_path / "missing.json")
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_environment_config(path)


def test_probability_and_price_assumptions_are_labeled_synthetic():
    pack_data = json.loads((CONFIG_DIR / "nfl_pack.json").read_text(encoding="utf-8"))
    config_readme = (CONFIG_DIR / "README.md").read_text(encoding="utf-8")

    assert "synthetic" in pack_data["description"].lower()
    assert "not real product odds or market prices" in pack_data["description"].lower()
    assert "environment.json` is an editable synthetic starter" in config_readme
    assert "low/normal/high asks of 8/10/12" in config_readme
