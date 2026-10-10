from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

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


def test_yaml_loader_matches_equivalent_json(tmp_path, config):
    """PF-20: YAML configs must validate and hash identically to equivalent JSON."""
    data = json.loads((CONFIG_DIR / "environment.json").read_text(encoding="utf-8"))
    data["pack_config"] = str(CONFIG_DIR / "nfl_pack.json")
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    loaded = load_environment_config(path)
    assert loaded.config_hash == config.config_hash
    assert loaded.to_dict() == config.to_dict()


def test_yaml_loader_rejects_invalid_configuration(tmp_path):
    """YAML goes through the same strict validation as JSON, not a looser path."""
    data = json.loads((CONFIG_DIR / "environment.json").read_text(encoding="utf-8"))
    data["pack_config"] = str(CONFIG_DIR / "nfl_pack.json")
    data["market"]["quotes"]["LOW"]["pack_ask"] = -1
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match="pack_ask"):
        load_environment_config(path)


def test_env_frozen_yaml_matches_environment_json_hash(config):
    """PF-20: the frozen base config must be bit-for-bit equivalent to the
    config currently used by train/evaluate, so freezing it changes nothing."""
    frozen = load_environment_config(CONFIG_DIR / "env_frozen.yaml")
    assert frozen.config_hash == config.config_hash
    assert frozen.to_dict() == config.to_dict()


@pytest.mark.parametrize(
    "variant_path, expected_fee",
    [
        ("env_frozen_fee_low.yaml", 0.025),
        ("env_frozen_fee_high.yaml", 0.10),
    ],
)
def test_env_frozen_fee_variants_change_only_the_fee(config, variant_path, expected_fee):
    """PF-20: predetermined fee-sensitivity variants must differ from the
    frozen base only in selling_fee_rate, and must have their own distinct
    hash so comparisons can tell them apart."""
    variant = load_environment_config(CONFIG_DIR / variant_path)
    assert variant.selling_fee_rate == expected_fee
    assert variant.config_hash != config.config_hash
    assert replace(variant, selling_fee_rate=config.selling_fee_rate).to_dict() == config.to_dict()


def test_probability_and_price_assumptions_are_labeled_synthetic():
    pack_data = json.loads((CONFIG_DIR / "nfl_pack.json").read_text(encoding="utf-8"))
    config_readme = (CONFIG_DIR / "README.md").read_text(encoding="utf-8")

    assert "synthetic" in pack_data["description"].lower()
    assert "not real product odds or market prices" in pack_data["description"].lower()
    assert "environment.json` is an editable synthetic starter" in config_readme
    assert "low/normal/high asks of 8/10/12" in config_readme
