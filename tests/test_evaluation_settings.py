from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from packfolio.agents.dqn_agent import CheckpointMetadata
from packfolio.baselines import CashOnlyPolicy
from packfolio.config import load_environment_config
from packfolio.evaluate import evaluate_checkpoint_on_split, read_evaluation_manifest, read_evaluation_rows_jsonl
from packfolio.scenarios import SIMULATOR_VERSION


@pytest.fixture
def evaluation_inputs(tmp_path, monkeypatch):
    config_path = Path(__file__).resolve().parents[1] / "configs/environment.json"
    config = load_environment_config(config_path)
    checkpoint = tmp_path / "model" / "checkpoint.pt"
    checkpoint.parent.mkdir()
    metadata = CheckpointMetadata(environment_config_hash=config.config_hash,
                                  training_seed=42, hidden_layer_widths=(8,))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "split": "validation", "owner": "fixture", "environment_config": str(config_path),
        "config_hash": config.config_hash, "simulator_version": SIMULATOR_VERSION, "seeds": [1001],
    }), encoding="utf-8")
    monkeypatch.setattr("packfolio.evaluate.load_checkpoint_metadata", lambda _path: metadata)
    monkeypatch.setattr("packfolio.evaluate.load_trained_agent", lambda _path: CashOnlyPolicy())
    monkeypatch.setattr("packfolio.evaluate.importlib.metadata.version", lambda _name: "0.1.0")
    return config, checkpoint, split, tmp_path / "evaluation"


def training_manifest(config, checkpoint):
    return {
        "manifest_version": 1, "checkpoint_file": checkpoint.name,
        "environment_config_hash": config.config_hash, "training_seed": 42,
        "inventory_capacity": 10, "observation_reference_price": 10.0,
    }


@pytest.mark.parametrize("capacity", [1, 5, 11, 0, -1, True, 10.0])
def test_invalid_capacity_fails_before_loading_or_creating_outputs(tmp_path, monkeypatch, capacity):
    def unexpected_load(_path):
        pytest.fail("invalid capacity reached checkpoint loading")

    monkeypatch.setattr("packfolio.evaluate.load_checkpoint_metadata", unexpected_load)
    output = tmp_path / "evaluation"
    with pytest.raises((ValueError, TypeError), match="inventory_capacity"):
        evaluate_checkpoint_on_split(checkpoint_path=tmp_path / "missing.pt",
                                     split_manifest_path=tmp_path / "missing.json",
                                     output_directory=output, inventory_capacity=capacity)
    assert not output.exists()


@pytest.mark.parametrize("with_training_manifest", [False, True])
def test_settings_are_recorded_without_changing_existing_identities(evaluation_inputs, with_training_manifest):
    config, checkpoint, split, output = evaluation_inputs
    if with_training_manifest:
        (checkpoint.parent / "run_manifest.json").write_text(
            json.dumps(training_manifest(config, checkpoint)), encoding="utf-8"
        )
    evaluate_checkpoint_on_split(checkpoint_path=checkpoint, split_manifest_path=split, output_directory=output)
    settings = json.loads((output / "evaluation_settings.json").read_text(encoding="utf-8"))
    identity = {"environment_config_hash": config.config_hash,
                "inventory_capacity": 10, "observation_reference_price": 10.0}
    assert {name: settings[name] for name in identity} == identity
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert settings["settings_hash"] == hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert settings["training_settings_source"] == (
        "training_manifest" if with_training_manifest else "legacy_v1_fixed_settings"
    )
    assert settings["training_manifest_file"] == (
        str(checkpoint.parent / "run_manifest.json") if with_training_manifest else None
    )
    assert settings["settings_version"] == 1
    manifest = read_evaluation_manifest(output / settings["evaluation_manifest_file"])
    assert manifest.manifest_version == 1
    assert manifest.environment_config_hash == config.config_hash
    rows = read_evaluation_rows_jsonl(output / manifest.evaluation_rows_file)
    assert rows[0].config_hash == config.config_hash
    assert rows[0].scenario_id == f"{config.config_hash}:1001"
    assert rows[0].final_portfolio_value == config.initial_cash


@pytest.mark.parametrize("field, value", [
    ("inventory_capacity", 5), ("inventory_capacity", True),
    ("observation_reference_price", 20.0), ("observation_reference_price", True),
    ("environment_config_hash", "0" * 64), ("training_seed", 43),
    ("checkpoint_file", "other.pt"), ("manifest_version", 2), ("manifest_version", True),
])
def test_training_provenance_mismatches_fail_before_running(evaluation_inputs, monkeypatch, field, value):
    config, checkpoint, split, output = evaluation_inputs
    manifest = training_manifest(config, checkpoint)
    manifest[field] = value
    (checkpoint.parent / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def unexpected_policy_load(_path):
        pytest.fail("invalid provenance reached policy loading")

    monkeypatch.setattr("packfolio.evaluate.load_trained_agent", unexpected_policy_load)
    with pytest.raises((ValueError, TypeError), match=field):
        evaluate_checkpoint_on_split(checkpoint_path=checkpoint, split_manifest_path=split, output_directory=output)
    assert not output.exists()


def test_present_but_incomplete_training_manifest_is_not_treated_as_legacy(evaluation_inputs):
    _, checkpoint, split, output = evaluation_inputs
    (checkpoint.parent / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match="manifest_version"):
        evaluate_checkpoint_on_split(checkpoint_path=checkpoint, split_manifest_path=split, output_directory=output)
    assert not output.exists()


def test_configuration_mismatch_creates_no_output_directory(evaluation_inputs, monkeypatch):
    config, checkpoint, split, output = evaluation_inputs
    monkeypatch.setattr("packfolio.evaluate.load_checkpoint_metadata", lambda _path: CheckpointMetadata(
        environment_config_hash=replace(config, initial_cash=200.0).config_hash,
        training_seed=42, hidden_layer_widths=(8,),
    ))
    with pytest.raises(ValueError, match="environment_config_hash"):
        evaluate_checkpoint_on_split(checkpoint_path=checkpoint, split_manifest_path=split, output_directory=output)
    assert not output.exists()
