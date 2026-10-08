"""Saved-file-backed five-strategy PF-19 integration and corruption checks."""

import csv
import json
from pathlib import Path
import shutil

import pytest
import yaml

pytest.importorskip("torch")

from packfolio import compare
from packfolio.evaluate import read_evaluation_rows_jsonl, read_step_traces_jsonl
from packfolio.train import train


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "configs/splits/validation.json"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pf19-training")
    agent = yaml.safe_load((ROOT / "configs/agent.yaml").read_text(encoding="utf-8"))
    agent.update(total_steps=8, hidden_layer_widths=[8], batch_size=2,
                 learning_starts=2, replay_capacity=16, target_update_interval=2)
    config = directory / "agent.yaml"
    config.write_text(yaml.safe_dump(agent), encoding="utf-8")
    return train(ROOT / "configs/environment.json", config, directory / "runs") / "checkpoint.pt"


@pytest.fixture(scope="module")
def saved_comparison(tmp_path_factory, checkpoint):
    directory = tmp_path_factory.mktemp("pf19-evidence")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(compare, "OUTPUT_ROOT", directory)
        table = compare.run_comparison(split_manifest=VALIDATION,
            output_directory=directory / "comparison", checkpoint=checkpoint)
    return table.parent


@pytest.fixture
def evidence(tmp_path, saved_comparison):
    return Path(shutil.copytree(saved_comparison, tmp_path / "comparison"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def edit_record(path, field, value, *, index=0):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[index][field] = value
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_table_roundtrips_all_fifteen_saved_episodes_and_full_traces(saved_comparison):
    with (saved_comparison / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 15
    assert {record["policy_id"] for record in records} == set(compare.POLICY_IDS)
    assert {int(record["scenario_seed"]) for record in records} == {2001, 2002, 2003}
    for record in records:
        rows = read_evaluation_rows_jsonl(saved_comparison / record["episode_file"])
        row = next(row for row in rows if row.scenario_id == record["scenario_id"])
        steps = read_step_traces_jsonl(saved_comparison / record["trace_file"])
        assert len(steps) == int(record["episode_steps"]) == 100
        assert float(record["final_portfolio_value"]) == row.final_portfolio_value == steps[-1].cash_after
        assert steps[-1].sealed_count_after == 0
        assert int(record["infeasible_action_count"]) == sum(step.action_was_infeasible for step in steps)
        assert sum(int(record[f"requested_{action}_count"]) for action in
                   ("hold", "buy_pack", "open_and_sell", "sell_pack")) == 100
        assert sum(int(record[f"executed_{action}_count"]) for action in
                   ("hold", "buy_pack", "open_and_sell", "sell_pack")) == 100
        if record["policy_id"] == "CASH_ONLY":
            assert row.final_portfolio_value == 100
        if record["policy_id"] == "EV_ONE_STEP":
            assert record["information_class"] == "model-informed"
        if record["policy_id"] == "DQN":
            assert record["training_seed"] and record["model_id"]
        else:
            assert not record["training_seed"] and not record["model_id"]


def test_table_is_regenerated_only_from_saved_files(evidence, monkeypatch):
    original = (evidence / "comparison.csv").read_bytes()
    (evidence / "comparison.csv").unlink()
    def forbidden(*args, **kwargs):
        pytest.fail("report generation must not run a policy")
    monkeypatch.setattr(compare, "evaluate_episode", forbidden)
    monkeypatch.setattr(compare, "evaluate_ev_episode", forbidden)
    monkeypatch.setattr(compare, "evaluate_checkpoint_on_split", forbidden)
    assert compare.build_comparison(evidence).read_bytes() == original


def test_repeat_evaluation_reloads_checkpoint_and_reproduces_table(tmp_path, monkeypatch,
                                                                 checkpoint, saved_comparison):
    monkeypatch.setattr(compare, "OUTPUT_ROOT", tmp_path)
    table = compare.run_comparison(split_manifest=VALIDATION,
        output_directory=tmp_path / "repeat", checkpoint=checkpoint)
    assert table.read_bytes() == (saved_comparison / "comparison.csv").read_bytes()


@pytest.mark.parametrize("field,value", [
    ("cash_after", 999.0), ("fee_paid", 99.0),
    ("portfolio_value_after", 999.0), ("regime_after", "LOW"),
    ("step_index", 12), ("reward", 42.0),
])
def test_saved_trace_corruption_is_rejected(evidence, field, value):
    entry = read_json(evidence / "comparison_manifest.json")["episodes"][0]
    # Use a known different regime even if the first transition happens to be LOW.
    if field == "regime_after":
        current = read_step_traces_jsonl(evidence / entry["trace_file"])[0].regime_after.value
        value = "HIGH" if current != "HIGH" else "LOW"
    edit_record(evidence / entry["trace_file"], field, value)
    with pytest.raises(ValueError, match="replay"):
        compare.build_comparison(evidence)


@pytest.mark.parametrize("field,value", [
    ("final_portfolio_value", 999.0), ("cumulative_reward", 42.0),
    ("scenario_seed", 999), ("git_commit", "wrong"),
    ("model_id", "not-a-baseline-model"),
])
def test_episode_summary_corruption_is_rejected(evidence, field, value):
    entry = read_json(evidence / "comparison_manifest.json")["episodes"][0]
    edit_record(evidence / entry["rows_file"], field, value)
    with pytest.raises(ValueError):
        compare.build_comparison(evidence)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_complete_unique_pairing_is_required(evidence, change):
    path = evidence / "comparison_manifest.json"
    manifest = read_json(path)
    if change == "missing":
        manifest["episodes"].pop()
    else:
        manifest["episodes"].append(manifest["episodes"][0])
    write_json(path, manifest)
    with pytest.raises(ValueError, match="missing|duplicate"):
        compare.build_comparison(evidence)


def test_ev_classification_is_required(evidence):
    entries = read_json(evidence / "comparison_manifest.json")["episodes"]
    entry = next(entry for entry in entries if entry["policy_id"] == "EV_ONE_STEP")
    path = evidence / entry["policy_metadata_file"]
    metadata = read_json(path)
    metadata["information_class"] = "uninformed"
    write_json(path, metadata)
    with pytest.raises(ValueError, match="model-informed"):
        compare.build_comparison(evidence)


def test_requested_and_executed_counts_must_match_full_trace(evidence):
    entry = read_json(evidence / "comparison_manifest.json")["episodes"][0]
    path = evidence / entry["rows_file"]
    edit_record(path, "requested_hold_count", 99)
    edit_record(path, "requested_buy_pack_count", 1)
    edit_record(path, "executed_hold_count", 99)
    edit_record(path, "executed_buy_pack_count", 1)
    with pytest.raises(ValueError, match="action count"):
        compare.build_comparison(evidence)


def test_independent_accounting_rejects_a_matching_but_faulty_replay(evidence, monkeypatch):
    entry = read_json(evidence / "comparison_manifest.json")["episodes"][0]
    edit_record(evidence / entry["trace_file"], "fee_paid", 0.5)
    build = compare.build_environment
    def faulty_environment(config):
        env = build(config)
        original_step = env.step
        def faulty_step(action):
            observation, reward, terminated, truncated, info = original_step(action)
            if info["step_index"] == 0:
                info["fee_paid"] = 0.5
            return observation, reward, terminated, truncated, info
        env.step = faulty_step
        return env
    monkeypatch.setattr(compare, "build_environment", faulty_environment)
    with pytest.raises(ValueError, match="single sale fee"):
        compare.build_comparison(evidence)


def test_missing_step_cannot_reduce_the_action_limit(evidence):
    entry = read_json(evidence / "comparison_manifest.json")["episodes"][0]
    path = evidence / entry["trace_file"]
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="horizon/action limit"):
        compare.build_comparison(evidence)


def test_terminal_inventory_must_be_liquidated(evidence):
    entry = read_json(evidence / "comparison_manifest.json")["episodes"][0]
    edit_record(evidence / entry["trace_file"], "sealed_count_after", 1, index=99)
    with pytest.raises(ValueError, match="replay"):
        compare.build_comparison(evidence)


@pytest.mark.parametrize("field,value", [
    ("inventory_capacity", 11), ("reference_price", 1000),
    ("scenario_version", "unknown"), ("config_hash", "0" * 64),
    ("checkpoint_sha256", "0" * 64),
])
def test_changed_settings_or_checkpoint_identity_are_rejected(evidence, field, value):
    path = evidence / "comparison_manifest.json"
    manifest = read_json(path)
    manifest[field] = value
    write_json(path, manifest)
    with pytest.raises(ValueError, match="settings|checkpoint"):
        compare.build_comparison(evidence)


def test_incompatible_checkpoint_fails_before_output_creation(tmp_path, monkeypatch, checkpoint):
    import torch
    data = torch.load(checkpoint, weights_only=True, map_location="cpu")
    data["environment_config_hash"] = "0" * 64
    incompatible = tmp_path / "incompatible.pt"
    torch.save(data, incompatible)
    monkeypatch.setattr(compare, "OUTPUT_ROOT", tmp_path)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="configuration mismatch"):
        compare.run_comparison(split_manifest=VALIDATION, checkpoint=incompatible,
                               output_directory=output)
    assert not output.exists()


def test_actual_runtime_budget_includes_evaluation_and_replay(saved_comparison):
    budget = read_json(saved_comparison / "runtime_budget.json")
    assert "development" in budget["purpose"]
    assert budget["evaluation_environment_steps"] == 1500
    assert budget["validation_replay_steps"] == 1500
    assert budget["scenarios"] == 3 and budget["strategies"] == 5
    assert all(value > 0 for value in budget["timings_seconds"].values())


def test_pilot_branch_uses_unchanged_existing_training_configuration(tmp_path, monkeypatch, checkpoint):
    import packfolio.train
    calls = []
    def pilot(environment, agent, output):
        calls.append((environment.resolve(), agent, output))
        return checkpoint.parent
    monkeypatch.setattr(packfolio.train, "train", pilot)
    monkeypatch.setattr(compare, "OUTPUT_ROOT", tmp_path)
    table = compare.run_comparison(split_manifest=VALIDATION, output_directory=tmp_path / "pilot")
    assert len(calls) == 1
    assert calls[0][0] == ROOT / "configs/environment.json"
    assert calls[0][1] == ROOT / "configs/agent.yaml"
    assert calls[0][2].is_relative_to(table.parent)
    assert yaml.safe_load(calls[0][1].read_text(encoding="utf-8"))["total_steps"] == 2000
    assert read_json(table.parent / "comparison_manifest.json")["checkpoint_role"] == "development pilot"
