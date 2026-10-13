"""PF-22 workflow on temporary scenarios and handcrafted, untrained checkpoints."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest
import yaml

torch = pytest.importorskip("torch")

from packfolio import held_out
from packfolio.agents.dqn_agent import ACTION_ORDER, FEATURE_ORDER, FEATURE_SCALE, build_q_network
from packfolio.config import load_environment_config
from packfolio.evaluate import read_evaluation_rows_jsonl
from packfolio.train import load_agent_config
from packfolio.tune import FINAL_REPLICA_SEEDS


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("pf22-fixtures")
    shutil.copyfile(ROOT / "configs/env_frozen.yaml", root / "environment.yaml")
    shutil.copyfile(ROOT / "configs/nfl_pack.json", root / "nfl_pack.json")
    shutil.copyfile(ROOT / "configs/agent_selected.yaml", root / "selected.yaml")
    config = load_environment_config(root / "environment.yaml")
    write_json(root / "split.json", {"split": "final_test", "owner": "temporary fixture",
        "environment_config": "environment.yaml", "config_hash": config.config_hash,
        "simulator_version": "packfolio-scenarios-v1", "seeds": [9101, 9102]})
    selected = load_agent_config(root / "selected.yaml")
    model_root = root / "models"
    model_root.mkdir()
    replicas = []
    for seed in FINAL_REPLICA_SEEDS:
        directory = model_root / f"fixture-{seed}"
        directory.mkdir()
        agent = replace(selected, seed=seed)
        (directory / "agent_config.yaml").write_text(yaml.safe_dump(agent.to_dict()), encoding="utf-8")
        with torch.random.fork_rng(devices=[]):
            network = build_q_network(agent.hidden_layer_widths)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network[-1].bias[seed % 4] = 1
        torch.save({"format_version": 1,
            "architecture": {"observation_size": 8, "hidden_layer_widths": list(agent.hidden_layer_widths),
                "action_count": 4, "feature_order": list(FEATURE_ORDER), "action_order": list(ACTION_ORDER),
                "feature_scale": [float(v) for v in FEATURE_SCALE]},
            "q_network_state_dict": network.state_dict(), "training_seed": seed,
            "environment_config_hash": config.config_hash}, directory / "checkpoint.pt")
        write_json(directory / "run_manifest.json", {"manifest_version": 1,
            "status": "complete", "training_seed": seed, "actual_environment_steps": 2000,
            "environment_config_hash": config.config_hash, "checkpoint_file": "checkpoint.pt",
            "inventory_capacity": 10, "observation_reference_price": 10.0,
            "environment_episode_seeds": [8100 + seed % 10]})
        replicas.append({"seed": seed, "status": "success", "run_directory": directory.name})
    write_json(model_root / "final_replicas_manifest.json", {"final_replicas_version": 1,
        "final_replica_seeds": list(FINAL_REPLICA_SEEDS), "replicas": replicas})
    return root


def run(root, output):
    return held_out.run_held_out(split_manifest=root / "split.json",
        replica_manifest=root / "models/final_replicas_manifest.json",
        frozen_config=root / "environment.yaml", selected_agent_config=root / "selected.yaml",
        output_directory=output, bootstrap_resamples=100)


@pytest.fixture(scope="module")
def saved(tmp_path_factory, inputs):
    output = tmp_path_factory.mktemp("pf22-output") / "results"
    run(inputs, output)
    return output


@pytest.fixture
def evidence(tmp_path, saved):
    return Path(shutil.copytree(saved, tmp_path / "results"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_all_five_replicas_and_four_baselines_have_unique_paired_records(saved):
    episodes = csv_rows(saved / "episodes.csv")
    summaries = csv_rows(saved / "summary.csv")
    assert len(episodes) == 18 and len(summaries) == 9
    assert len([r for r in episodes if r["policy_id"] != "DQN"]) == 8
    assert {int(r["training_seed"]) for r in episodes if r["policy_id"] == "DQN"} == set(FINAL_REPLICA_SEEDS)
    assert {int(r["scenario_seed"]) for r in episodes} == {9101, 9102}
    assert all(r["episode_steps"] == "100" for r in episodes)
    for record in episodes:
        raw = next(r for r in read_evaluation_rows_jsonl(saved / record["episode_file"])
                   if r.scenario_id == record["scenario_id"])
        assert float(record["final_portfolio_value"]) == raw.final_portfolio_value
        if record["policy_id"] == "EV_ONE_STEP":
            assert record["information_class"] == "model-informed"
    for summary in summaries:
        assert summary["small_sample_warning"] == "True"
        assert summary["ci_sampling_unit"] == "scenario; fixed policy/model"
        if summary["policy_id"] == "DQN":
            assert all(summary[b + "_paired_mean_difference"] for b in held_out.BASELINES)
            model_values = {r["scenario_id"]: float(r["final_portfolio_value"]) for r in episodes
                            if r["model_id"] == summary["model_id"]}
            for baseline in held_out.BASELINES:
                baseline_values = {r["scenario_id"]: float(r["final_portfolio_value"]) for r in episodes
                                   if r["policy_id"] == baseline}
                expected = sum(model_values[s] - baseline_values[s] for s in model_values) / 2
                assert float(summary[baseline + "_paired_mean_difference"]) == pytest.approx(expected)
    uncertainty = read_json(saved / "uncertainty.json")
    assert uncertainty["training_uncertainty_included"] is False
    assert uncertainty["scenario_count"] == 2


def test_each_baseline_and_replica_runs_exactly_once_per_scenario(tmp_path, inputs, monkeypatch):
    from collections import Counter
    import packfolio.evaluate
    calls = []
    original = packfolio.evaluate.evaluate_episode
    def spy(policy, env, metadata):
        calls.append((metadata.policy_id, metadata.model_id, metadata.scenario_id))
        return original(policy, env, metadata)
    monkeypatch.setattr(packfolio.evaluate, "evaluate_episode", spy)
    monkeypatch.setattr(held_out, "evaluate_episode", spy)
    checkpoints = list((inputs / "models").rglob("checkpoint.pt"))
    before = {str(p): held_out._digest(p) for p in checkpoints}
    run(inputs, tmp_path / "results")
    assert len(calls) == 18 and set(Counter(calls).values()) == {1}
    for baseline in held_out.BASELINES:
        assert sum(policy == baseline for policy, _, _ in calls) == 2
    assert {str(p): held_out._digest(p) for p in checkpoints} == before


def test_saved_aggregation_does_not_evaluate_policies_and_ignores_input_order(evidence, monkeypatch):
    expected = (evidence / "summary.csv").read_bytes()
    expected_episodes = (evidence / "episodes.csv").read_bytes()
    manifest = read_json(evidence / "held_out_manifest.json")
    for group in manifest["groups"]:
        path = evidence / group["rows_file"]
        path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n", encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("aggregation must not run a policy or environment")
    monkeypatch.setattr(held_out, "evaluate_episode", forbidden)
    monkeypatch.setattr(held_out, "evaluate_checkpoint_on_split", forbidden)
    monkeypatch.setattr(held_out, "build_environment", forbidden)
    held_out.aggregate_results(evidence)
    assert (evidence / "summary.csv").read_bytes() == expected
    assert (evidence / "episodes.csv").read_bytes() == expected_episodes


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "provenance", "infeasible"])
def test_missing_duplicate_or_incompatible_episode_records_fail(evidence, mutation):
    manifest = read_json(evidence / "held_out_manifest.json")
    path = evidence / manifest["groups"][0]["rows_file"]
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "provenance":
        records[0]["training_seed"] += 1
    else:
        records[0]["infeasible_action_count"] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(ValueError):
        held_out.aggregate_results(evidence)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "capacity", "identity"])
def test_saved_manifest_requires_all_five_models_and_matching_settings(evidence, mutation):
    path = evidence / "held_out_manifest.json"
    manifest = read_json(path)
    if mutation == "missing":
        manifest["groups"].pop()
    elif mutation == "duplicate":
        manifest["groups"].append(manifest["groups"][0])
    elif mutation == "capacity":
        manifest["inventory_capacity"] = 11
    else:
        manifest["models"].pop()
    write_json(path, manifest)
    with pytest.raises(ValueError):
        held_out.aggregate_results(evidence)


def test_corrupt_cash_flow_is_rejected(evidence):
    manifest = read_json(evidence / "held_out_manifest.json")
    path = evidence / next(iter(manifest["groups"][0]["trace_files"].values()))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["cash_after"] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    with pytest.raises(ValueError, match="cash_after"):
        held_out.aggregate_results(evidence)


def test_ev_information_class_cannot_be_lost(evidence):
    manifest = read_json(evidence / "held_out_manifest.json")
    group = next(g for g in manifest["groups"] if g["policy_id"] == "EV_ONE_STEP")
    path = evidence / next(iter(group["metadata_files"].values()))
    metadata = read_json(path)
    metadata["information_class"] = "uninformed"
    write_json(path, metadata)
    with pytest.raises(ValueError, match="model-informed"):
        held_out.aggregate_results(evidence)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "incompatible", "training_overlap"])
def test_invalid_checkpoint_bundle_fails_before_any_output_or_evaluation(tmp_path, inputs, monkeypatch, mutation):
    root = Path(shutil.copytree(inputs, tmp_path / "inputs"))
    manifest_path = root / "models/final_replicas_manifest.json"
    manifest = read_json(manifest_path)
    if mutation == "missing":
        (root / "models" / manifest["replicas"][0]["run_directory"] / "checkpoint.pt").unlink()
    elif mutation == "duplicate":
        manifest["replicas"][1]["seed"] = manifest["replicas"][0]["seed"]
        write_json(manifest_path, manifest)
    else:
        path = root / "models" / manifest["replicas"][0]["run_directory"] / "run_manifest.json"
        data = read_json(path)
        if mutation == "incompatible":
            data["inventory_capacity"] = 11
        else:
            data["environment_episode_seeds"] = [9101]
        write_json(path, data)
    def forbidden(*args, **kwargs):
        pytest.fail("preflight failure reached evaluation")
    monkeypatch.setattr(held_out, "evaluate_checkpoint_on_split", forbidden)
    output = tmp_path / "results"
    with pytest.raises((ValueError, FileNotFoundError)):
        run(root, output)
    assert not output.exists()


def test_inputs_changed_after_evaluation_are_rejected(tmp_path, evidence):
    manifest_path = evidence / "held_out_manifest.json"
    manifest = read_json(manifest_path)
    altered = tmp_path / "changed.json"
    altered.write_text("{}", encoding="utf-8")
    manifest["source_sha256"] = {str(altered): "0" * 64}
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="artifact changed"):
        held_out.aggregate_results(evidence)


def test_training_or_validation_split_is_not_accepted(tmp_path, inputs):
    root = Path(shutil.copytree(inputs, tmp_path / "inputs"))
    split = read_json(root / "split.json")
    split["split"] = "validation"
    write_json(root / "split.json", split)
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="final_test"):
        run(root, output)
    assert not output.exists()


def test_existing_results_are_not_overwritten(tmp_path, inputs):
    output = tmp_path / "results"
    output.mkdir()
    sentinel = output / "existing.csv"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        run(inputs, output)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
