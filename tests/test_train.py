import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from packfolio.agents.dqn_agent import load_trained_agent
from packfolio.config import load_environment_config
from packfolio.train import AgentConfig, load_agent_config, train


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_CONFIG = REPOSITORY_ROOT / "configs" / "environment.json"


def write_agent_config(path: Path, **overrides: object) -> None:
    values: dict[str, object] = {
        "seed": 42,
        "total_steps": 8,
        "hidden_layer_widths": [8],
        "learning_rate": 0.001,
        "gamma": 1.0,
        "replay_capacity": 16,
        "batch_size": 2,
        "learning_starts": 2,
        "target_update_interval": 2,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 8,
        "gradient_clip_norm": 10.0,
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def test_agent_config_is_separate_and_requires_gamma_one(tmp_path):
    config_path = tmp_path / "agent.yaml"
    write_agent_config(config_path)

    config = load_agent_config(config_path)

    assert config.gamma == 1.0
    assert config.hidden_layer_widths == (8,)
    assert config.total_steps == 8
    with pytest.raises(ValueError, match="gamma must be 1.0"):
        AgentConfig(
            seed=1,
            total_steps=1,
            hidden_layer_widths=(8,),
            learning_rate=0.001,
            gamma=0.99,
            replay_capacity=2,
            batch_size=1,
            learning_starts=1,
            target_update_interval=1,
            epsilon_start=1.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            gradient_clip_norm=1.0,
        )


def test_agent_config_rejects_missing_and_extra_fields(tmp_path):
    config_path = tmp_path / "agent.yaml"
    write_agent_config(config_path, unexpected=3)
    with pytest.raises(ValueError, match="extra fields: unexpected"):
        load_agent_config(config_path)

    write_agent_config(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values.pop("seed")
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields: seed"):
        load_agent_config(config_path)


def test_short_training_run_logs_metadata_and_saves_reloadable_checkpoint(tmp_path):
    pytest.importorskip("torch")
    env_config = load_environment_config(ENVIRONMENT_CONFIG)
    agent_config = tmp_path / "agent.yaml"
    write_agent_config(agent_config)

    run_directory = train(ENVIRONMENT_CONFIG, agent_config, tmp_path / "runs")

    assert run_directory.is_dir()
    assert (run_directory / "checkpoint.pt").is_file()
    assert (run_directory / "environment_config.json").read_bytes() == (
        ENVIRONMENT_CONFIG.read_bytes()
    )
    assert (run_directory / "agent_config.yaml").read_bytes() == agent_config.read_bytes()

    with (run_directory / "run_manifest.json").open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    assert manifest["status"] == "complete"
    assert manifest["actual_environment_steps"] == 8
    assert manifest["environment_config_hash"] == env_config.config_hash
    assert manifest["training_seed"] == 42
    assert manifest["git_commit"]
    assert manifest["runtime_seconds"] >= 0.0
    assert manifest["episodes"] == 1
    assert len(manifest["environment_episode_seeds"]) == 1
    assert manifest["dependencies"]
    dependency_names = [package["name"].casefold() for package in manifest["dependencies"]]
    assert len(dependency_names) == len(set(dependency_names))
    assert {"exploration", "environment_seed_generator", "torch", "replay_sampling"} <= set(
        manifest["derived_seeds"]
    )

    metrics_path = run_directory / manifest["metrics_file"]
    metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["episode_steps"] == 8
    assert metrics[0]["completed"] is False
    assert metrics[0]["termination"] == "STEP_BUDGET"
    for key in ("episode_reward", "final_portfolio_value", "invalid_action_rate", "mean_loss"):
        assert np.isfinite(metrics[0][key])
    assert 0.0 <= metrics[0]["invalid_action_rate"] <= 1.0
    assert metrics[0]["gradient_updates"] > 0

    observation = np.asarray([1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    action = load_trained_agent(run_directory / "checkpoint.pt").choose_action(observation)
    assert int(action) in range(4)


def test_training_runs_have_unique_directories_and_independent_agent_settings(tmp_path):
    pytest.importorskip("torch")
    original_hash = load_environment_config(ENVIRONMENT_CONFIG).config_hash
    first_config = tmp_path / "agent-one.yaml"
    second_config = tmp_path / "agent-two.yaml"
    write_agent_config(first_config)
    write_agent_config(second_config, learning_rate=0.002, seed=43)

    first = train(ENVIRONMENT_CONFIG, first_config, tmp_path / "runs")
    second = train(ENVIRONMENT_CONFIG, second_config, tmp_path / "runs")

    assert first != second
    assert first.parent == second.parent
    manifests = []
    for run_directory in (first, second):
        with (run_directory / "run_manifest.json").open(encoding="utf-8") as manifest_file:
            manifests.append(json.load(manifest_file))
    assert manifests[0]["environment_config_hash"] == original_hash
    assert manifests[1]["environment_config_hash"] == original_hash
    assert manifests[0]["training_seed"] != manifests[1]["training_seed"]
    assert manifests[0]["run_id"] != manifests[1]["run_id"]


def test_identical_training_seeds_reproduce_metrics_and_weights(tmp_path):
    torch = pytest.importorskip("torch")
    agent_config = tmp_path / "agent.yaml"
    write_agent_config(agent_config, total_steps=5)

    first = train(ENVIRONMENT_CONFIG, agent_config, tmp_path / "runs")
    second = train(ENVIRONMENT_CONFIG, agent_config, tmp_path / "runs")

    first_metrics = (first / "training_metrics.jsonl").read_text(encoding="utf-8")
    second_metrics = (second / "training_metrics.jsonl").read_text(encoding="utf-8")
    assert first_metrics == second_metrics
    first_checkpoint = torch.load(first / "checkpoint.pt", weights_only=True)
    second_checkpoint = torch.load(second / "checkpoint.pt", weights_only=True)
    assert first_checkpoint["actual_environment_steps"] == 5
    for name, weights in first_checkpoint["q_network_state_dict"].items():
        assert torch.equal(weights, second_checkpoint["q_network_state_dict"][name])
