"""PF-18: verify a saved DQN checkpoint reloads and evaluates deterministically.

These tests train a tiny agent once, then prove that evaluating its saved
checkpoint through `evaluate_checkpoint_on_split` -- including from a genuinely
separate `python -m packfolio.evaluate` subprocess -- reproduces identical
action sequences and portfolio traces, and that the written evaluation
manifest links the model, its configuration, the commit, and the scenario
set. It also checks that incompatible configurations and checkpoints fail
clearly before any episode runs.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

pytest.importorskip("torch")

from packfolio.config import load_environment_config
from packfolio.evaluate import (
    evaluate_checkpoint_on_split,
    read_evaluation_manifest,
    read_evaluation_rows_jsonl,
    read_step_traces_jsonl,
)
from packfolio.train import train


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_CONFIG = REPOSITORY_ROOT / "configs" / "environment.json"
NFL_PACK_CONFIG = REPOSITORY_ROOT / "configs" / "nfl_pack.json"


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


def write_split_manifest(
    directory: Path,
    *,
    environment_config_path: Path = ENVIRONMENT_CONFIG,
    seeds: list[int],
    split: str = "validation",
    owner: str = "Test",
) -> Path:
    """Copy an environment config (with its pack file) and a split manifest into `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "environment.json").write_text(
        environment_config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (directory / "nfl_pack.json").write_text(
        NFL_PACK_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = load_environment_config(directory / "environment.json")
    split_path = directory / f"{split}.json"
    split_path.write_text(
        json.dumps(
            {
                "split": split,
                "owner": owner,
                "environment_config": "environment.json",
                "config_hash": config.config_hash,
                "simulator_version": "packfolio-scenarios-v1",
                "seeds": seeds,
            }
        ),
        encoding="utf-8",
    )
    return split_path


@pytest.fixture
def trained_checkpoint(tmp_path: Path) -> Path:
    agent_config_path = tmp_path / "agent.yaml"
    write_agent_config(agent_config_path)
    run_directory = train(ENVIRONMENT_CONFIG, agent_config_path, tmp_path / "runs")
    return run_directory / "checkpoint.pt"


def test_reload_reproduces_deterministic_actions_and_portfolio_traces(
    tmp_path, trained_checkpoint
):
    split_path = write_split_manifest(tmp_path / "split", seeds=[11, 12])

    first = evaluate_checkpoint_on_split(
        checkpoint_path=trained_checkpoint,
        split_manifest_path=split_path,
        output_directory=tmp_path / "eval_first",
    )
    second = evaluate_checkpoint_on_split(
        checkpoint_path=trained_checkpoint,
        split_manifest_path=split_path,
        output_directory=tmp_path / "eval_second",
    )

    first_rows = read_evaluation_rows_jsonl(first / "evaluation_rows.jsonl")
    second_rows = read_evaluation_rows_jsonl(second / "evaluation_rows.jsonl")
    assert first_rows == second_rows
    assert len(first_rows) == 2

    for row in first_rows:
        trace_name = row.scenario_id.replace(":", "_") + ".jsonl"
        first_trace = read_step_traces_jsonl(first / "step_traces" / trace_name)
        second_trace = read_step_traces_jsonl(second / "step_traces" / trace_name)
        assert first_trace == second_trace
        assert len(first_trace) == row.episode_steps
        assert [step.requested_action for step in first_trace] == [
            step.requested_action for step in second_trace
        ]
        assert [step.portfolio_value_after for step in first_trace] == [
            step.portfolio_value_after for step in second_trace
        ]


def test_a_fresh_subprocess_reproduces_the_in_process_evaluation(
    tmp_path, trained_checkpoint
):
    """A second member must be able to evaluate the saved model without retraining."""
    split_path = write_split_manifest(tmp_path / "split", seeds=[21])

    in_process = evaluate_checkpoint_on_split(
        checkpoint_path=trained_checkpoint,
        split_manifest_path=split_path,
        output_directory=tmp_path / "eval_in_process",
    )

    out_subprocess = tmp_path / "eval_subprocess"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "packfolio.evaluate",
            "--checkpoint",
            str(trained_checkpoint),
            "--split-manifest",
            str(split_path),
            "--output",
            str(out_subprocess),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    in_process_text = (in_process / "evaluation_rows.jsonl").read_text(encoding="utf-8")
    subprocess_text = (out_subprocess / "evaluation_rows.jsonl").read_text(encoding="utf-8")
    assert in_process_text == subprocess_text


def test_evaluation_manifest_links_model_configuration_commit_and_scenario_set(
    tmp_path, trained_checkpoint
):
    split_path = write_split_manifest(tmp_path / "split", seeds=[31, 32, 33])

    output_directory = evaluate_checkpoint_on_split(
        checkpoint_path=trained_checkpoint,
        split_manifest_path=split_path,
        output_directory=tmp_path / "eval",
    )
    manifest = read_evaluation_manifest(output_directory / "evaluation_manifest.json")

    config = load_environment_config(split_path.parent / "environment.json")
    assert manifest.checkpoint_file == str(trained_checkpoint.resolve())
    assert manifest.environment_config_hash == config.config_hash
    assert manifest.git_commit
    assert len(manifest.git_commit) == 40
    assert manifest.scenario_manifest_file == str(split_path.resolve())
    assert manifest.scenario_split == "validation"
    assert len(manifest.scenario_ids) == 3
    assert all(scenario_id.startswith(config.config_hash) for scenario_id in manifest.scenario_ids)

    rows = read_evaluation_rows_jsonl(output_directory / manifest.evaluation_rows_file)
    assert tuple(row.scenario_id for row in rows) == manifest.scenario_ids


def test_mismatched_environment_config_fails_before_running_any_episode(
    tmp_path, trained_checkpoint
):
    mismatched_environment = json.loads(ENVIRONMENT_CONFIG.read_text(encoding="utf-8"))
    mismatched_environment["initial_cash"] = 999.0
    mismatched_directory = tmp_path / "mismatched"
    mismatched_directory.mkdir()
    (mismatched_directory / "environment.json").write_text(
        json.dumps(mismatched_environment), encoding="utf-8"
    )
    split_path = write_split_manifest(
        mismatched_directory,
        environment_config_path=mismatched_directory / "environment.json",
        seeds=[41],
    )

    output_directory = tmp_path / "eval_mismatch"
    with pytest.raises(ValueError, match="environment_config_hash"):
        evaluate_checkpoint_on_split(
            checkpoint_path=trained_checkpoint,
            split_manifest_path=split_path,
            output_directory=output_directory,
        )

    assert not (output_directory / "evaluation_rows.jsonl").exists()
    assert not (output_directory / "evaluation_manifest.json").exists()


def test_incompatible_checkpoint_architecture_fails_clearly(tmp_path, trained_checkpoint):
    torch = pytest.importorskip("torch")
    checkpoint = torch.load(trained_checkpoint, map_location="cpu", weights_only=True)
    checkpoint["architecture"]["feature_order"] = ("WRONG_ORDER",)
    corrupted_checkpoint = tmp_path / "corrupted_checkpoint.pt"
    torch.save(checkpoint, corrupted_checkpoint)

    split_path = write_split_manifest(tmp_path / "split", seeds=[51])

    with pytest.raises(ValueError, match="feature order"):
        evaluate_checkpoint_on_split(
            checkpoint_path=corrupted_checkpoint,
            split_manifest_path=split_path,
            output_directory=tmp_path / "eval_corrupted",
        )
