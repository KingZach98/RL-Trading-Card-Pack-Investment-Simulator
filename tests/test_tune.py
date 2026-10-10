"""PF-21: compare predetermined learning-rate/exploration candidates and train
the final independent replicas of the selected configuration.

These tests use a tiny training budget and a tiny environment so the full
orchestration (candidate grid, seed loop, validation-only selection, failure
recording, and retaining every final replica) runs in a fixed, small smoke
budget rather than real training time. They are integration coverage for
``packfolio.tune``, not a substitute for the real PF-21 tuning run recorded in
``docs/hyperparameter_tuning.md``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from packfolio import tune
from packfolio.scenarios import load_split_manifest

from tests.test_evaluate_reload import write_split_manifest


def _tiny_baseline_agent_config(path: Path, **overrides: object) -> None:
    import yaml

    values: dict[str, object] = {
        "seed": 0,
        "total_steps": 24,
        "hidden_layer_widths": [4],
        "learning_rate": 0.01,
        "gamma": 1.0,
        "replay_capacity": 32,
        "batch_size": 4,
        "learning_starts": 4,
        "target_update_interval": 8,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay_steps": 12,
        "gradient_clip_norm": 10.0,
    }
    values.update(overrides)
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


@pytest.fixture
def tiny_fixture(tmp_path):
    agent_config_path = tmp_path / "agent.yaml"
    _tiny_baseline_agent_config(agent_config_path)
    validation_split_path = write_split_manifest(
        tmp_path / "splits", seeds=[9001, 9002], split="validation"
    )
    environment_config_path = tmp_path / "splits" / "environment.json"
    return {
        "agent_config_path": agent_config_path,
        "validation_split_path": validation_split_path,
        "environment_config_path": environment_config_path,
    }


def test_run_tuning_trains_and_selects_from_validation_only(tmp_path, tiny_fixture):
    candidates = (
        tune.Candidate("baseline", {}),
        tune.Candidate("high_learning_rate", {"learning_rate": 0.05}),
    )
    seeds = (1, 2)
    summary = tune.run_tuning(
        environment_config_path=tiny_fixture["environment_config_path"],
        baseline_agent_config_path=tiny_fixture["agent_config_path"],
        validation_split_path=tiny_fixture["validation_split_path"],
        candidates=candidates,
        seeds=seeds,
        output_root=tmp_path / "tuning",
    )

    assert len(summary["results"]) == len(candidates) * len(seeds)
    assert all(record["status"] == "success" for record in summary["results"])
    assert set(summary["candidate_mean_validation_score"]) == {"baseline", "high_learning_rate"}
    assert summary["selected_candidate"] in {"baseline", "high_learning_rate"}
    assert summary["validation_split_path"] == str(tiny_fixture["validation_split_path"].resolve())
    assert (tmp_path / "tuning" / "tuning_results.json").exists()

    # Every candidate/seed pair actually produced an independent checkpoint.
    run_directories = {record["run_directory"] for record in summary["results"]}
    assert len(run_directories) == len(summary["results"])


def test_run_tuning_records_failures_without_crashing(tmp_path, tiny_fixture):
    candidates = (
        tune.Candidate("baseline", {}),
        tune.Candidate("broken", {"epsilon_decay_steps": -1}),
    )
    summary = tune.run_tuning(
        environment_config_path=tiny_fixture["environment_config_path"],
        baseline_agent_config_path=tiny_fixture["agent_config_path"],
        validation_split_path=tiny_fixture["validation_split_path"],
        candidates=candidates,
        seeds=(1,),
        output_root=tmp_path / "tuning",
    )

    statuses = {record["candidate"]: record["status"] for record in summary["results"]}
    assert statuses == {"baseline": "success", "broken": "failed"}
    failed_record = next(r for r in summary["results"] if r["candidate"] == "broken")
    assert "error" in failed_record and "traceback" in failed_record
    # A candidate with zero successful seeds must never be selected.
    assert summary["selected_candidate"] == "baseline"
    assert "broken" not in summary["candidate_mean_validation_score"]


def test_run_tuning_rejects_total_steps_override(tmp_path, tiny_fixture):
    candidates = (tune.Candidate("bigger_budget", {"total_steps": 999}),)
    with pytest.raises(ValueError, match="total_steps"):
        tune.run_tuning(
            environment_config_path=tiny_fixture["environment_config_path"],
            baseline_agent_config_path=tiny_fixture["agent_config_path"],
            validation_split_path=tiny_fixture["validation_split_path"],
            candidates=candidates,
            seeds=(1,),
            output_root=tmp_path / "tuning",
        )


def test_run_tuning_rejects_duplicate_candidate_names(tmp_path, tiny_fixture):
    candidates = (tune.Candidate("dup", {}), tune.Candidate("dup", {"learning_rate": 0.02}))
    with pytest.raises(ValueError, match="unique"):
        tune.run_tuning(
            environment_config_path=tiny_fixture["environment_config_path"],
            baseline_agent_config_path=tiny_fixture["agent_config_path"],
            validation_split_path=tiny_fixture["validation_split_path"],
            candidates=candidates,
            seeds=(1,),
            output_root=tmp_path / "tuning",
        )


def test_run_tuning_refuses_a_non_validation_split(tmp_path, tiny_fixture):
    training_split_path = write_split_manifest(
        tmp_path / "splits", seeds=[1001], split="training"
    )
    with pytest.raises(ValueError, match="validation"):
        tune.run_tuning(
            environment_config_path=tiny_fixture["environment_config_path"],
            baseline_agent_config_path=tiny_fixture["agent_config_path"],
            validation_split_path=training_split_path,
            candidates=(tune.Candidate("baseline", {}),),
            seeds=(1,),
            output_root=tmp_path / "tuning",
        )


def test_train_final_replicas_retains_every_seed_including_failures(tmp_path, tiny_fixture, monkeypatch):
    summary = tune.run_tuning(
        environment_config_path=tiny_fixture["environment_config_path"],
        baseline_agent_config_path=tiny_fixture["agent_config_path"],
        validation_split_path=tiny_fixture["validation_split_path"],
        candidates=(tune.Candidate("baseline", {}),),
        seeds=(1,),
        output_root=tmp_path / "tuning",
    )

    real_train = tune.train

    def flaky_train(environment_config_path, agent_config_path, output_root):
        # Fail for exactly one of the three replica seeds to prove failures
        # are recorded rather than silently dropped from the manifest.
        import yaml

        with open(agent_config_path, encoding="utf-8") as handle:
            if yaml.safe_load(handle)["seed"] == 102:
                raise RuntimeError("simulated training failure")
        return real_train(environment_config_path, agent_config_path, output_root)

    monkeypatch.setattr(tune, "train", flaky_train)

    manifest = tune.train_final_replicas(
        tuning_summary=summary,
        environment_config_path=tiny_fixture["environment_config_path"],
        baseline_agent_config_path=tiny_fixture["agent_config_path"],
        seeds=(100, 101, 102),
        output_root=tmp_path / "final-replicas",
    )

    assert [r["seed"] for r in manifest["replicas"]] == [100, 101, 102]
    assert [r["status"] for r in manifest["replicas"]] == ["success", "success", "failed"]
    assert manifest["selected_candidate"] == "baseline"
    assert (tmp_path / "final-replicas" / "final_replicas_manifest.json").exists()


def test_tuning_seeds_and_final_replica_seeds_are_disjoint():
    assert not set(tune.TUNING_SEEDS) & set(tune.FINAL_REPLICA_SEEDS)
