"""Plot calculations and rendering on synthetic, temporary evaluation fixtures."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

import matplotlib.image as mpimg
import numpy as np
import pytest

from packfolio import plots
from test_fee_sensitivity import fee_smoke, inputs, saved


@pytest.fixture
def rows(saved):
    manifest = plots._json(saved / "held_out_manifest.json")
    group = next(g for g in manifest["groups"] if g["policy_id"] == "CASH_ONLY")
    row = plots.read_evaluation_rows_jsonl(saved / group["rows_file"])[0]
    return [replace(row, scenario_seed=seed, scenario_id=f"{row.config_hash}:{seed}",
                    final_portfolio_value=value) for seed, value in [(1, 80), (2, 100), (3, 140)]]


def test_metrics_use_pf22_definitions_and_paired_scenarios(rows):
    rules = {"bootstrap_seed": 22, "bootstrap_resamples": 100,
             "confidence": .95, "loss_tolerance": 1e-9}
    actual = plots.metrics(rows, rules, rows)
    assert actual["mean_final_value"] == pytest.approx(320/3)
    assert actual["median_final_value"] == 100
    assert actual["sample_std_final_value"] == pytest.approx(np.std([80, 100, 140], ddof=1))
    assert actual["loss_probability"] == pytest.approx(1/3)
    assert actual["fifth_percentile_final_value"] == 82
    assert actual["executed_hold_rate"] == 1
    assert actual["paired_change"] == actual["paired_low"] == actual["paired_high"] == 0
    assert actual == plots.metrics(list(reversed(rows)), rules, rows)
    with pytest.raises(ValueError, match="paired"):
        plots.metrics(rows, rules, rows[:-1])


def test_episode_selection_is_median_based_and_seed_tied(rows):
    assert plots.select_episode(rows).scenario_seed == 2
    tied = [replace(rows[0], scenario_seed=9), replace(rows[-1], scenario_seed=4)]
    assert plots.select_episode(tied).scenario_seed == 4
    assert plots.select_episode(tied[::-1]).scenario_seed == 4
    with pytest.raises(ValueError): plots.select_episode([])


@pytest.fixture
def evidence(tmp_path, fee_smoke, saved, inputs, monkeypatch):
    # Reuse the existing two-scenario fixtures; never touch real experiments.
    root = tmp_path / "repository"
    shutil.copytree(saved, root / "results")
    fee_root, summary = fee_smoke
    shutil.copytree(fee_root / "evidence", root / "outputs/corrected")
    shutil.copyfile(summary, root / "results/fee_sensitivity.csv")
    shutil.copytree(inputs / "models", root / "runs/pf21-final-replicas")
    (root / "archive.zip").write_bytes(b"synthetic archive fixture")
    manifest = plots._json(saved / "held_out_manifest.json")
    publication = {
        "official_csv": {"path": "results/fee_sensitivity.csv", "sha256": plots._digest(summary)},
        "corrected_run": {"directory": "outputs/corrected",
            "manifest_sha256": plots._digest(root / "outputs/corrected/manifest.json"),
            "episodes_sha256": plots._digest(root / "outputs/corrected/episodes.jsonl.gz")},
        "evidence_archive": {"path": "archive.zip", "sha256": plots._digest(root / "archive.zip")},
        "checkpoint_sha256": {m["model_id"]: plots._digest(m["checkpoint"]) for m in manifest["models"]},
    }
    (root / "results/fee_sensitivity_manifest.json").write_text(json.dumps(publication))
    history = [{"episode": i, "episode_steps": 100, "completed": True,
                "episode_reward": i/20, "mean_loss": .1/i} for i in range(1, 21)]
    monkeypatch.setattr(plots, "_history", lambda directory: history)
    return root


def test_six_nonblank_figures_and_deterministic_rendering(evidence, tmp_path):
    before = {str(p): plots._digest(p) for p in evidence.rglob("*") if p.is_file()}
    first = plots.generate_figures(evidence, tmp_path / "first")
    second = plots.generate_figures(evidence, tmp_path / "second")
    assert len(first) == 6
    assert {p.stem for p in first} == set(plots.FIGURES)
    for a, b in zip(first, second):
        pixels = mpimg.imread(a)
        assert pixels.shape[0] >= 500 and pixels.shape[1] >= 1000
        assert pixels[:, :, :3].std() > .05
        assert a.read_bytes() == b.read_bytes()
    assert all(plots._digest(p) == digest for p, digest in before.items())
    with pytest.raises(ValueError, match="exists"):
        plots.generate_figures(evidence, tmp_path / "first")


def test_hash_failure_prevents_output(evidence, tmp_path):
    with (evidence / "results/fee_sensitivity.csv").open("a") as stream:
        stream.write("corruption")
    output = tmp_path / "figures"
    with pytest.raises(ValueError, match="hash"):
        plots.generate_figures(evidence, output)
    assert not output.exists()


def test_duplicate_reference_record_is_rejected(evidence):
    manifest = plots._json(evidence / "results/held_out_manifest.json")
    path = evidence / "results" / manifest["groups"][0]["rows_file"]
    with path.open("a") as stream: stream.write(path.read_text().splitlines()[0]+"\n")
    with pytest.raises(ValueError, match="duplicate"):
        plots.load_data(evidence)


def test_training_history_requires_committed_complete_records(tmp_path, monkeypatch):
    directory = tmp_path / "model"
    directory.mkdir()
    rows = [{"episode": i, "episode_steps": 100, "completed": True,
             "episode_reward": .1, "mean_loss": .01} for i in range(1, 21)]
    original = "\n".join(json.dumps(r) for r in rows).encode()
    path = directory / "training_metrics.jsonl"
    path.write_bytes(original)
    monkeypatch.setattr(plots, "ROOT", tmp_path)
    monkeypatch.setattr(plots.subprocess, "check_output", lambda *args, **kwargs: original)
    assert len(plots._history(directory)) == 20
    path.write_bytes(original.replace(b"\n", b"\r\n"))
    assert len(plots._history(directory)) == 20
    path.write_bytes(original+b"\n{}")
    with pytest.raises(ValueError, match="committed"):
        plots._history(directory)
