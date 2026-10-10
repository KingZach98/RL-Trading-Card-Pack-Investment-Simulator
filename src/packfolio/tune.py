"""PF-21: compare predetermined learning-rate/exploration candidates under an
equal training-step budget, select by a validation-only criterion fixed
before running, and train the final independent model replicas.

See :mod:`packfolio.train` for the underlying DQN trainer and
:mod:`packfolio.evaluate` for checkpoint reload/evaluation. This module only
orchestrates repeated calls to both, plus selection and bookkeeping; it does
not change training, environment, or evaluation semantics.

Read ``docs/hyperparameter_tuning.md`` for the full write-up (candidates,
seeds, selection rule, results, and the selected configuration).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import traceback
from collections.abc import Mapping, Sequence

import yaml

from packfolio.evaluate import evaluate_checkpoint_on_split, read_evaluation_rows_jsonl
from packfolio.scenarios import load_split_manifest
from packfolio.train import AgentConfig, load_agent_config, train

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ENVIRONMENT_CONFIG = _REPOSITORY_ROOT / "configs" / "environment.json"
_DEFAULT_BASELINE_AGENT_CONFIG = _REPOSITORY_ROOT / "configs" / "agent.yaml"
_DEFAULT_VALIDATION_SPLIT = _REPOSITORY_ROOT / "configs" / "splits" / "validation.json"
_DEFAULT_TUNING_OUTPUT_ROOT = _REPOSITORY_ROOT / "runs" / "pf21-tuning"
_DEFAULT_FINAL_REPLICA_OUTPUT_ROOT = _REPOSITORY_ROOT / "runs" / "pf21-final-replicas"

SELECTION_METRIC = "final_portfolio_value"
SELECTION_RULE = (
    "Among candidates with at least one successful training seed, select the "
    "candidate whose mean-across-seeds value of "
    "(mean final_portfolio_value over the validation split's scenarios) is "
    "highest. Ties are broken by candidate declaration order in "
    "TUNING_CANDIDATES. This rule, the candidate grid, and both seed lists "
    "are fixed before any run; see docs/hyperparameter_tuning.md."
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One predetermined learning-rate/exploration setting to compare.

    ``overrides`` are field replacements applied to the shared baseline
    :class:`~packfolio.train.AgentConfig` (``configs/agent.yaml`` by
    default). An empty mapping reproduces the baseline unchanged.
    """

    name: str
    overrides: Mapping[str, object]


# Predetermined candidate grid: compares learning rate (candidates 2-3 against
# the baseline) and exploration decay speed (candidates 4-5 against the
# baseline) one axis at a time, holding every other hyperparameter, the
# training budget, and the training seeds fixed. Do not add, remove, or
# retune candidates after seeing validation results; that would defeat the
# purpose of a predetermined comparison (see PF-20's freeze rationale).
TUNING_CANDIDATES: tuple[Candidate, ...] = (
    Candidate("baseline", {}),
    Candidate("low_learning_rate", {"learning_rate": 0.0003}),
    Candidate("high_learning_rate", {"learning_rate": 0.003}),
    Candidate("slow_exploration_decay", {"epsilon_decay_steps": 3000}),
    Candidate("fast_exploration_decay", {"epsilon_decay_steps": 750}),
)

# Predetermined, shared across every candidate. train() derives the network
# initialization, exploration draws, and the entire training scenario stream
# deterministically from this single seed (see AgentConfig/train.train), so
# reusing the same seeds across candidates pairs each candidate's seed-1 run
# against every other candidate's seed-1 run on matched initial weights and
# matched training scenarios, isolating the hyperparameter effect.
TUNING_SEEDS: tuple[int, ...] = (20261010, 20261011, 20261012)

# Predetermined and disjoint from TUNING_SEEDS: final replicas must be new,
# independent runs, not a re-use of whichever tuning seed happened to score
# best, so a lucky tuning seed cannot pass itself off as a final replica.
FINAL_REPLICA_SEEDS: tuple[int, ...] = (20261013, 20261014, 20261015, 20261016, 20261017)

if set(TUNING_SEEDS) & set(FINAL_REPLICA_SEEDS):
    raise AssertionError("TUNING_SEEDS and FINAL_REPLICA_SEEDS must be disjoint")


def _agent_config_for(baseline: AgentConfig, overrides: Mapping[str, object], seed: int) -> AgentConfig:
    return replace(baseline, seed=seed, **dict(overrides))


def _write_agent_config_yaml(path: Path, config: AgentConfig) -> None:
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")


def _load_validation_split(validation_split_path: Path) -> None:
    """Fail clearly if a caller ever points this module at a non-validation split."""
    split = load_split_manifest(validation_split_path)
    if split.split != "validation":
        raise ValueError(
            "PF-21 selection must use the validation split only; "
            f"got split={split.split!r} from {validation_split_path}"
        )


def _run_one(
    *,
    candidate: Candidate,
    seed: int,
    baseline_agent_config: AgentConfig,
    environment_config_path: Path,
    validation_split_path: Path,
    output_root: Path,
) -> dict[str, object]:
    agent_config = _agent_config_for(baseline_agent_config, candidate.overrides, seed)
    with tempfile.TemporaryDirectory() as tmp:
        agent_config_path = Path(tmp) / "agent_config.yaml"
        _write_agent_config_yaml(agent_config_path, agent_config)
        run_directory = train(environment_config_path, agent_config_path, output_root)

    eval_directory = run_directory / "eval-validation"
    evaluate_checkpoint_on_split(
        checkpoint_path=run_directory / "checkpoint.pt",
        split_manifest_path=validation_split_path,
        output_directory=eval_directory,
    )
    rows = read_evaluation_rows_jsonl(eval_directory / "evaluation_rows.jsonl")
    if not rows:
        raise ValueError("validation evaluation produced no rows")
    per_scenario = {row.scenario_id: row.final_portfolio_value for row in rows}
    validation_score = sum(per_scenario.values()) / len(per_scenario)
    return {
        "run_directory": str(run_directory),
        "validation_score": validation_score,
        "per_scenario_final_portfolio_value": per_scenario,
    }


def _select_best_candidate(
    results: Sequence[Mapping[str, object]],
    candidates: Sequence[Candidate],
) -> tuple[str, dict[str, float]]:
    per_candidate: dict[str, list[float]] = {candidate.name: [] for candidate in candidates}
    for record in results:
        if record["status"] == "success":
            per_candidate[record["candidate"]].append(record["validation_score"])
    aggregated = {
        name: sum(scores) / len(scores) for name, scores in per_candidate.items() if scores
    }
    if not aggregated:
        raise RuntimeError(
            "no candidate produced a successful validation run; cannot select "
            "a configuration (see tuning_results.json for recorded failures)"
        )
    best_name = max(
        (candidate.name for candidate in candidates if candidate.name in aggregated),
        key=lambda name: aggregated[name],
    )
    return best_name, aggregated


def run_tuning(
    *,
    environment_config_path: str | Path = _DEFAULT_ENVIRONMENT_CONFIG,
    baseline_agent_config_path: str | Path = _DEFAULT_BASELINE_AGENT_CONFIG,
    validation_split_path: str | Path = _DEFAULT_VALIDATION_SPLIT,
    candidates: tuple[Candidate, ...] = TUNING_CANDIDATES,
    seeds: tuple[int, ...] = TUNING_SEEDS,
    output_root: str | Path = _DEFAULT_TUNING_OUTPUT_ROOT,
) -> dict[str, object]:
    """Train and validate every candidate/seed pair; select by validation only.

    Every outcome (success or failure) for every candidate/seed pair is
    recorded in the returned summary and in ``tuning_results.json`` under
    ``output_root`` -- not just the winner. Test results are never read.
    """
    environment_config_path = Path(environment_config_path).resolve()
    baseline_agent_config_path = Path(baseline_agent_config_path).resolve()
    validation_split_path = Path(validation_split_path).resolve()
    _load_validation_split(validation_split_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_agent_config = load_agent_config(baseline_agent_config_path)
    names = [candidate.name for candidate in candidates]
    if len(set(names)) != len(names):
        raise ValueError("candidate names must be unique")
    for candidate in candidates:
        if "total_steps" in candidate.overrides and (
            candidate.overrides["total_steps"] != baseline_agent_config.total_steps
        ):
            raise ValueError(
                f"candidate {candidate.name!r} must not change total_steps "
                "(PF-21 requires every candidate to share the same planned "
                "training budget); reducing the shared budget requires its "
                "own recorded decision instead of a per-candidate override"
            )

    results: list[dict[str, object]] = []
    for candidate in candidates:
        for seed in seeds:
            record: dict[str, object] = {
                "candidate": candidate.name,
                "overrides": dict(candidate.overrides),
                "seed": seed,
            }
            try:
                record.update(
                    status="success",
                    **_run_one(
                        candidate=candidate,
                        seed=seed,
                        baseline_agent_config=baseline_agent_config,
                        environment_config_path=environment_config_path,
                        validation_split_path=validation_split_path,
                        output_root=output_root,
                    ),
                )
            except Exception as error:  # noqa: BLE001 - every failure must be recorded, not hidden
                record.update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
            results.append(record)

    selected_name, aggregated_scores = _select_best_candidate(results, candidates)
    selected_overrides = next(c.overrides for c in candidates if c.name == selected_name)

    summary: dict[str, object] = {
        "tuning_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment_config_path": str(environment_config_path),
        "baseline_agent_config_path": str(baseline_agent_config_path),
        "validation_split_path": str(validation_split_path),
        "training_budget_total_steps": baseline_agent_config.total_steps,
        "tuning_seeds": list(seeds),
        "selection_metric": SELECTION_METRIC,
        "selection_rule": SELECTION_RULE,
        "candidates": [{"name": c.name, "overrides": dict(c.overrides)} for c in candidates],
        "results": results,
        "candidate_mean_validation_score": aggregated_scores,
        "selected_candidate": selected_name,
        "selected_overrides": dict(selected_overrides),
    }
    with (output_root / "tuning_results.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return summary


def train_final_replicas(
    *,
    tuning_summary: Mapping[str, object],
    environment_config_path: str | Path = _DEFAULT_ENVIRONMENT_CONFIG,
    baseline_agent_config_path: str | Path = _DEFAULT_BASELINE_AGENT_CONFIG,
    seeds: tuple[int, ...] = FINAL_REPLICA_SEEDS,
    output_root: str | Path = _DEFAULT_FINAL_REPLICA_OUTPUT_ROOT,
) -> dict[str, object]:
    """Train one independent model replica per seed using the selected config.

    Every requested replica is retained in the returned manifest, including
    any that fail to train -- selection already happened in :func:`run_tuning`
    using validation data only; this step must not perform a second,
    silent "keep only the best replica" selection.
    """
    environment_config_path = Path(environment_config_path).resolve()
    baseline_agent_config_path = Path(baseline_agent_config_path).resolve()
    baseline_agent_config = load_agent_config(baseline_agent_config_path)
    selected_overrides = tuning_summary["selected_overrides"]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    replicas: list[dict[str, object]] = []
    for seed in seeds:
        agent_config = _agent_config_for(baseline_agent_config, selected_overrides, seed)
        replica: dict[str, object] = {"seed": seed}
        with tempfile.TemporaryDirectory() as tmp:
            agent_config_path = Path(tmp) / "agent_config.yaml"
            _write_agent_config_yaml(agent_config_path, agent_config)
            try:
                run_directory = train(environment_config_path, agent_config_path, output_root)
                replica.update(status="success", run_directory=str(run_directory))
            except Exception as error:  # noqa: BLE001 - record every outcome
                replica.update(
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(),
                )
        replicas.append(replica)

    manifest: dict[str, object] = {
        "final_replicas_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_candidate": tuning_summary["selected_candidate"],
        "selected_overrides": dict(selected_overrides),
        "final_replica_seeds": list(seeds),
        "replicas": replicas,
    }
    with (output_root / "final_replicas_manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predetermined DQN learning-rate/exploration candidates "
            "on validation data, then train the final independent replicas "
            "of the selected configuration."
        )
    )
    parser.add_argument(
        "--environment-config", type=Path, default=_DEFAULT_ENVIRONMENT_CONFIG
    )
    parser.add_argument(
        "--baseline-agent-config", type=Path, default=_DEFAULT_BASELINE_AGENT_CONFIG
    )
    parser.add_argument(
        "--validation-split", type=Path, default=_DEFAULT_VALIDATION_SPLIT
    )
    parser.add_argument(
        "--tuning-output-root", type=Path, default=_DEFAULT_TUNING_OUTPUT_ROOT
    )
    parser.add_argument(
        "--final-replica-output-root", type=Path, default=_DEFAULT_FINAL_REPLICA_OUTPUT_ROOT
    )
    parser.add_argument(
        "--skip-final-replicas",
        action="store_true",
        help="Only run the tuning comparison; do not train final replicas.",
    )
    args = parser.parse_args(argv)

    summary = run_tuning(
        environment_config_path=args.environment_config,
        baseline_agent_config_path=args.baseline_agent_config,
        validation_split_path=args.validation_split,
        output_root=args.tuning_output_root,
    )
    print(f"Selected candidate: {summary['selected_candidate']}")
    print(args.tuning_output_root / "tuning_results.json")

    if not args.skip_final_replicas:
        train_final_replicas(
            tuning_summary=summary,
            environment_config_path=args.environment_config,
            baseline_agent_config_path=args.baseline_agent_config,
            output_root=args.final_replica_output_root,
        )
        print(args.final_replica_output_root / "final_replicas_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
