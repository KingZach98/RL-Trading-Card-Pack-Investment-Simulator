from __future__ import annotations

import argparse
from math import isclose
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import importlib.metadata
from pathlib import Path
import json
import re

from packfolio.agents.dqn_agent import (
    CheckpointMetadata,
    load_checkpoint_metadata,
    load_trained_agent,
)
from packfolio.env import DEFAULT_INVENTORY_CAPACITY, build_environment
from packfolio.config import EnvironmentConfig, _integer, _positive_float
from packfolio.provenance import git_metadata
from packfolio.scenarios import ScenarioSpec, SplitManifest, load_split_manifest
from packfolio.types import (
    Action,
    EvaluationRow,
    MarketRegime,
    StepInfo,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationMetadata:
    simulator_version: str
    policy_id: str
    model_id: str | None
    training_seed: int | None
    scenario_id: str
    scenario_seed: int
    config_hash: str
    git_commit: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    row: EvaluationRow
    step_traces: tuple[StepInfo, ...]


def evaluate_episode(
    policy: object,
    env: object,
    metadata: EvaluationMetadata,
) -> EvaluationResult:
    observation, _reset_info = env.reset(seed=metadata.scenario_seed)
    steps: list[StepInfo] = []
    cumulative_reward = 0.0

    while True:
        action = policy.choose_action(observation)
        if not isinstance(action, Action):
            raise TypeError("policy.choose_action() must return Action")

        observation, reward, terminated, truncated, info = env.step(action)
        step = _step_info_from_mapping(info)
        if step.requested_action is not action:
            raise ValueError(
                "StepInfo.requested_action must match the action passed to env.step()"
            )
        if not isclose(float(reward), step.reward, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "environment reward and StepInfo.reward disagree"
            )
        steps.append(step)
        cumulative_reward += step.reward

        if terminated or truncated:
            row = _build_evaluation_row(
                metadata=metadata,
                steps=tuple(steps),
                cumulative_reward=cumulative_reward,
                terminated=terminated,
                truncated=truncated,
            )
            return EvaluationResult(row=row, step_traces=tuple(steps))


def write_evaluation_rows_jsonl(
    path: str | Path,
    rows: Iterable[EvaluationRow],
) -> None:
    """Write one PF-03 EvaluationRow dictionary per JSONL line."""
    with Path(path).open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row.to_dict(), sort_keys=True))
            output_file.write("\n")


def write_step_traces_jsonl(
    path: str | Path,
    steps: Iterable[StepInfo],
) -> None:
    """Write one PF-03 StepInfo dictionary per JSONL line."""
    with Path(path).open("w", encoding="utf-8") as output_file:
        for step in steps:
            output_file.write(json.dumps(step.to_dict(), sort_keys=True))
            output_file.write("\n")


def read_evaluation_rows_jsonl(path: str | Path) -> tuple[EvaluationRow, ...]:
    with Path(path).open(encoding="utf-8") as input_file:
        return tuple(
            EvaluationRow.from_dict(json.loads(line))
            for line in input_file
            if line.strip()
        )


def read_step_traces_jsonl(path: str | Path) -> tuple[StepInfo, ...]:
    with Path(path).open(encoding="utf-8") as input_file:
        return tuple(
            StepInfo.from_dict(json.loads(line))
            for line in input_file
            if line.strip()
        )


def _step_info_from_mapping(info: object) -> StepInfo:
    if isinstance(info, StepInfo):
        return info
    if not isinstance(info, Mapping):
        raise TypeError("env.step() info must be a mapping")
    return StepInfo.from_dict(info)


def _build_evaluation_row(
    *,
    metadata: EvaluationMetadata,
    steps: tuple[StepInfo, ...],
    cumulative_reward: float,
    terminated: bool,
    truncated: bool,
) -> EvaluationRow:
    if not steps:
        raise ValueError("cannot build an evaluation row without steps")

    requested_counts = _count_actions(step.requested_action for step in steps)
    executed_counts = _count_actions(step.executed_action for step in steps)
    infeasible_count = sum(int(step.action_was_infeasible) for step in steps)
    first_step = steps[0]
    last_step = steps[-1]

    return EvaluationRow(
        simulator_version=metadata.simulator_version,
        policy_id=metadata.policy_id,
        model_id=metadata.model_id,
        training_seed=metadata.training_seed,
        scenario_id=metadata.scenario_id,
        scenario_seed=metadata.scenario_seed,
        config_hash=metadata.config_hash,
        git_commit=metadata.git_commit,
        episode_steps=len(steps),
        initial_portfolio_value=first_step.portfolio_value_before,
        final_portfolio_value=last_step.portfolio_value_after,
        cumulative_reward=cumulative_reward,
        requested_hold_count=requested_counts[Action.HOLD],
        requested_buy_pack_count=requested_counts[Action.BUY_PACK],
        requested_open_and_sell_count=requested_counts[Action.OPEN_AND_SELL],
        requested_sell_pack_count=requested_counts[Action.SELL_PACK],
        executed_hold_count=executed_counts[Action.HOLD],
        executed_buy_pack_count=executed_counts[Action.BUY_PACK],
        executed_open_and_sell_count=executed_counts[Action.OPEN_AND_SELL],
        executed_sell_pack_count=executed_counts[Action.SELL_PACK],
        infeasible_action_count=infeasible_count,
        terminated=terminated,
        truncated=truncated,
        termination_reason=last_step.termination_reason,
    )


def _count_actions(actions: Iterable[Action]) -> dict[Action, int]:
    counts = {action: 0 for action in Action}
    for action in actions:
        counts[action] += 1
    return counts


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationManifest:
    """Links a reloaded checkpoint, its configuration, commit, and scenario set.

    This is the PF-18 "model manifest": together with the checkpoint file and
    the accompanying PF-25 evaluation_settings.json it
    is everything a second person needs, in a fresh process, to reproduce the
    exact action sequences and portfolio traces an evaluation run recorded.
    """

    manifest_version: int
    created_at_utc: str
    policy_id: str
    model_id: str
    training_seed: int
    checkpoint_file: str
    environment_config_hash: str
    git_commit: str
    scenario_split: str
    scenario_manifest_file: str
    scenario_ids: tuple[str, ...]
    simulator_version: str
    evaluation_rows_file: str
    step_traces_directory: str

    def __post_init__(self) -> None:
        if self.manifest_version != 1:
            raise ValueError("unsupported evaluation manifest_version")
        for name in (
            "created_at_utc",
            "policy_id",
            "model_id",
            "checkpoint_file",
            "git_commit",
            "scenario_split",
            "scenario_manifest_file",
            "simulator_version",
            "evaluation_rows_file",
            "step_traces_directory",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        if not re.fullmatch(r"[0-9a-f]{64}", self.environment_config_hash):
            raise ValueError("environment_config_hash must be a lowercase SHA-256 digest")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValueError("training_seed must be a nonnegative integer")
        if not isinstance(self.scenario_ids, tuple) or not self.scenario_ids:
            raise ValueError("scenario_ids must be a nonempty tuple")
        if any(not isinstance(scenario_id, str) for scenario_id in self.scenario_ids):
            raise TypeError("scenario_ids must contain strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "created_at_utc": self.created_at_utc,
            "policy_id": self.policy_id,
            "model_id": self.model_id,
            "training_seed": self.training_seed,
            "checkpoint_file": self.checkpoint_file,
            "environment_config_hash": self.environment_config_hash,
            "git_commit": self.git_commit,
            "scenario_split": self.scenario_split,
            "scenario_manifest_file": self.scenario_manifest_file,
            "scenario_ids": list(self.scenario_ids),
            "simulator_version": self.simulator_version,
            "evaluation_rows_file": self.evaluation_rows_file,
            "step_traces_directory": self.step_traces_directory,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EvaluationManifest":
        required = set(_field_name for _field_name in cls.__slots__)
        data = _json_record(data, required, "evaluation manifest")
        return cls(
            manifest_version=data["manifest_version"],
            created_at_utc=data["created_at_utc"],
            policy_id=data["policy_id"],
            model_id=data["model_id"],
            training_seed=data["training_seed"],
            checkpoint_file=data["checkpoint_file"],
            environment_config_hash=data["environment_config_hash"],
            git_commit=data["git_commit"],
            scenario_split=data["scenario_split"],
            scenario_manifest_file=data["scenario_manifest_file"],
            scenario_ids=tuple(data["scenario_ids"]),
            simulator_version=data["simulator_version"],
            evaluation_rows_file=data["evaluation_rows_file"],
            step_traces_directory=data["step_traces_directory"],
        )


def _json_record(
    data: object, required: set[str], record_name: str
) -> dict[str, object]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{record_name} data must be a mapping")
    actual = set(data)
    if actual != required:
        missing = required - actual
        extra = actual - required
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"extra fields: {', '.join(sorted(str(item) for item in extra))}")
        raise ValueError(f"invalid {record_name} data ({'; '.join(details)})")
    return dict(data)


def write_evaluation_manifest(path: str | Path, manifest: EvaluationManifest) -> None:
    with Path(path).open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest.to_dict(), manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")


def read_evaluation_manifest(path: str | Path) -> EvaluationManifest:
    with Path(path).open(encoding="utf-8") as manifest_file:
        return EvaluationManifest.from_dict(json.load(manifest_file))


def _sanitize_scenario_id_for_filename(scenario_id: str) -> str:
    """Scenario IDs contain ':', which is not a safe filename character on Windows."""
    return scenario_id.replace(":", "_")


def _checkpoint_evaluation_settings(
    config: EnvironmentConfig,
    metadata: CheckpointMetadata,
    checkpoint_path: Path,
) -> dict[str, object]:
    reference_price = config.market.quotes[MarketRegime.NORMAL].pack_ask
    training_manifest_path = checkpoint_path.parent / "run_manifest.json"
    source = "legacy_v1_fixed_settings"
    if training_manifest_path.exists():
        with training_manifest_path.open(encoding="utf-8") as manifest_file:
            training_manifest = json.load(manifest_file)
        if not isinstance(training_manifest, Mapping):
            raise ValueError("training manifest must be a JSON object")
        manifest_version = _integer(training_manifest.get("manifest_version"), "manifest_version", minimum=1)
        if manifest_version != 1:
            raise ValueError("unsupported training manifest_version")
        if training_manifest.get("checkpoint_file") != checkpoint_path.name:
            raise ValueError("training manifest checkpoint_file does not match checkpoint")
        if training_manifest.get("environment_config_hash") != metadata.environment_config_hash:
            raise ValueError("training manifest environment_config_hash does not match checkpoint")
        training_seed = _integer(training_manifest.get("training_seed"), "training_seed")
        if training_seed != metadata.training_seed:
            raise ValueError("training manifest training_seed does not match checkpoint")
        capacity = _integer(training_manifest.get("inventory_capacity"), "inventory_capacity", minimum=1)
        if capacity != DEFAULT_INVENTORY_CAPACITY:
            raise ValueError("training manifest inventory_capacity differs from fixed training capacity")
        recorded_price = _positive_float(
            training_manifest.get("observation_reference_price"), "observation_reference_price"
        )
        if recorded_price != reference_price:
            raise ValueError("training manifest observation_reference_price does not match environment")
        source = "training_manifest"

    # Keep the scenario/config hash unchanged: these settings have a separate
    # identity, so old checkpoints and latent scenario draws remain compatible.
    settings = {
        "environment_config_hash": config.config_hash,
        "inventory_capacity": DEFAULT_INVENTORY_CAPACITY,
        "observation_reference_price": reference_price,
    }
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "settings_version": 1,
        **settings,
        "settings_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "training_settings_source": source,
        "training_manifest_file": str(training_manifest_path) if source == "training_manifest" else None,
        "checkpoint_file": str(checkpoint_path),
        "evaluation_manifest_file": "evaluation_manifest.json",
    }


def evaluate_checkpoint_on_split(
    *,
    checkpoint_path: str | Path,
    split_manifest_path: str | Path,
    output_directory: str | Path,
    inventory_capacity: int = DEFAULT_INVENTORY_CAPACITY,
    policy_id: str = "DQN",
) -> Path:
    """Reload a saved DQN checkpoint and evaluate it on a scenario split.

    This is the fresh-process entry point PF-18 requires: given only a
    checkpoint file and a scenario split manifest (no in-memory training
    object), it rebuilds the trained policy and the exact environment it was
    trained against, replays every scenario in the split with
    :func:`evaluate_episode`, and writes an :class:`EvaluationManifest` that
    links the model, its configuration, the commit, and the scenario set.

    Raises ``ValueError`` immediately, before any episode runs, if the
    checkpoint's recorded ``environment_config_hash`` does not match the
    split manifest's environment configuration, so an incompatible
    observation/action schema fails clearly rather than silently producing
    meaningless actions.

    Capacity must match the current v1 training convention. A companion
    training manifest is validated when present; detached v1 checkpoints use
    the documented fixed settings. Resolved settings and their separate hash
    are written to evaluation_settings.json without changing existing schemas.
    """
    inventory_capacity = _integer(inventory_capacity, "inventory_capacity", minimum=1)
    if inventory_capacity != DEFAULT_INVENTORY_CAPACITY:
        raise ValueError("inventory_capacity must match the fixed training capacity (10)")
    checkpoint_path = Path(checkpoint_path).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    output_directory = Path(output_directory)

    checkpoint_metadata: CheckpointMetadata = load_checkpoint_metadata(checkpoint_path)
    split: SplitManifest = load_split_manifest(split_manifest_path)
    if split.config.config_hash != checkpoint_metadata.environment_config_hash:
        raise ValueError(
            "checkpoint environment_config_hash "
            f"({checkpoint_metadata.environment_config_hash}) does not match "
            f"the split manifest's environment configuration "
            f"({split.config.config_hash}); refusing to evaluate a checkpoint "
            "against an incompatible observation/action schema"
        )

    settings = _checkpoint_evaluation_settings(split.config, checkpoint_metadata, checkpoint_path)
    policy = load_trained_agent(checkpoint_path)
    git_commit, _git_dirty = git_metadata()
    simulator_version = importlib.metadata.version("packfolio")
    model_id = checkpoint_path.parent.name

    traces_directory = output_directory / "step_traces"
    output_directory.mkdir(parents=True, exist_ok=True)
    traces_directory.mkdir(exist_ok=True)

    rows: list[EvaluationRow] = []
    scenario: ScenarioSpec
    for scenario in split.scenarios:
        env = build_environment(split.config, inventory_capacity=inventory_capacity)
        row_metadata = EvaluationMetadata(
            simulator_version=simulator_version,
            policy_id=policy_id,
            model_id=model_id,
            training_seed=checkpoint_metadata.training_seed,
            scenario_id=scenario.scenario_id,
            scenario_seed=scenario.seed,
            config_hash=split.config.config_hash,
            git_commit=git_commit,
        )
        result = evaluate_episode(policy, env, row_metadata)
        rows.append(result.row)
        trace_path = (
            traces_directory
            / f"{_sanitize_scenario_id_for_filename(scenario.scenario_id)}.jsonl"
        )
        write_step_traces_jsonl(trace_path, result.step_traces)

    evaluation_rows_file = "evaluation_rows.jsonl"
    write_evaluation_rows_jsonl(output_directory / evaluation_rows_file, rows)

    manifest = EvaluationManifest(
        manifest_version=1,
        created_at_utc=datetime.now(UTC).isoformat(),
        policy_id=policy_id,
        model_id=model_id,
        training_seed=checkpoint_metadata.training_seed,
        checkpoint_file=str(checkpoint_path),
        environment_config_hash=split.config.config_hash,
        git_commit=git_commit,
        scenario_split=split.split,
        scenario_manifest_file=str(split_manifest_path),
        scenario_ids=tuple(scenario.scenario_id for scenario in split.scenarios),
        simulator_version=simulator_version,
        evaluation_rows_file=evaluation_rows_file,
        step_traces_directory=traces_directory.name,
    )
    write_evaluation_manifest(output_directory / "evaluation_manifest.json", manifest)
    with (output_directory / "evaluation_settings.json").open("w", encoding="utf-8") as settings_file:
        json.dump(settings, settings_file, indent=2, sort_keys=True, allow_nan=False)
        settings_file.write("\n")
    return output_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reload a saved DQN checkpoint in a fresh process and evaluate it "
            "on a scenario split manifest."
        )
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to a checkpoint.pt file"
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Path to a scenario split manifest JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for evaluation rows, step traces, and the manifest",
    )
    parser.add_argument(
        "--policy-id",
        default="DQN",
        help="Policy identifier recorded in evaluation rows (default: DQN)",
    )
    args = parser.parse_args(argv)
    output_directory = evaluate_checkpoint_on_split(
        checkpoint_path=args.checkpoint,
        split_manifest_path=args.split_manifest,
        output_directory=args.output,
        policy_id=args.policy_id,
    )
    print(output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvaluationManifest",
    "EvaluationMetadata",
    "EvaluationResult",
    "evaluate_checkpoint_on_split",
    "evaluate_episode",
    "read_evaluation_manifest",
    "read_evaluation_rows_jsonl",
    "read_step_traces_jsonl",
    "write_evaluation_manifest",
    "write_evaluation_rows_jsonl",
    "write_step_traces_jsonl",
]
