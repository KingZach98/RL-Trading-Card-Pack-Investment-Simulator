from __future__ import annotations

import json
from math import isclose
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from packfolio.types import (
    Action,
    EvaluationRow,
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


__all__ = [
    "EvaluationMetadata",
    "EvaluationResult",
    "evaluate_episode",
    "read_evaluation_rows_jsonl",
    "read_step_traces_jsonl",
    "write_evaluation_rows_jsonl",
    "write_step_traces_jsonl",
]
