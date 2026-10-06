"""Agent sanity checks: greedy rollouts, action/Q-value traces, and smoke-run guards.

This module separates learning bugs from genuinely difficult or unprofitable
environments (PF-17). It runs a trained checkpoint greedily against an
environment configuration, records the executed action distribution (not
just the requested one, since an infeasible request silently becomes
``HOLD``; see :mod:`packfolio.env`) and per-step Q-values, and raises
immediately on any non-finite network output or reward. Diagnostic fixtures
under ``configs/diagnostics`` give known-answer cases; this module is not a
substitute for final experiments on the frozen market.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from packfolio.agents.dqn_agent import (
    ACTION_COUNT,
    build_q_network,
    load_trained_agent,
    preprocess_observations,
)
from packfolio.config import EnvironmentConfig, load_environment_config
from packfolio.env import PackfolioEnv
from packfolio.types import Action, MarketRegime


_INVENTORY_CAPACITY = 10


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """One greedy rollout episode, for inspection and debug output."""

    seed: int
    requested_actions: tuple[Action, ...]
    executed_actions: tuple[Action, ...]
    rewards: tuple[float, ...]
    q_values: tuple[tuple[float, ...], ...]
    episode_return: float
    final_portfolio_value: float

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "requested_actions": [action.name for action in self.requested_actions],
            "executed_actions": [action.name for action in self.executed_actions],
            "rewards": list(self.rewards),
            "q_values": [list(row) for row in self.q_values],
            "episode_return": self.episode_return,
            "final_portfolio_value": self.final_portfolio_value,
        }


@dataclass(frozen=True, slots=True)
class RolloutDiagnostics:
    """Aggregate statistics across greedy rollout episodes."""

    episodes: tuple[EpisodeTrace, ...]
    requested_action_counts: Counter
    executed_action_counts: Counter
    mean_episode_return: float
    min_q_value: float
    max_q_value: float
    mean_q_value: float

    def executed_action_frequency(self, action: Action) -> float:
        total = sum(self.executed_action_counts.values())
        if total == 0:
            return 0.0
        return self.executed_action_counts.get(action, 0) / total

    def requested_action_frequency(self, action: Action) -> float:
        total = sum(self.requested_action_counts.values())
        if total == 0:
            return 0.0
        return self.requested_action_counts.get(action, 0) / total

    def to_dict(self) -> dict[str, object]:
        return {
            "episodes": [episode.to_dict() for episode in self.episodes],
            "requested_action_counts": {
                action.name: count
                for action, count in self.requested_action_counts.items()
            },
            "executed_action_counts": {
                action.name: count
                for action, count in self.executed_action_counts.items()
            },
            "mean_episode_return": self.mean_episode_return,
            "min_q_value": self.min_q_value,
            "max_q_value": self.max_q_value,
            "mean_q_value": self.mean_q_value,
        }


def _build_rollout_network(checkpoint: dict[str, Any], torch: Any) -> Any:
    architecture = checkpoint["architecture"]
    network = build_q_network(tuple(architecture["hidden_layer_widths"]))
    network.load_state_dict(checkpoint["q_network_state_dict"])
    network.eval()
    return network


def run_greedy_rollout(
    checkpoint_path: str | Path,
    environment_config: EnvironmentConfig,
    *,
    episodes: int = 20,
    seed: int = 0,
    inventory_capacity: int = _INVENTORY_CAPACITY,
    reference_price: float | None = None,
) -> RolloutDiagnostics:
    """Roll out a trained checkpoint greedily and report action/Q-value diagnostics.

    Raises ``FloatingPointError`` if the network or environment ever produces
    a non-finite value; raises via :class:`~packfolio.agents.dqn_agent.DQNAgent`
    if Q-values are the wrong shape or non-finite.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    try:
        torch = __import__("torch")
    except ImportError as error:
        raise ImportError(
            "Rollout diagnostics require PyTorch; install Packfolio with its "
            "'agent' extra."
        ) from error

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    agent = load_trained_agent(checkpoint_path)
    network = _build_rollout_network(checkpoint, torch)

    if reference_price is None:
        reference_price = environment_config.market.quotes[MarketRegime.NORMAL].pack_ask
    env = PackfolioEnv(
        environment_config,
        inventory_capacity=inventory_capacity,
        reference_price=reference_price,
    )

    rng = np.random.default_rng(seed)
    episode_traces: list[EpisodeTrace] = []
    requested_counts: Counter = Counter()
    executed_counts: Counter = Counter()
    all_q_values: list[float] = []

    for _ in range(episodes):
        episode_seed = int(rng.integers(0, 2**64, dtype=np.uint64))
        observation, _ = env.reset(seed=episode_seed)
        requested: list[Action] = []
        executed: list[Action] = []
        rewards: list[float] = []
        q_value_rows: list[tuple[float, ...]] = []
        terminated = False
        while not terminated:
            features = preprocess_observations(observation)
            with torch.inference_mode():
                q_values = network(
                    torch.as_tensor(features[None, :], dtype=torch.float32)
                )
            q_row = q_values.detach().cpu().numpy()[0]
            if q_row.shape != (ACTION_COUNT,) or not np.isfinite(q_row).all():
                raise FloatingPointError(
                    "network produced non-finite or malformed Q-values during a smoke rollout"
                )
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError("PackfolioEnv unexpectedly truncated a rollout episode")
            if not math.isfinite(float(reward)):
                raise FloatingPointError("environment returned a non-finite reward")
            requested_action = Action(info["requested_action"])
            executed_action = Action(info["executed_action"])
            requested.append(requested_action)
            executed.append(executed_action)
            rewards.append(float(reward))
            q_value_rows.append(tuple(float(value) for value in q_row))
            requested_counts[requested_action] += 1
            executed_counts[executed_action] += 1
            all_q_values.extend(q_row.tolist())

        episode_traces.append(
            EpisodeTrace(
                seed=episode_seed,
                requested_actions=tuple(requested),
                executed_actions=tuple(executed),
                rewards=tuple(rewards),
                q_values=tuple(q_value_rows),
                episode_return=sum(rewards),
                final_portfolio_value=float(info["portfolio_value_after"]),
            )
        )

    return RolloutDiagnostics(
        episodes=tuple(episode_traces),
        requested_action_counts=requested_counts,
        executed_action_counts=executed_counts,
        mean_episode_return=sum(trace.episode_return for trace in episode_traces) / episodes,
        min_q_value=min(all_q_values),
        max_q_value=max(all_q_values),
        mean_q_value=sum(all_q_values) / len(all_q_values),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a greedy diagnostic rollout of a trained DQN checkpoint and "
            "print an action-frequency/Q-value report (PF-17 debug trace)."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--environment-config", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    environment_config = load_environment_config(args.environment_config)
    diagnostics = run_greedy_rollout(
        args.checkpoint,
        environment_config,
        episodes=args.episodes,
        seed=args.seed,
    )
    report = diagnostics.to_dict()
    report.pop("episodes")  # the CLI prints a summary; full traces are large.
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EpisodeTrace", "RolloutDiagnostics", "run_greedy_rollout"]
