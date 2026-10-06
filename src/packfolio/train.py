"""Reproducible, configuration-driven DQN training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from uuid import uuid4

import numpy as np
import yaml
from numpy.typing import NDArray

from packfolio.agents.dqn_agent import (
    ACTION_COUNT,
    ACTION_ORDER,
    FEATURE_ORDER,
    FEATURE_SCALE,
    build_q_network,
    preprocess_observations,
)
from packfolio.config import load_environment_config
from packfolio.env import DEFAULT_INVENTORY_CAPACITY, build_environment
from packfolio.provenance import git_metadata, installed_dependencies
from packfolio.types import Action, OBSERVATION_SIZE, MarketRegime


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ENVIRONMENT_CONFIG = _REPOSITORY_ROOT / "configs" / "environment.json"
_DEFAULT_AGENT_CONFIG = _REPOSITORY_ROOT / "configs" / "agent.yaml"
_INVENTORY_CAPACITY = DEFAULT_INVENTORY_CAPACITY


def _positive_integer(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be positive")
    return value


def _nonnegative_seed(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if not 0 <= value < 2**64:
        raise ValueError(f"{field_name} must be in [0, 2**64)")
    return value


def _positive_finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _epsilon(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{field_name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class AgentConfig:
    seed: int
    total_steps: int
    hidden_layer_widths: tuple[int, ...]
    learning_rate: float
    gamma: float
    replay_capacity: int
    batch_size: int
    learning_starts: int
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: int
    gradient_clip_norm: float

    def __post_init__(self) -> None:
        _nonnegative_seed(self.seed, "seed")
        for field_name in (
            "total_steps",
            "replay_capacity",
            "batch_size",
            "learning_starts",
            "target_update_interval",
            "epsilon_decay_steps",
        ):
            _positive_integer(getattr(self, field_name), field_name)
        if (
            not isinstance(self.hidden_layer_widths, tuple)
            or not self.hidden_layer_widths
            or any(type(width) is not int or width < 1 for width in self.hidden_layer_widths)
        ):
            raise ValueError("hidden_layer_widths must be a nonempty tuple of positive integers")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size must not exceed replay_capacity")
        if self.learning_starts > self.replay_capacity:
            raise ValueError("learning_starts must not exceed replay_capacity")
        if isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float)):
            raise TypeError("gamma must be a number")
        if not math.isfinite(float(self.gamma)) or self.gamma != 1.0:
            raise ValueError("gamma must be 1.0 for the configured final-value objective")
        object.__setattr__(self, "gamma", 1.0)
        object.__setattr__(self, "learning_rate", _positive_finite(self.learning_rate, "learning_rate"))
        object.__setattr__(
            self, "gradient_clip_norm", _positive_finite(self.gradient_clip_norm, "gradient_clip_norm")
        )
        object.__setattr__(self, "epsilon_start", _epsilon(self.epsilon_start, "epsilon_start"))
        object.__setattr__(self, "epsilon_end", _epsilon(self.epsilon_end, "epsilon_end"))
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end must not exceed epsilon_start")

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "total_steps": self.total_steps,
            "hidden_layer_widths": list(self.hidden_layer_widths),
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "target_update_interval": self.target_update_interval,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
        }


_AGENT_CONFIG_FIELDS = {
    "seed",
    "total_steps",
    "hidden_layer_widths",
    "learning_rate",
    "gamma",
    "replay_capacity",
    "batch_size",
    "learning_starts",
    "target_update_interval",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay_steps",
    "gradient_clip_norm",
}


def load_agent_config(path: str | Path) -> AgentConfig:
    """Read and validate the separate YAML file containing DQN settings."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise TypeError("agent configuration must be a YAML mapping")
    actual_fields = set(values)
    if actual_fields != _AGENT_CONFIG_FIELDS:
        missing = _AGENT_CONFIG_FIELDS - actual_fields
        extra = actual_fields - _AGENT_CONFIG_FIELDS
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        if extra:
            details.append(f"extra fields: {', '.join(sorted(extra))}")
        raise ValueError(f"invalid agent configuration ({'; '.join(details)})")

    widths = values["hidden_layer_widths"]
    if not isinstance(widths, list):
        raise TypeError("hidden_layer_widths must be a YAML list")
    return AgentConfig(
        seed=_nonnegative_seed(values["seed"], "seed"),
        total_steps=_positive_integer(values["total_steps"], "total_steps"),
        hidden_layer_widths=tuple(
            _positive_integer(width, "hidden_layer_widths item") for width in widths
        ),
        learning_rate=_positive_finite(values["learning_rate"], "learning_rate"),
        gamma=values["gamma"],
        replay_capacity=_positive_integer(values["replay_capacity"], "replay_capacity"),
        batch_size=_positive_integer(values["batch_size"], "batch_size"),
        learning_starts=_positive_integer(values["learning_starts"], "learning_starts"),
        target_update_interval=_positive_integer(
            values["target_update_interval"], "target_update_interval"
        ),
        epsilon_start=_epsilon(values["epsilon_start"], "epsilon_start"),
        epsilon_end=_epsilon(values["epsilon_end"], "epsilon_end"),
        epsilon_decay_steps=_positive_integer(
            values["epsilon_decay_steps"], "epsilon_decay_steps"
        ),
        gradient_clip_norm=_positive_finite(
            values["gradient_clip_norm"], "gradient_clip_norm"
        ),
    )


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._states = np.empty((capacity, OBSERVATION_SIZE), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty((capacity, OBSERVATION_SIZE), dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.float32)
        self._capacity = capacity
        self._size = 0
        self._next_index = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: NDArray[np.float32],
        action: Action,
        reward: float,
        next_state: NDArray[np.float32],
        terminated: bool,
    ) -> None:
        index = self._next_index
        self._states[index] = state
        self._actions[index] = int(action)
        self._rewards[index] = reward
        self._next_states[index] = next_state
        self._terminated[index] = float(terminated)
        self._next_index = (index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        indices = rng.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._terminated[indices],
        )


def _new_run_directory(output_root: Path) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid4().hex[:8]
        run_directory = output_root / run_id
        try:
            run_directory.mkdir()
        except FileExistsError:
            continue
        return run_id, run_directory
    raise FileExistsError("could not allocate a unique training run directory")


def _epsilon_at_step(config: AgentConfig, step: int) -> float:
    fraction = min(step / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def _optimize(
    *,
    torch: Any,
    network: Any,
    target_network: Any,
    optimizer: Any,
    loss_function: Any,
    replay: ReplayBuffer,
    rng: np.random.Generator,
    config: AgentConfig,
) -> float:
    states, actions, rewards, next_states, terminated = replay.sample(
        config.batch_size, rng
    )
    state_tensor = torch.as_tensor(
        preprocess_observations(states), dtype=torch.float32
    )
    action_tensor = torch.as_tensor(actions, dtype=torch.int64)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    next_state_tensor = torch.as_tensor(
        preprocess_observations(next_states), dtype=torch.float32
    )
    terminated_tensor = torch.as_tensor(terminated, dtype=torch.float32)

    predicted = network(state_tensor).gather(1, action_tensor[:, None]).squeeze(1)
    with torch.no_grad():
        next_values = target_network(next_state_tensor).max(dim=1).values
        target = reward_tensor + config.gamma * next_values * (1.0 - terminated_tensor)
    loss = loss_function(predicted, target)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("DQN training produced a non-finite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip_norm)
    optimizer.step()
    loss_value = float(loss.detach().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError("DQN training produced a non-finite loss")
    return loss_value


def train(
    environment_config_path: str | Path = _DEFAULT_ENVIRONMENT_CONFIG,
    agent_config_path: str | Path = _DEFAULT_AGENT_CONFIG,
    output_root: str | Path = "runs",
) -> Path:
    """Train the configured DQN and return its unique run directory."""
    try:
        torch = __import__("torch")
    except ImportError as error:
        raise ImportError(
            "DQN training requires PyTorch; install Packfolio with its 'agent' extra."
        ) from error

    environment_config_file = Path(environment_config_path).resolve()
    agent_config_file = Path(agent_config_path).resolve()
    environment_config = load_environment_config(environment_config_file)
    agent_config = load_agent_config(agent_config_file)
    run_id, run_directory = _new_run_directory(Path(output_root))

    environment_copy = run_directory / "environment_config.json"
    agent_copy = run_directory / "agent_config.yaml"
    shutil.copyfile(environment_config_file, environment_copy)
    shutil.copyfile(agent_config_file, agent_copy)
    resolved_environment_copy = run_directory / "environment_resolved.json"
    with resolved_environment_copy.open("w", encoding="utf-8") as output_file:
        json.dump(
            environment_config.to_dict(),
            output_file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")

    seed_sequence = np.random.SeedSequence(agent_config.seed)
    (
        exploration_sequence,
        environment_sequence,
        torch_sequence,
        replay_sequence,
    ) = seed_sequence.spawn(4)
    exploration_seed = int(exploration_sequence.generate_state(1, dtype=np.uint64)[0])
    environment_seed = int(environment_sequence.generate_state(1, dtype=np.uint64)[0])
    torch_seed = int(torch_sequence.generate_state(1, dtype=np.uint64)[0])
    replay_seed = int(replay_sequence.generate_state(1, dtype=np.uint64)[0])
    exploration_rng = np.random.default_rng(exploration_seed)
    environment_seed_rng = np.random.default_rng(environment_seed)
    replay_rng = np.random.default_rng(replay_seed)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch_seed)
        network = build_q_network(agent_config.hidden_layer_widths)
        target_network = build_q_network(agent_config.hidden_layer_widths)
    target_network.load_state_dict(network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(network.parameters(), lr=agent_config.learning_rate)
    loss_function = torch.nn.SmoothL1Loss()
    replay = ReplayBuffer(agent_config.replay_capacity)
    env = build_environment(environment_config, inventory_capacity=_INVENTORY_CAPACITY)

    metric_path = run_directory / "training_metrics.jsonl"
    environment_seeds: list[int] = []
    global_step = 0
    gradient_updates = 0
    target_network_updates = 0
    episode_count = 0
    started_at_utc = datetime.now(UTC)
    start_time = time.perf_counter()

    with metric_path.open("w", encoding="utf-8") as metrics_file:
        while global_step < agent_config.total_steps:
            episode_seed = int(
                environment_seed_rng.integers(0, 2**64, dtype=np.uint64)
            )
            environment_seeds.append(episode_seed)
            observation, _ = env.reset(seed=episode_seed)
            episode_reward = 0.0
            episode_invalid_actions = 0
            episode_losses: list[float] = []
            final_portfolio_value = environment_config.initial_cash
            episode_steps = 0
            terminated = False

            while global_step < agent_config.total_steps and not terminated:
                epsilon = _epsilon_at_step(agent_config, global_step)
                if exploration_rng.random() < epsilon:
                    action = Action(int(exploration_rng.integers(ACTION_COUNT)))
                else:
                    features = preprocess_observations(observation)
                    with torch.inference_mode():
                        q_values = network(
                            torch.as_tensor(features[None, :], dtype=torch.float32)
                        )
                    action = Action(int(torch.argmax(q_values[0]).item()))

                next_observation, reward, terminated, truncated, info = env.step(action)
                if truncated:
                    raise RuntimeError("PackfolioEnv unexpectedly truncated an episode")
                if not math.isfinite(float(reward)):
                    raise FloatingPointError("environment returned a non-finite reward")
                replay.add(observation, action, float(reward), next_observation, terminated)
                episode_reward += float(reward)
                episode_steps += 1
                global_step += 1
                episode_invalid_actions += int(info["action_was_infeasible"])
                final_portfolio_value = float(info["portfolio_value_after"])
                observation = next_observation

                if (
                    len(replay) >= agent_config.batch_size
                    and global_step >= agent_config.learning_starts
                ):
                    episode_losses.append(
                        _optimize(
                            torch=torch,
                            network=network,
                            target_network=target_network,
                            optimizer=optimizer,
                            loss_function=loss_function,
                            replay=replay,
                            rng=replay_rng,
                            config=agent_config,
                        )
                    )
                    gradient_updates += 1

                if global_step % agent_config.target_update_interval == 0:
                    target_network.load_state_dict(network.state_dict())
                    target_network_updates += 1

            episode_count += 1
            metric = {
                "episode": episode_count,
                "environment_seed": episode_seed,
                "episode_steps": episode_steps,
                "completed": terminated,
                "termination": "HORIZON" if terminated else "STEP_BUDGET",
                "episode_reward": episode_reward,
                "final_portfolio_value": final_portfolio_value,
                "invalid_action_count": episode_invalid_actions,
                "invalid_action_rate": episode_invalid_actions / episode_steps,
                "mean_loss": (
                    sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
                ),
                "gradient_updates": len(episode_losses),
            }
            if not all(
                math.isfinite(float(metric[name]))
                for name in (
                    "episode_reward",
                    "final_portfolio_value",
                    "invalid_action_rate",
                    "mean_loss",
                )
            ):
                raise FloatingPointError("training produced a non-finite episode metric")
            metrics_file.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
            metrics_file.flush()

    duration_seconds = time.perf_counter() - start_time
    checkpoint_path = run_directory / "checkpoint.pt"
    checkpoint = {
        "format_version": 1,
        "architecture": {
            "observation_size": OBSERVATION_SIZE,
            "hidden_layer_widths": list(agent_config.hidden_layer_widths),
            "action_count": ACTION_COUNT,
            "feature_order": list(FEATURE_ORDER),
            "action_order": list(ACTION_ORDER),
            "feature_scale": [float(value) for value in FEATURE_SCALE],
        },
        "q_network_state_dict": network.state_dict(),
        "target_network_state_dict": target_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "environment_config_hash": environment_config.config_hash,
        "agent_config": agent_config.to_dict(),
        "training_seed": agent_config.seed,
        "actual_environment_steps": global_step,
    }
    torch.save(checkpoint, checkpoint_path)

    git_commit, git_dirty = git_metadata()
    manifest: dict[str, object] = {
        "manifest_version": 1,
        "run_id": run_id,
        "status": "complete",
        "started_at_utc": started_at_utc.isoformat(),
        "runtime_seconds": duration_seconds,
        "git_commit": git_commit,
        "git_working_tree_dirty": git_dirty,
        "python_version": sys.version,
        "dependencies": installed_dependencies(),
        "environment_config_file": environment_copy.name,
        "environment_config_source": str(environment_config_file),
        "environment_resolved_file": resolved_environment_copy.name,
        "environment_config_hash": environment_config.config_hash,
        "agent_config_file": agent_copy.name,
        "training_seed": agent_config.seed,
        "derived_seeds": {
            "exploration": exploration_seed,
            "environment_seed_generator": environment_seed,
            "torch": torch_seed,
            "replay_sampling": replay_seed,
        },
        "environment_episode_seeds": environment_seeds,
        "actual_environment_steps": global_step,
        "episodes": episode_count,
        "gradient_updates": gradient_updates,
        "target_network_updates": target_network_updates,
        "gamma": agent_config.gamma,
        "checkpoint_file": checkpoint_path.name,
        "metrics_file": metric_path.name,
        "inventory_capacity": _INVENTORY_CAPACITY,
        "observation_reference_price": environment_config.market.quotes[
            MarketRegime.NORMAL
        ].pack_ask,
    }
    with (run_directory / "run_manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True, allow_nan=False)
        manifest_file.write("\n")
    return run_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a Packfolio DQN agent.")
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=_DEFAULT_ENVIRONMENT_CONFIG,
        help="JSON simulator configuration",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=_DEFAULT_AGENT_CONFIG,
        help="YAML DQN hyperparameter configuration",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Parent directory for unique training run directories",
    )
    args = parser.parse_args(argv)
    run_directory = train(
        args.environment_config, args.agent_config, args.output_root
    )
    print(run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())