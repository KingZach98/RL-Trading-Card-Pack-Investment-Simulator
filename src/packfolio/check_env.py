import argparse
from pathlib import Path
import sys
import warnings

from gymnasium.utils.env_checker import check_env

from packfolio.config import load_environment_config
from packfolio.env import PackfolioEnv
from packfolio.types import Action, OBSERVATION_SIZE, ObservationIndex


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "environment.json"


def check_pytorch(env, *, horizon, seed=1001):
    import torch

    traces = []
    with torch.random.fork_rng(devices=[]):
        network = torch.nn.Linear(OBSERVATION_SIZE, len(Action), device="cpu", dtype=torch.float32)
        with torch.no_grad():
            network.weight.zero_()
        for _ in range(2):
            observation, _ = env.reset(seed=seed)
            trace = []
            for index in range(horizon):
                state = torch.from_numpy(observation).unsqueeze(0)
                assert state.shape == (1, OBSERVATION_SIZE), "observation shape does not match the network"
                assert state.dtype == torch.float32
                assert torch.isfinite(state).all().item()
                with torch.no_grad():
                    network.bias.zero_()
                    network.bias[index % len(Action)] = 1.0
                    values = network(state)
                    assert values.shape == (1, len(Action))
                    assert torch.isfinite(values).all().item()
                    action = Action(values.argmax(dim=1).item())
                observation, reward, terminated, truncated, info = env.step(action)
                reward_tensor = torch.tensor([reward], dtype=torch.float32, device="cpu")
                assert torch.isfinite(reward_tensor).all().item()
                assert env.observation_space.contains(observation)
                assert terminated is (index == horizon - 1)
                assert truncated is False
                trace.append((observation.tolist(), reward, terminated, truncated, info))
            final_state = torch.from_numpy(observation).unsqueeze(0)
            assert final_state.shape == (1, OBSERVATION_SIZE)
            assert torch.isfinite(final_state).all().item()
            assert observation[ObservationIndex.SEALED_COUNT_RATIO] == 0
            assert observation[ObservationIndex.REMAINING_STEPS_RATIO] == 0
            try:
                env.step(Action.HOLD)
            except RuntimeError as error:
                assert "episode has ended" in str(error)
            else:
                raise AssertionError("step() accepted an action after termination")
            traces.append(trace)
    assert traces[0] == traces[1], "the seeded PyTorch run did not replay"
    return len(traces[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check Packfolio's Gymnasium and PyTorch interfaces.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--inventory-capacity", type=int, default=10)
    parser.add_argument("--reference-price", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args(argv)
    config = load_environment_config(args.config)
    env = PackfolioEnv(config, inventory_capacity=args.inventory_capacity, reference_price=args.reference_price)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*A Box observation space maximum value is infinity",
                                    category=UserWarning)
            check_env(env, skip_render_check=True)
        steps = check_pytorch(env, horizon=config.horizon, seed=args.seed)
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        print("PyTorch is not installed. Run: py -m uv sync --locked --extra dev --extra agent", file=sys.stderr)
        return 1
    finally:
        env.close()
    print(f"Gymnasium and PyTorch boundary checks passed ({steps} steps, replay matched).")
    print(f"Seed: {args.seed}; capacity: {args.inventory_capacity}; reference price: {args.reference_price}")
    print(f"Config hash: {config.config_hash}")
    print("This checks an untrained test network, not DQN training or agent quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
