import os

import matplotlib.pyplot as plt
import numpy as np

from packfolio.environment.trading_env import TradingCardEnv


INVENTORIES = [1, 10, 100]
N_RUNS = 100

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
PPO_VALUES = {
    1: 1791.03,
    10: 17479.38,
    100: 160863.05,
}

# --------------------------------------------------
# Run one episode
# --------------------------------------------------

def run_episode(initial_boxes, policy):
    env = TradingCardEnv(initial_boxes=initial_boxes)
    observation, info = env.reset()

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = policy(env)

        observation, reward, terminated, truncated, info = env.step(action)

    return env.realized_value


# --------------------------------------------------
# Baseline policies
# --------------------------------------------------

def always_keep(env):
    # Liquidate 0% -> KEEP everything
    return np.array([-1.0, 0.0], dtype=np.float32)


def sell_immediately(env):
    # Liquidate 100%, OPEN 0% -> SELL everything
    return np.array([1.0, -1.0], dtype=np.float32)

def always_open(env):
    # Liquidate 100%, OPEN 100%
    return np.array([1.0, 1.0], dtype=np.float32)

def fifty_fifty(env):
    # 50% OPEN, 50% SELL
    return np.array([1.0, 0.0], dtype=np.float32)

def random_policy(env):
    return env.action_space.sample()

def evaluate_deterministic(name, policy, initial_boxes):
    value = run_episode(initial_boxes, policy)

    print(f"{name:<22} ${value:,.2f}")

    return {
        "name": name,
        "mean": value,
        "median": value,
        "std": 0.0,
        "min": value,
        "max": value,
    }

def evaluate_stochastic(name, policy, initial_boxes):
    values = []

    for _ in range(N_RUNS):
        value = run_episode(initial_boxes, policy)
        values.append(value)

    values = np.array(values)

    result = {
        "name": name,
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "min": np.min(values),
        "max": np.max(values),
    }

    print(
        f"{name:<22} "
        f"Mean: ${result['mean']:,.2f} | "
        f"Median: ${result['median']:,.2f} | "
        f"Std: ${result['std']:,.2f} | "
        f"Min: ${result['min']:,.2f} | "
        f"Max: ${result['max']:,.2f}"
    )

    return result

def plot_results(results, initial_boxes):
    names = [result["name"] for result in results]
    means = [result["mean"] for result in results]
    stds = [result["std"] for result in results]

    ppo_value = PPO_VALUES[initial_boxes]

    x = np.arange(len(names))

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.8,
    )

    # PPO reference line
    plt.axhline(
        y=ppo_value,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"PPO = ${ppo_value:,.0f}",
    )

    plt.xticks(x, names)
    plt.ylabel("Final Portfolio Value (USD)")
    plt.title(
        f"Strategy Comparison - "
        f"{initial_boxes} Starting Box"
        f"{'es' if initial_boxes > 1 else ''}"
    )

    plt.grid(axis="y", alpha=0.2)

    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${mean:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.legend()
    plt.tight_layout()

    path = (
        f"{RESULTS_DIR}/"
        f"strategy_comparison_{initial_boxes}_boxes.png"
    )

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved plot: {path}")

def main():

    for initial_boxes in INVENTORIES:

        print()
        print("=" * 90)
        print(f"{initial_boxes} STARTING BOXES")
        print("=" * 90)

        results = []

        results.append(
            evaluate_deterministic(
                "Always Keep",
                always_keep,
                initial_boxes,
            )
        )

        results.append(
            evaluate_deterministic(
                "Sell Immediately",
                sell_immediately,
                initial_boxes,
            )
        )

        results.append(
            evaluate_stochastic(
                "Always Open",
                always_open,
                initial_boxes,
            )
        )

        # 50/50 only makes sense with more than one box
        if initial_boxes > 1:
            results.append(
                evaluate_stochastic(
                    "50/50 Open/Sell",
                    fifty_fifty,
                    initial_boxes,
                )
            )

        results.append(
            evaluate_stochastic(
                "Random",
                random_policy,
                initial_boxes,
            )
        )

        plot_results(
            results,
            initial_boxes,
        )

if __name__ == "__main__":
    main()