from stable_baselines3 import PPO
from packfolio.environment.trading_env import TradingCardEnv
import pandas as pd
import matplotlib.pyplot as plt
import os


MODEL_PATH = "models/ppo_trading_card"
INVENTORIES = [1, 10, 100]

os.makedirs("results", exist_ok=True)


def evaluate(initial_boxes, model):
    env = TradingCardEnv(initial_boxes=initial_boxes)

    observation, info = env.reset()

    print(f"\nPPO POLICY - {initial_boxes} STARTING BOXES")
    print("=" * 70)

    terminated = False
    truncated = False

    history = []

    while not terminated and not truncated:
        # State BEFORE taking action
        state = env.get_state()

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        open_count, sell_count = env.convert_action(action)

        keep_count = (
            env.remaining_boxes
            - open_count
            - sell_count
        )

        history.append({
            "date": state["date"],
            "price": state["sealed_price"],
            "boxes": env.remaining_boxes,
            "open": open_count,
            "sell": sell_count,
            "keep": keep_count,
        })

        observation, reward, terminated, truncated, info = env.step(
            action
        )

    # Final result
    print(f"Final realized value: ${env.realized_value:,.2f}")

    # --------------------------------------------------
    # Plot policy
    # --------------------------------------------------

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(12, 6))

    # Market price
    plt.plot(
        df["date"],
        df["price"],
        linewidth=2,
        label="Sealed box price"
    )

    # SELL actions
    sell_df = df[df["sell"] > 0]

    plt.scatter(
        sell_df["date"],
        sell_df["price"],
        s=70,
        color="red",
        marker="v",
        label="SELL",
        zorder=3
    )

    for _, row in sell_df.iterrows():
        plt.annotate(
            f'{int(row["sell"])}',
            (row["date"], row["price"]),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

    # OPEN actions
    open_df = df[df["open"] > 0]

    plt.scatter(
        open_df["date"],
        open_df["price"],
        s=70,
        color="green",
        marker="^",
        label="OPEN",
        zorder=3
    )

    for _, row in open_df.iterrows():
        plt.annotate(
            f'{int(row["open"])}',
            (row["date"], row["price"]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

    plt.title(
        f"PPO Policy - {initial_boxes} Starting Box"
        f"{'es' if initial_boxes > 1 else ''}"
    )

    plt.xlabel("Date")
    plt.ylabel("Sealed Box Price (USD)")

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    path = f"results/ppo_policy_{initial_boxes}_boxes.png"

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(f"Saved plot: {path}")

    return env.realized_value


def main():
    # IMPORTANT:
    # Same PPO model is used for all inventory sizes.
    # The model was trained with 100 starting boxes.
    model = PPO.load(MODEL_PATH)

    ppo_values = {}

    for initial_boxes in INVENTORIES:
        value = evaluate(
            initial_boxes,
            model
        )

        ppo_values[initial_boxes] = value

    print("\n" + "=" * 70)
    print("PPO SUMMARY")
    print("=" * 70)

    for initial_boxes, value in ppo_values.items():
        print(
            f"{initial_boxes:>3} boxes: "
            f"${value:,.2f}"
        )


if __name__ == "__main__":
    main()