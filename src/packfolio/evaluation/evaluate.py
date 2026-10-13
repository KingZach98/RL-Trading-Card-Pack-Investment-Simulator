from stable_baselines3 import PPO
from packfolio.environment.trading_env import TradingCardEnv
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "models/ppo_trading_card"



def evaluate():
    env = TradingCardEnv(initial_boxes=1)

    model = PPO.load(MODEL_PATH)

    observation, info = env.reset()

    print("\nPPO POLICY - 100 BOXES")
    print("=" * 70)

    terminated = False
    truncated = False

    rows = []
    history = []
    while not terminated and not truncated:
        # Current state BEFORE taking the action
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
        f"PPO Policy - {env.initial_boxes} Starting Boxes"
    )

    plt.xlabel("Date")
    plt.ylabel("Sealed Box Price (USD)")

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        f"results/ppo_policy_{env.initial_boxes}_boxes.png",
        dpi=300
    )

    plt.show()

if __name__ == "__main__":
    evaluate()