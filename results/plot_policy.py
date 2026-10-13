import pandas as pd
import matplotlib.pyplot as plt


box_counts = [1, 10, 100]

for boxes in box_counts:
    df = pd.read_csv(
        f"results/ppo_policy_{boxes}_boxes.csv"
    )

    df["date"] = pd.to_datetime(df["date"])

    # Percentage of starting inventory still kept
    df["inventory_pct"] = (
        df["boxes_before"] / boxes * 100
    )

    plt.plot(
        df["date"],
        df["inventory_pct"],
        label=f"{boxes} boxes"
    )


plt.xlabel("Date")
plt.ylabel("Remaining inventory (%)")
plt.title("PPO Inventory Strategy Over Time")

plt.ylim(0, 105)
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    "results/ppo_inventory_strategy.png",
    dpi=300
)

plt.show()