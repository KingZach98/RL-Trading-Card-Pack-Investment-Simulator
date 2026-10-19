import numpy as np
import matplotlib.pyplot as plt 

from packfolio.simulator.box_sim import (
    open_box,
    calculate_box_value,
)

DATE = "2021-02-01"
POOL_SIZE = 10_000
SIMULATIONS = 10_000

box_values = np.empty(POOL_SIZE)

for i in range(POOL_SIZE):
    box = open_box()
    box_values[i] = calculate_box_value(box, DATE)

    if (i + 1) % 1000 == 0:
        print(f"{i + 1}/{POOL_SIZE}")


print("\nPool generated.")
print(f"Single box mean:   ${np.mean(box_values):,.2f}")
print(f"Single box median: ${np.median(box_values):,.2f}")


for num_boxes in [1, 10, 100, 1000]:
    samples = np.random.choice(
        box_values,
        size=(SIMULATIONS, num_boxes),
        replace=True
    )

    portfolio_values = samples.sum(axis=1)

    sell_value = num_boxes * 1284.74

    prob_open_beats_sell = np.mean(
        portfolio_values > sell_value
    )

    print(f"SELL value:${sell_value:,.2f}")
    print(
        f"P(OPEN > SELL):"
        f"{prob_open_beats_sell * 100:.2f}%"
    )

    print(f"\n{num_boxes} BOXES")
    print("-" * 30)
    print(f"Mean:${np.mean(portfolio_values):,.2f}")
    print(f"Median:${np.median(portfolio_values):,.2f}")
    print(f"Std:${np.std(portfolio_values):,.2f}")
    print(f"5%:${np.percentile(portfolio_values, 5):,.2f}")
    print(f"95%:${np.percentile(portfolio_values, 95):,.2f}")
    print(f"Min:${np.min(portfolio_values):,.2f}")
    print(f"Max:${np.max(portfolio_values):,.2f}")

    sealed_price = 1284.74

sealed_price = 1284.74
mean_value = np.mean(box_values)
probability = np.mean(box_values > sealed_price) * 100

x_max = np.percentile(box_values, 99)

bins = np.linspace(0, x_max, 60)

plt.figure(figsize=(10, 5.5))

plt.hist(
    box_values,
    bins=bins,
    alpha=0.8
)

plt.axvline(
    mean_value,
    color="green",
    linestyle="--",
    linewidth=2,
    label=f"Mean OPEN = ${mean_value:,.0f}"
)

plt.axvline(
    sealed_price,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"SELL = ${sealed_price:,.0f}"
)

plt.text(
    sealed_price + 40,
    plt.ylim()[1] * 0.55,
    f"{probability:.2f}%\nof openings\nbeat SELL",
    color="red",
    fontsize=10
)

plt.xlim(0, x_max)

plt.xlabel("Value of opened box (USD)")
plt.ylabel("Number of simulations")
plt.title("Distribution of Simulated Box Opening Values")

plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    "results/open_value_distribution.png",
    dpi=300
)

plt.show()