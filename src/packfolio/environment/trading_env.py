import pandas as pd

from packfolio.simulator.box_sim import open_box, calculate_box_value
from data.market_data import get_card_market_factor


BOX_PRICE_PATH = "data/2020_panini_prizm_hobby_box_prices.csv"

KEEP = 0
OPEN = 1
SELL = 2


class TradingCardEnv:
    def __init__(self, initial_boxes=1):
        self.initial_boxes = initial_boxes
        self.market_data = pd.read_csv(BOX_PRICE_PATH)
        self.market_data = self.market_data[
            self.market_data["date"] >= "2021-02"
        ].reset_index(drop=True)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.remaining_boxes = self.initial_boxes
        self.realized_value = 0.0
        self.done = False

        return self.get_state()

    def get_state(self):

        row = self.market_data.iloc[self.current_step]

        date = row["date"]
        sealed_price = float(row["price_usd"])
        card_market_factor = get_card_market_factor(date)

        return {
            "date": date,
            "remaining_boxes": self.remaining_boxes,
            "sealed_price": sealed_price,
            "card_market_factor": card_market_factor,
        }

    def step(self, action):
        """
        Perform one action.

        Actions:
            KEEP = 0
            OPEN = 1
            SELL = 2
        """
        if self.done:
            raise RuntimeError("Episode has already finished.")

        open_count, sell_count = action

        # Validate action
        if open_count < 0 or sell_count < 0:
            raise ValueError("OPEN and SELL counts cannot be negative.")

        if open_count + sell_count > self.remaining_boxes:
            raise ValueError(
                f"Cannot use {open_count + sell_count} boxes "
                f"when only {self.remaining_boxes} remain."
            )

        state = self.get_state()

        date = state["date"]
        sealed_price = state["sealed_price"]

        # Everything not opened or sold is kept
        keep_count = self.remaining_boxes - open_count - sell_count

        opened_value = 0.0

        for _ in range(open_count):
            box = open_box()
            opened_value += calculate_box_value(box, date)

        sold_value = sell_count * sealed_price

        reward = opened_value + sold_value

        self.realized_value += reward

        # Only kept boxes remain in inventory
        self.remaining_boxes = keep_count

        if self.remaining_boxes == 0:
            self.done = True

            return None, reward, self.done

        # Move to next month
        self.current_step += 1

        if self.current_step >= len(self.market_data):
            final_price = float(
                self.market_data.iloc[-1]["price_usd"]
            )

            liquidation_value = (
                self.remaining_boxes * final_price
            )

            reward += liquidation_value
            self.realized_value += liquidation_value

            self.remaining_boxes = 0
            self.done = True

            return None, reward, self.done

        return self.get_state(), reward, self.done

if __name__ == "__main__":
    env = TradingCardEnv(initial_boxes=100)

    state = env.reset()
    print("Initial state:", state)

    # Open 0, sell 30, keep 70
    state, reward, done = env.step((0, 30))

    print("\nAfter February:")
    print("State:", state)
    print("Reward:", reward)
    print("Realized value:", env.realized_value)
    print("Done:", done)