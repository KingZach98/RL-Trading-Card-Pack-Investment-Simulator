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

        if action not in (KEEP, OPEN, SELL):
            raise ValueError(f"Invalid action: {action}")

        state = self.get_state()

        date = state["date"]
        sealed_price = state["sealed_price"]

        reward = 0.0

        if action == KEEP:
            pass

        elif action == OPEN:
            box = open_box()
            opened_value = calculate_box_value(box, date)

            reward = opened_value
            self.realized_value += opened_value
            self.remaining_boxes -= 1

        elif action == SELL:
            reward = sealed_price
            self.realized_value += sealed_price
            self.remaining_boxes -= 1

        # If no boxes remain, there are no more decisions to make.
        if self.remaining_boxes == 0:
            self.done = True
            return None, reward, self.done

        #increase month
        self.current_step += 1

        # If we reach the end of the historical data,
        # automatically sell all remaining sealed boxes.
        if self.current_step >= len(self.market_data):
            final_price = float(self.market_data.iloc[-1]["price_usd"])

            liquidation_value = self.remaining_boxes * final_price

            reward += liquidation_value
            self.realized_value += liquidation_value

            self.remaining_boxes = 0
            self.done = True

            return None, reward, self.done

        return self.get_state(), reward, self.done

if __name__ == "__main__":

    env = TradingCardEnv(initial_boxes=1)

    state = env.reset()
    print("Initial state:", state)

    state, reward, done = env.step(OPEN)
    ##state, reward, done = env.step(SELL)
    ##state, reward, done = env.step(KEEP)

    print("State:", state)
    print("Reward:", reward)
    print("Done:", done)
    print("Final realized value:", env.realized_value)