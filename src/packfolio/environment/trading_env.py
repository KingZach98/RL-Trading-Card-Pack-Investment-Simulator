import pandas as pd

from packfolio.simulator.box_sim import open_box, calculate_box_value
from data.market_data import get_card_market_factor
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

BOX_PRICE_PATH = "data/2020_panini_prizm_hobby_box_prices.csv"

KEEP = 0
OPEN = 1
SELL = 2


class TradingCardEnv(gym.Env):
    def __init__(self, initial_boxes=1):

        self.reward_scale = 100_000.0

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )


        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.initial_boxes = initial_boxes
        self.market_data = pd.read_csv(BOX_PRICE_PATH)
        self.market_data = self.market_data[
            self.market_data["date"] >= "2021-02"
        ].reset_index(drop=True)

        self.max_sealed_price = float(
                    self.market_data["price_usd"].max()
                )
        self.reset()

    def get_observation(self):
        state = self.get_state()

        time = self.current_step / (len(self.market_data) - 1)
        inventory = self.remaining_boxes / self.initial_boxes
        sealed_price = state["sealed_price"] / self.max_sealed_price
        card_market_factor = state["card_market_factor"]

        return np.array(
            [
                time,
                inventory,
                sealed_price,
                card_market_factor
            ],
            dtype=np.float32
        )

    def convert_action(self, action):
        action = np.asarray(action, dtype=np.float32)

        if action.shape != (3,):
            raise ValueError(
                "Action must contain [open_weight, sell_weight, keep_weight]."
            )

        weights = (action + 1.0) / 2.0

        total = weights.sum()

        if total == 0:
            return 0, 0

        proportions = weights / total

        open_count = int(
            np.floor(proportions[0] * self.remaining_boxes)
        )

        sell_count = int(
            np.floor(proportions[1] * self.remaining_boxes)
        )

        return open_count, sell_count

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.remaining_boxes = self.initial_boxes
        self.realized_value = 0.0
        self.done = False

        observation = self.get_observation()

        info = {
            "date": self.get_state()["date"],
            "remaining_boxes": self.remaining_boxes,
            "realized_value": self.realized_value,
        }

        return observation, info

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
        if self.done:
            raise RuntimeError("Episode has already finished.")

        open_count, sell_count = self.convert_action(action)

        state = self.get_state()

        date = state["date"]
        sealed_price = state["sealed_price"]

        keep_count = (
            self.remaining_boxes
            - open_count
            - sell_count
        )

        opened_value = 0.0

        for _ in range(open_count):
            box = open_box()
            opened_value += calculate_box_value(box, date)

        sold_value = sell_count * sealed_price

        # Reward in actual dollars
        dollar_reward = opened_value + sold_value

        self.realized_value += dollar_reward
        self.remaining_boxes = keep_count

        terminated = False
        truncated = False

        # No inventory remaining
        if self.remaining_boxes == 0:
            terminated = True
            self.done = True

            info = {
                "date": date,
                "opened_boxes": open_count,
                "sold_boxes": sell_count,
                "kept_boxes": 0,
                "opened_value": opened_value,
                "sold_value": sold_value,
                "realized_value": self.realized_value,
            }

            reward = dollar_reward / self.reward_scale

            return (
                np.zeros(4, dtype=np.float32),
                reward,
                terminated,
                truncated,
                info,
            )

        self.current_step += 1

        # End of historical data
        if self.current_step >= len(self.market_data):
            final_price = float(
                self.market_data.iloc[-1]["price_usd"]
            )

            liquidation_value = (
                self.remaining_boxes * final_price
            )

            # Add liquidation in DOLLARS first
            dollar_reward += liquidation_value
            self.realized_value += liquidation_value

            liquidated_boxes = self.remaining_boxes
            self.remaining_boxes = 0

            terminated = True
            self.done = True

            info = {
                "date": date,
                "opened_boxes": open_count,
                "sold_boxes": sell_count,
                "kept_boxes": 0,
                "liquidated_boxes": liquidated_boxes,
                "opened_value": opened_value,
                "sold_value": sold_value,
                "liquidation_value": liquidation_value,
                "realized_value": self.realized_value,
            }

            # Scale the ENTIRE reward
            reward = dollar_reward / self.reward_scale

            return (
                np.zeros(4, dtype=np.float32),
                reward,
                terminated,
                truncated,
                info,
            )

        observation = self.get_observation()

        info = {
            "date": date,
            "opened_boxes": open_count,
            "sold_boxes": sell_count,
            "kept_boxes": self.remaining_boxes,
            "opened_value": opened_value,
            "sold_value": sold_value,
            "realized_value": self.realized_value,
        }

        reward = dollar_reward / self.reward_scale

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

if __name__ == "__main__":
    env = TradingCardEnv(initial_boxes=100)

    check_env(env, warn=True)

    print("Environment check passed!")