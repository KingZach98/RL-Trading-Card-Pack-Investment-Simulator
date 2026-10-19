from stable_baselines3 import PPO

from packfolio.environment.trading_env import TradingCardEnv


def train():
    env = TradingCardEnv(initial_boxes=100)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        gamma=1.0,
        verbose=1,
    )

    model.learn(
        total_timesteps=50_000
    )

    model.save("models/ppo_trading_card")

    env.close()


if __name__ == "__main__":
    train()