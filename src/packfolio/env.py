import gymnasium as gym
from gymnasium import spaces
import numpy as np

from packfolio.config import EnvironmentConfig, REGIME_ORDER, _integer, _positive_float
from packfolio.scenarios import Scenario, ScenarioSpec
from packfolio.types import Action, OBSERVATION_SIZE, PackOutcomeId, PortfolioSnapshot


class PackfolioEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}
    def __init__(self, config: EnvironmentConfig, *, inventory_capacity: int, reference_price: float) -> None:
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("config must be EnvironmentConfig")
        self._inventory_capacity = _integer(inventory_capacity, "inventory_capacity", minimum=1)
        self._reference_price = _positive_float(reference_price, "reference_price")
        if config.selling_fee_rate == 1:
            raise ValueError("selling_fee_rate must be less than 1")
        for outcome in config.pack.outcomes:
            PackOutcomeId(outcome.outcome_id)

        self._config = config
        self._scenario: Scenario | None = None
        self._portfolio: PortfolioSnapshot | None = None
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(low=np.zeros(OBSERVATION_SIZE, dtype=np.float32),high=np.array([np.inf, 1, np.inf, np.inf, 1, 1, 1, 1], dtype=np.float32),
                                            dtype=np.float32,)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dict or None")
        if options:
            raise ValueError("reset options are not supported")
        spec = ScenarioSpec(self._config.config_hash, seed) if seed is not None else None
        super().reset(seed=seed)
        if spec is None:
            episode_seed = int(self.np_random.integers(2**64, dtype=np.uint64))
            spec = ScenarioSpec(self._config.config_hash, episode_seed)
        self._scenario = Scenario(self._config, spec)
        self._portfolio = PortfolioSnapshot(cash=self._config.initial_cash, sealed_count=0)
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        if self._scenario is None or self._portfolio is None:
            raise RuntimeError("call reset() before building an observation")
        market = self._scenario.current_market
        remaining_steps = self._config.horizon - self._scenario.timestep
        with np.errstate(over="ignore"):
            observation = np.array([self._portfolio.cash / self._config.initial_cash, self._portfolio.sealed_count / self._inventory_capacity,
                                    market.pack_ask / self._reference_price, market.card_value_multiplier,remaining_steps / self._config.horizon,
                                    *(float(market.regime is regime) for regime in REGIME_ORDER),], dtype=np.float32)
        if not np.isfinite(observation).all():
            raise ValueError("observation values must be finite as float32")
        return observation


__all__ = ["PackfolioEnv"]
