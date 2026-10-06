from numbers import Integral

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from packfolio.config import EnvironmentConfig, REGIME_ORDER, _integer, _positive_float
from packfolio.portfolio import apply_action, liquidate, liquidation_value
from packfolio.scenarios import Scenario, ScenarioSpec
from packfolio.types import Action, MarketRegime, OBSERVATION_SIZE, PackOutcome, PackOutcomeId, PortfolioSnapshot, StepInfo, TerminationReason


DEFAULT_INVENTORY_CAPACITY = 10


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

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._scenario is None or self._portfolio is None:
            raise RuntimeError("call reset() before step()")
        if self._scenario.done:
            raise RuntimeError("episode has ended; call reset() before step()")
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        if not 0 <= action < len(Action):
            raise ValueError("action must be in 0..3")
        requested_action = Action(int(action))

        step_index = self._scenario.timestep
        market_before = self._scenario.current_market
        portfolio_before = self._portfolio
        selling_fee = self._config.selling_fee_rate
        value_before = liquidation_value(portfolio_before, market_before, selling_fee=selling_fee)

        feasible = True
        if requested_action is Action.BUY_PACK:
            feasible = (portfolio_before.cash >= market_before.pack_ask
                        and portfolio_before.sealed_count < self._inventory_capacity)
        elif requested_action in (Action.OPEN_AND_SELL, Action.SELL_PACK):
            feasible = portfolio_before.sealed_count > 0
        executed_action = requested_action if feasible else Action.HOLD

        pack_outcome = None
        if executed_action is Action.OPEN_AND_SELL:
            draw = self._scenario.open_pack()
            configured = next(outcome for outcome in self._config.pack.outcomes if outcome.outcome_id == draw.outcome_id)
            pack_outcome = PackOutcome(
                outcome_id=PackOutcomeId(draw.outcome_id),
                base_gross_value=configured.base_bundle_value,
                gross_value=draw.gross_bundle_value,
            )
        update = apply_action(
            portfolio_before, executed_action, market_before, selling_fee=selling_fee,
            inventory_capacity=self._inventory_capacity, pack_outcome=pack_outcome,
        )
        market_after = self._scenario.advance()
        self._portfolio = update.portfolio
        terminated = self._scenario.done
        fee_paid = update.fee_paid
        if terminated:
            terminal_update = liquidate(self._portfolio, market_after, selling_fee=selling_fee)
            self._portfolio = terminal_update.portfolio
            fee_paid += terminal_update.fee_paid
        value_after = liquidation_value(self._portfolio, market_after, selling_fee=selling_fee)

        # Reset starts with cash only, so initial_cash is the starting value.
        reward = (value_after - value_before) / self._config.initial_cash
        step_info = StepInfo(
            step_index=step_index,
            requested_action=requested_action,
            executed_action=executed_action,
            action_was_infeasible=not feasible,
            regime_before=market_before.regime,
            regime_after=market_after.regime,
            cash_before=portfolio_before.cash,
            cash_after=self._portfolio.cash,
            sealed_count_before=portfolio_before.sealed_count,
            sealed_count_after=self._portfolio.sealed_count,
            pack_outcome_id=pack_outcome.outcome_id if pack_outcome is not None else None,
            gross_opened_value=pack_outcome.gross_value if pack_outcome is not None else None,
            fee_paid=fee_paid,
            portfolio_value_before=value_before,
            portfolio_value_after=value_after,
            reward=reward,
            termination_reason=TerminationReason.HORIZON if terminated else None,
        )
        return self._get_observation(), reward, terminated, False, step_info.to_dict()

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


def build_environment(
    config: EnvironmentConfig,
    *,
    inventory_capacity: int = DEFAULT_INVENTORY_CAPACITY,
) -> PackfolioEnv:
    """Build a `PackfolioEnv` using the project's fixed reference-price convention.

    Training and evaluation must use identical preprocessing and environment
    construction for a reloaded checkpoint to reproduce its training-time
    behavior, so both call this helper instead of each computing their own
    ``reference_price``.
    """
    return PackfolioEnv(
        config,
        inventory_capacity=inventory_capacity,
        reference_price=config.market.quotes[MarketRegime.NORMAL].pack_ask,
    )


__all__ = ["DEFAULT_INVENTORY_CAPACITY", "PackfolioEnv", "build_environment"]
