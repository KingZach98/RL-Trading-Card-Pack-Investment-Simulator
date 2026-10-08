"""Model-informed, myopic EV policy from Project Bible pages 13 and 26."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from math import fsum, isfinite
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, TYPE_CHECKING

import numpy as np

from packfolio.baselines import _can_buy, _check_observation, _has_sealed_pack
from packfolio.config import EnvironmentConfig, REGIME_ORDER, _integer, _positive_float
from packfolio.env import DEFAULT_INVENTORY_CAPACITY
from packfolio.market import snapshot_for
from packfolio.packs import expected_gross_value
from packfolio.types import Action, MarketRegime, ObservationIndex

if TYPE_CHECKING:
    from packfolio.evaluate import EvaluationMetadata, EvaluationResult


TIE_ORDER = (Action.HOLD, Action.SELL_PACK, Action.OPEN_AND_SELL, Action.BUY_PACK)


@dataclass(frozen=True, slots=True)
class _MarketTerms:
    ask: float
    sealed_net: float
    opening_net: float
    next_sealed_net: float


@dataclass(frozen=True, slots=True)
class EVOneStepPolicy:
    """Static model knowledge plus public observations; never a scenario or RNG.

    Scores use the configured regime quotes and float32-decoded cash/inventory.
    BUY eligibility deliberately reuses PF-12's rounding-interval convention.
    """

    config: EnvironmentConfig
    inventory_capacity: int = DEFAULT_INVENTORY_CAPACITY
    reference_price: float | None = None
    _terms: Mapping[MarketRegime, _MarketTerms] = field(init=False, repr=False)

    policy_id: ClassVar[str] = "EV_ONE_STEP"
    information_class: ClassVar[str] = "model-informed"

    def __post_init__(self) -> None:
        if not isinstance(self.config, EnvironmentConfig):
            raise TypeError("config must be EnvironmentConfig")
        _integer(self.inventory_capacity, "inventory_capacity", minimum=1)
        price = self.reference_price
        if price is None:
            price = self.config.market.quotes[MarketRegime.NORMAL].pack_ask
        object.__setattr__(self, "reference_price", _positive_float(price, "reference_price"))
        if not 0 <= self.config.selling_fee_rate < 1:
            raise ValueError("selling_fee_rate must be in [0, 1)")

        net_fraction = 1 - self.config.selling_fee_rate
        snapshots = {regime: snapshot_for(regime, self.config.market) for regime in REGIME_ORDER}
        terms = {}
        for regime in REGIME_ORDER:
            market = snapshots[regime]
            next_net = fsum(
                probability * (net_fraction * snapshots[target].pack_ask)
                for probability, target in zip(
                    self.config.market.transition_matrix[regime], REGIME_ORDER, strict=True
                )
            )
            opening_net = net_fraction * expected_gross_value(
                self.config.pack, market.card_value_multiplier
            )
            if not isfinite(next_net) or not isfinite(opening_net):
                raise ValueError("expected market values must be finite")
            terms[regime] = _MarketTerms(
                market.pack_ask, net_fraction * market.pack_ask, opening_net, next_net
            )
        object.__setattr__(self, "_terms", MappingProxyType(terms))

    def _decode(self, observation: np.ndarray) -> tuple[float, int, MarketRegime]:
        _check_observation(observation)
        if not np.isfinite(observation).all() or (observation < 0).any():
            raise ValueError("observation must be finite and nonnegative")
        inventory_ratio = observation[ObservationIndex.SEALED_COUNT_RATIO]
        if inventory_ratio > 1 or observation[ObservationIndex.REMAINING_STEPS_RATIO] > 1:
            raise ValueError("inventory and remaining-step ratios must be in [0, 1]")
        regime_features = observation[ObservationIndex.REGIME_LOW:]
        if not np.isin(regime_features, (0, 1)).all() or regime_features.sum() != 1:
            raise ValueError("regime features must be one-hot")
        regime = REGIME_ORDER[int(np.argmax(regime_features))]
        quote = self.config.market.quotes[regime]
        with np.errstate(over="ignore"):
            expected_ask = np.float32(quote.pack_ask / self.reference_price)
            expected_multiplier = np.float32(quote.card_value_multiplier)
        if (observation[ObservationIndex.PACK_ASK_RATIO] != expected_ask
                or observation[ObservationIndex.CARD_VALUE_MULTIPLIER] != expected_multiplier):
            raise ValueError("observation market features disagree with policy configuration")
        count = round(float(inventory_ratio) * self.inventory_capacity)
        if np.float32(count / self.inventory_capacity) != inventory_ratio:
            raise ValueError("inventory ratio does not represent a sealed-pack count")
        cash = float(observation[ObservationIndex.CASH_RATIO]) * self.config.initial_cash
        if not isfinite(cash):
            raise ValueError("decoded cash must be finite")
        return cash, count, regime

    def _scores_and_advantages(
        self, observation: np.ndarray
    ) -> tuple[dict[Action, float], dict[Action, float]]:
        cash, count, regime = self._decode(observation)
        terms = self._terms[regime]
        feasible = {Action.HOLD}
        if _has_sealed_pack(observation):
            feasible.update((Action.SELL_PACK, Action.OPEN_AND_SELL))
        if _can_buy(observation, initial_cash=self.config.initial_cash,
                    reference_price=self.reference_price):
            feasible.add(Action.BUY_PACK)

        scores = {}
        advantages = {}
        for action in TIE_ORDER:
            if action not in feasible:
                continue
            if action is Action.HOLD:
                score = fsum((cash, count * terms.next_sealed_net))
                advantage = 0.0
            elif action is Action.BUY_PACK:
                score = fsum((cash, -terms.ask, (count + 1) * terms.next_sealed_net))
                advantage = terms.next_sealed_net - terms.ask
            else:
                proceeds = terms.sealed_net if action is Action.SELL_PACK else terms.opening_net
                score = fsum((cash, proceeds, (count - 1) * terms.next_sealed_net))
                advantage = proceeds - terms.next_sealed_net
            if not isfinite(score):
                raise ValueError("action scores must be finite")
            scores[action] = score
            advantages[action] = advantage
        return scores, advantages

    def score_actions(self, observation: np.ndarray) -> dict[Action, float]:
        """Return the Bible's absolute expected-next-value scores for eligible actions."""
        return self._scores_and_advantages(observation)[0]

    def choose_action(self, observation: np.ndarray) -> Action:
        _, advantages = self._scores_and_advantages(observation)
        # C + N*mu cancels in every comparison; avoid losing small gains in a
        # large common cash balance. Dict insertion order implements exact ties.
        return max(advantages, key=advantages.__getitem__)


def evaluate_ev_episode(
    policy: EVOneStepPolicy,
    env: object,
    metadata: EvaluationMetadata,
    *,
    output_directory: str | Path,
) -> EvaluationResult:
    """Use the common runner and writers, with a separate model-informed label."""
    from packfolio.evaluate import (
        EvaluationMetadata, evaluate_episode, write_evaluation_rows_jsonl,
        write_step_traces_jsonl,
    )

    if not isinstance(policy, EVOneStepPolicy):
        raise TypeError("policy must be EVOneStepPolicy")
    if not isinstance(metadata, EvaluationMetadata):
        raise TypeError("metadata must be EvaluationMetadata")
    if metadata.policy_id != policy.policy_id:
        raise ValueError("metadata policy_id must be EV_ONE_STEP")
    if metadata.config_hash != policy.config.config_hash:
        raise ValueError("metadata config_hash must match policy configuration")
    if metadata.model_id is not None or metadata.training_seed is not None:
        raise ValueError("EV_ONE_STEP has no trained model or training seed")

    result = evaluate_episode(policy, env, metadata)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_evaluation_rows_jsonl(output / "evaluation_rows.jsonl", (result.row,))
    write_step_traces_jsonl(output / "step_traces.jsonl", result.step_traces)
    classification = {
        "metadata_version": 1,
        "policy_id": policy.policy_id,
        "information_class": policy.information_class,
        "config_hash": metadata.config_hash,
        "scenario_id": metadata.scenario_id,
        "inventory_capacity": policy.inventory_capacity,
        "observation_reference_price": policy.reference_price,
        "tie_order": [action.name for action in TIE_ORDER],
        "feasibility_convention": "PF-12 float32 rounding intervals",
        "evaluation_rows_file": "evaluation_rows.jsonl",
        "step_traces_file": "step_traces.jsonl",
    }
    with (output / "policy_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(classification, metadata_file, indent=2, sort_keys=True, allow_nan=False)
        metadata_file.write("\n")
    return result


__all__ = ["EVOneStepPolicy", "TIE_ORDER", "evaluate_ev_episode"]
