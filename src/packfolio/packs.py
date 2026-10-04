"""Configuration-driven, whole-pack outcomes with no fees or cash accounting."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROBABILITY_TOLERANCE = 1e-9


def _nonnegative_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite, nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, nonnegative number")


@dataclass(frozen=True)
class PackOutcome:
    """One possible complete pack; base_bundle_value is not a per-card value."""

    outcome_id: str
    probability: float
    base_bundle_value: float
    contents: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outcome_id, str) or not self.outcome_id.strip():
            raise ValueError("outcome_id must be a nonempty string")
        _nonnegative_finite(self.probability, "probability")
        _nonnegative_finite(self.base_bundle_value, "base_bundle_value")
        if not isinstance(self.contents, tuple) or not self.contents:
            raise ValueError("contents must be a nonempty tuple of card descriptions")
        if any(not isinstance(card, str) or not card.strip() for card in self.contents):
            raise ValueError("each card description must be a nonempty string")


@dataclass(frozen=True)
class PackConfig:
    """A categorical distribution over complete packs of the same size."""

    pack_id: str
    cards_per_pack: int
    outcomes: tuple[PackOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pack_id, str) or not self.pack_id.strip():
            raise ValueError("pack_id must be a nonempty string")
        if (
            isinstance(self.cards_per_pack, bool)
            or not isinstance(self.cards_per_pack, int)
            or self.cards_per_pack <= 0
        ):
            raise ValueError("cards_per_pack must be a positive integer")
        if not isinstance(self.outcomes, tuple) or not self.outcomes:
            raise ValueError("outcomes must be a nonempty tuple of PackOutcome objects")
        if any(not isinstance(outcome, PackOutcome) for outcome in self.outcomes):
            raise ValueError("outcomes must contain only PackOutcome objects")
        identifiers = [outcome.outcome_id for outcome in self.outcomes]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("outcome_id values must be unique")
        if any(len(outcome.contents) != self.cards_per_pack for outcome in self.outcomes):
            raise ValueError("every outcome must contain exactly cards_per_pack cards")
        # Bounding individual probabilities also prevents overflow in the sum.
        if any(outcome.probability > 1 + PROBABILITY_TOLERANCE for outcome in self.outcomes):
            raise ValueError("probabilities must sum to one within absolute tolerance 1e-9")
        total = math.fsum(outcome.probability for outcome in self.outcomes)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE):
            raise ValueError("probabilities must sum to one within absolute tolerance 1e-9")

    @property
    def probabilities(self) -> tuple[float, ...]:
        """Normalize accepted rounding error identically for sampling and EV."""
        total = math.fsum(outcome.probability for outcome in self.outcomes)
        return tuple(outcome.probability / total for outcome in self.outcomes)


@dataclass(frozen=True)
class PackDraw:
    """The full contents and gross market value of one sampled pack."""

    outcome_id: str
    gross_bundle_value: float
    contents: tuple[str, ...]


def load_pack_config(path: str | Path) -> PackConfig:
    """Read a JSON pack distribution, raising on invalid or incomplete data."""
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("pack configuration must be a JSON object")
    for key in ("pack_id", "cards_per_pack", "outcomes"):
        if key not in data:
            raise ValueError(f"pack configuration is missing {key}")
    if not isinstance(data["outcomes"], list):
        raise ValueError("outcomes must be a JSON array")
    outcomes = []
    for entry in data["outcomes"]:
        if not isinstance(entry, dict):
            raise ValueError("each outcome must be a JSON object")
        for key in ("outcome_id", "probability", "base_bundle_value", "contents"):
            if key not in entry:
                raise ValueError(f"outcome is missing {key}")
        if not isinstance(entry["contents"], list):
            raise ValueError("contents must be a JSON array of card descriptions")
        outcomes.append(
            PackOutcome(
                outcome_id=entry["outcome_id"],
                probability=entry["probability"],
                base_bundle_value=entry["base_bundle_value"],
                contents=tuple(entry["contents"]),
            )
        )
    return PackConfig(
        pack_id=data["pack_id"],
        cards_per_pack=data["cards_per_pack"],
        outcomes=tuple(outcomes),
    )


def _gross_value(base_bundle_value: float, market_multiplier: float) -> float:
    gross = base_bundle_value * market_multiplier
    if not math.isfinite(gross):
        raise ValueError("gross bundle value must be finite")
    return gross


def sample_pack(
    config: PackConfig, market_multiplier: float, rng: np.random.Generator
) -> PackDraw:
    """Draw one complete pack using only the supplied generator's state."""
    _nonnegative_finite(market_multiplier, "market_multiplier")
    # Validate all possible payouts before advancing the caller's generator.
    values = tuple(
        _gross_value(outcome.base_bundle_value, market_multiplier)
        for outcome in config.outcomes
    )
    index = int(rng.choice(len(config.outcomes), p=config.probabilities))
    outcome = config.outcomes[index]
    return PackDraw(outcome.outcome_id, values[index], outcome.contents)


def expected_gross_value(config: PackConfig, market_multiplier: float) -> float:
    """Analytical whole-pack EV at the current multiplier, before any fees."""
    _nonnegative_finite(market_multiplier, "market_multiplier")
    return math.fsum(
        probability * _gross_value(outcome.base_bundle_value, market_multiplier)
        for probability, outcome in zip(config.probabilities, config.outcomes, strict=True)
    )