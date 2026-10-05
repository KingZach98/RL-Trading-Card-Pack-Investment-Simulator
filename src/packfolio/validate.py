from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from packfolio.config import REGIME_ORDER, EnvironmentConfig, load_environment_config
from packfolio.market import advance_market
from packfolio.packs import expected_gross_value, sample_pack


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "environment.json"
PACK_SAMPLE_SIZE = 100_000
MARKET_SAMPLE_SIZE_PER_REGIME = 50_000
PACK_SEED = 70_701
MARKET_SEED = 70_702
SIGMA_MULTIPLIER = 5.0
MIN_PROBABILITY_TOLERANCE = 0.005
MIN_MEAN_TOLERANCE = 0.1


@dataclass(frozen=True, slots=True)
class Diagnostic:
    name: str
    sample_size: int
    expected: float
    observed: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return abs(self.observed - self.expected) <= self.tolerance

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sample_size": self.sample_size,
            "expected": self.expected,
            "observed": self.observed,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def probability_tolerance(probability: float, sample_size: int) -> float:
    standard_error = math.sqrt(probability * (1.0 - probability) / sample_size)
    return max(MIN_PROBABILITY_TOLERANCE, SIGMA_MULTIPLIER * standard_error)


def mean_tolerance(config: EnvironmentConfig, market_multiplier: float, sample_size: int) -> float:
    mean = expected_gross_value(config.pack, market_multiplier)
    variance = math.fsum(
        probability * (outcome.base_bundle_value * market_multiplier - mean) ** 2
        for probability, outcome in zip(config.pack.probabilities, config.pack.outcomes, strict=True)
    )
    standard_error = math.sqrt(variance / sample_size)
    return max(MIN_MEAN_TOLERANCE, SIGMA_MULTIPLIER * standard_error)


def pack_diagnostics(
    config: EnvironmentConfig,
    *,
    sample_size: int = PACK_SAMPLE_SIZE,
    seed: int = PACK_SEED,
    market_multiplier: float = 1.0,
) -> tuple[Diagnostic, ...]:
    rng = np.random.default_rng(seed)
    counts = {outcome.outcome_id: 0 for outcome in config.pack.outcomes}
    total_value = 0.0

    for _ in range(sample_size):
        draw = sample_pack(config.pack, market_multiplier, rng)
        counts[draw.outcome_id] += 1
        total_value += draw.gross_bundle_value

    diagnostics: list[Diagnostic] = []
    for outcome, expected in zip(config.pack.outcomes, config.pack.probabilities, strict=True):
        diagnostics.append(
            Diagnostic(
                name=f"pack_frequency:{outcome.outcome_id}",
                sample_size=sample_size,
                expected=expected,
                observed=counts[outcome.outcome_id] / sample_size,
                tolerance=probability_tolerance(expected, sample_size),
            )
        )

    diagnostics.append(
        Diagnostic(
            name=f"pack_mean_gross_value:multiplier_{market_multiplier:g}",
            sample_size=sample_size,
            expected=expected_gross_value(config.pack, market_multiplier),
            observed=total_value / sample_size,
            tolerance=mean_tolerance(config, market_multiplier, sample_size),
        )
    )
    return tuple(diagnostics)


def market_diagnostics(
    config: EnvironmentConfig,
    *,
    sample_size_per_regime: int = MARKET_SAMPLE_SIZE_PER_REGIME,
    seed: int = MARKET_SEED,
) -> tuple[Diagnostic, ...]:
    rng = np.random.default_rng(seed)
    diagnostics: list[Diagnostic] = []

    for current_regime in REGIME_ORDER:
        counts = {regime: 0 for regime in REGIME_ORDER}
        for _ in range(sample_size_per_regime):
            next_snapshot = advance_market(current_regime, rng, config.market)
            counts[next_snapshot.regime] += 1
        for next_regime, expected in zip(
            REGIME_ORDER,
            config.market.transition_matrix[current_regime],
            strict=True,
        ):
            diagnostics.append(
                Diagnostic(
                    name=f"market_transition:{current_regime.value}->{next_regime.value}",
                    sample_size=sample_size_per_regime,
                    expected=expected,
                    observed=counts[next_regime] / sample_size_per_regime,
                    tolerance=probability_tolerance(expected, sample_size_per_regime),
                )
            )
    return tuple(diagnostics)


def run_validation(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    pack_sample_size: int = PACK_SAMPLE_SIZE,
    market_sample_size_per_regime: int = MARKET_SAMPLE_SIZE_PER_REGIME,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_environment_config(config_path)
    diagnostics = (
        *pack_diagnostics(config, sample_size=pack_sample_size),
        *market_diagnostics(config, sample_size_per_regime=market_sample_size_per_regime),
    )
    return {
        "config_path": str(config_path),
        "config_hash": config.config_hash,
        "pack_sample_size": pack_sample_size,
        "market_sample_size_per_regime": market_sample_size_per_regime,
        "tolerance_rule": (
            "probability tolerance = max(0.005, 5 * binomial standard error); "
            "mean tolerance = max(0.1, 5 * standard error)"
        ),
        "passed": all(diagnostic.passed for diagnostic in diagnostics),
        "diagnostics": [diagnostic.to_dict() for diagnostic in diagnostics],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Packfolio simulator diagnostics.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to environment JSON configuration.",
    )
    args = parser.parse_args(argv)

    summary = run_validation(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
