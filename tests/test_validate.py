import json

import pytest

from packfolio.config import REGIME_ORDER, load_environment_config
from packfolio.validate import (
    Diagnostic,
    market_diagnostics,
    pack_diagnostics,
    probability_tolerance,
    run_validation,
)
from packfolio.types import MarketRegime


def test_probability_tolerance_uses_floor_and_sampling_error():
    assert probability_tolerance(0.01, 100_000) == pytest.approx(0.005)
    assert probability_tolerance(0.7, 50_000) > 0.005


def test_pack_diagnostics_are_reproducible_and_schema_like():
    config = load_environment_config("configs/environment.json")

    first = pack_diagnostics(config, sample_size=500, seed=123)
    second = pack_diagnostics(config, sample_size=500, seed=123)

    assert first == second
    assert {diagnostic.name for diagnostic in first} >= {
        "pack_frequency:base_bundle",
        "pack_mean_gross_value:multiplier_1",
    }
    assert all("passed" in diagnostic.to_dict() for diagnostic in first)


def test_market_diagnostics_are_reproducible_and_use_configured_matrix():
    config = load_environment_config("configs/environment.json")

    first = market_diagnostics(config, sample_size_per_regime=500, seed=456)
    second = market_diagnostics(config, sample_size_per_regime=500, seed=456)

    assert first == second
    normal_to_high = next(
        diagnostic
        for diagnostic in first
        if diagnostic.name == "market_transition:NORMAL->HIGH"
    )
    assert normal_to_high.expected == config.market.transition_matrix[
        MarketRegime.NORMAL
    ][REGIME_ORDER.index(MarketRegime.HIGH)]


def test_run_validation_summary_contains_identity_and_diagnostics():
    summary = run_validation(
        "configs/environment.json",
        pack_sample_size=1_000,
        market_sample_size_per_regime=1_000,
    )

    assert summary["config_hash"] == load_environment_config("configs/environment.json").config_hash
    assert summary["passed"] is True
    assert summary["diagnostics"]
    assert {
        "name",
        "sample_size",
        "expected",
        "observed",
        "tolerance",
        "passed",
    } <= set(summary["diagnostics"][0])


def test_failed_diagnostic_makes_summary_fail(monkeypatch):
    monkeypatch.setattr(
        "packfolio.validate.pack_diagnostics",
        lambda _config, sample_size: (Diagnostic("forced_failure", sample_size, 0.0, 1.0, 0.0),),
    )
    monkeypatch.setattr(
        "packfolio.validate.market_diagnostics",
        lambda _config, sample_size_per_regime: (),
    )

    summary = run_validation("configs/environment.json", pack_sample_size=10)

    assert summary["passed"] is False
    assert any(not diagnostic["passed"] for diagnostic in summary["diagnostics"])


def test_validation_summary_is_json_serializable():
    json.dumps(
        run_validation(
            "configs/environment.json",
            pack_sample_size=1_000,
            market_sample_size_per_regime=1_000,
        )
    )
