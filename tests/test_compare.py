"""PF-19 input guards and numerical accounting checks."""

from pathlib import Path

import pytest

from packfolio import compare


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "configs/splits/validation.json"


def test_development_manifest_uses_approved_seeds_without_changing_config():
    split = compare._development_split(VALIDATION)
    assert [spec.seed for spec in split.scenarios] == [2001, 2002, 2003]
    assert split.config.horizon == 100
    assert split.config.initial_cash == 100
    assert split.config.selling_fee_rate == 0.05


@pytest.mark.parametrize("split", ["training", "final_test"])
def test_non_development_split_is_rejected_before_creating_outputs(tmp_path, split):
    output = tmp_path / "outputs" / "comparison"
    with pytest.raises(ValueError, match="development validation"):
        compare.run_comparison(split_manifest=ROOT / f"configs/splits/{split}.json",
                               output_directory=output)
    assert not output.exists()


def test_artifacts_cannot_be_written_outside_outputs(tmp_path):
    output = tmp_path / "comparison"
    with pytest.raises(ValueError, match="outputs/"):
        compare.run_comparison(split_manifest=VALIDATION, output_directory=output)
    assert not output.exists()


def test_existing_evidence_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "OUTPUT_ROOT", tmp_path)
    output = tmp_path / "comparison"
    output.mkdir()
    evidence = output / "sentinel.json"
    evidence.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite"):
        compare.run_comparison(split_manifest=VALIDATION, output_directory=output)
    assert evidence.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("actual", [1.1, float("nan"), float("inf")])
def test_accounting_check_rejects_differences_and_nonfinite_values(actual):
    with pytest.raises(ValueError, match="trace"):
        compare._close(actual, 1.0, "fixture")


def test_accounting_check_allows_only_roundoff():
    compare._close(0.1 + 0.2, 0.3, "fixture")
