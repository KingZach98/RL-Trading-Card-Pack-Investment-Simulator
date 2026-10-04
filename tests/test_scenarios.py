from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from packfolio.config import load_environment_config
from packfolio.scenarios import (
    SIMULATOR_VERSION,
    Scenario,
    ScenarioSpec,
    load_split_manifest,
    validate_split_manifests,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def config():
    return load_environment_config(CONFIG_DIR / "environment.json")


@pytest.fixture
def spec(config):
    return ScenarioSpec(config.config_hash, 1001)


def test_extra_openings_never_change_future_market_or_first_draw(config, spec):
    baseline = Scenario(config, spec)
    extra = Scenario(config, spec)
    for timestep in range(config.horizon):
        assert baseline.timestep == extra.timestep == timestep
        assert baseline.current_market == extra.current_market
        if timestep % 3 == 0:
            assert baseline.open_pack() == extra.open_pack()
        # Extra slots at some times, including times when the baseline skips.
        for _ in range(timestep % 5):
            extra.open_pack()
        assert baseline.advance() == extra.advance()
    assert baseline.done and extra.done


def test_same_timestep_and_opening_index_share_latent_draw(config, spec):
    first = Scenario(config, spec)
    second = Scenario(config, spec)
    for timestep in range(20):
        if timestep % 2 == 0:
            # Matching all slots, not just the first.
            for _ in range(4):
                assert first.open_pack() == second.open_pack()
        else:
            for _ in range(7):
                first.open_pack()
        first.advance()
        second.advance()
    assert first.open_pack() == second.open_pack()


def test_identity_and_version_replay_scenario(config, spec):
    recovered = ScenarioSpec.from_id(spec.scenario_id, SIMULATOR_VERSION)
    assert recovered == spec
    original = Scenario(config, spec)
    replay = Scenario(config, recovered)
    for _ in range(config.horizon):
        assert original.current_market == replay.current_market
        assert original.open_pack() == replay.open_pack()
        assert original.advance() == replay.advance()


def test_version_one_replay_golden_trace(config, spec):
    assert SIMULATOR_VERSION == "packfolio-scenarios-v1"
    assert config.config_hash == "57ce865515268c2e271ab34776c6095276a48e5cd71736399732092fe73d7330"
    scenario = Scenario(config, spec)
    trace = []
    for _ in range(8):
        trace.append((
            scenario.current_market.regime.value,
            scenario.open_pack().outcome_id,
            scenario.open_pack().outcome_id,
        ))
        scenario.advance()
    assert trace == [
        ("NORMAL", "base_bundle", "base_bundle"),
        ("NORMAL", "autograph_bundle", "base_bundle"),
        ("NORMAL", "base_bundle", "rookie_bundle"),
        ("HIGH", "base_bundle", "rookie_bundle"),
        ("HIGH", "base_bundle", "rookie_bundle"),
        ("HIGH", "base_bundle", "base_bundle"),
        ("HIGH", "base_bundle", "autograph_bundle"),
        ("NORMAL", "base_bundle", "base_bundle"),
    ]
    assert scenario.current_market.regime.value == "NORMAL"


def test_global_numpy_rng_does_not_affect_scenarios(config, spec):
    state = np.random.get_state()
    try:
        first = Scenario(config, spec)
        expected = [(first.open_pack(), first.advance()) for _ in range(10)]
        np.random.seed(999)
        np.random.random(200)
        second = Scenario(config, spec)
        assert [(second.open_pack(), second.advance()) for _ in range(10)] == expected
    finally:
        np.random.set_state(state)


def test_different_root_seeds_produce_different_traces(config, spec):
    first = Scenario(config, spec)
    second = Scenario(config, replace(spec, seed=1002))
    trace_a = [(first.open_pack(), first.advance()) for _ in range(50)]
    trace_b = [(second.open_pack(), second.advance()) for _ in range(50)]
    assert trace_a != trace_b


def test_only_current_information_is_public(config, spec):
    scenario = Scenario(config, spec)
    public_names = {name for name in dir(scenario) if not name.startswith("_")}
    assert public_names == {"timestep", "current_market", "done", "advance", "open_pack"}
    with pytest.raises(TypeError):
        scenario.open_pack(timestep=10)
    assert scenario.timestep == 0
    before = scenario.current_market
    scenario.open_pack()
    assert scenario.current_market == before


def test_horizon_and_terminal_operations(config, spec):
    short = replace(config, horizon=2)
    scenario = Scenario(short, ScenarioSpec(short.config_hash, spec.seed))
    assert not scenario.done
    for _ in range(2):
        scenario.open_pack()
        scenario.advance()
    assert scenario.done
    assert scenario.timestep == 2
    terminal = scenario.current_market
    with pytest.raises(RuntimeError, match="horizon"):
        scenario.open_pack()
    with pytest.raises(RuntimeError, match="horizon"):
        scenario.advance()
    assert scenario.current_market == terminal


def test_replay_rejects_changed_config(config, spec):
    with pytest.raises(ValueError, match="config_hash"):
        Scenario(replace(config, initial_cash=101.0), spec)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "1", 2**64])
def test_invalid_seeds(config, seed):
    with pytest.raises((TypeError, ValueError), match="seed"):
        ScenarioSpec(config.config_hash, seed)


def test_zero_and_maximum_seed_are_supported(config):
    for seed in (0, 2**64 - 1):
        spec = ScenarioSpec(config.config_hash, seed)
        assert ScenarioSpec.from_id(spec.scenario_id, SIMULATOR_VERSION) == spec
        assert Scenario(config, spec).open_pack().contents


@pytest.mark.parametrize("digest", ["", "abc", "A" * 64, "g" * 64, 1])
def test_invalid_config_hash(digest):
    with pytest.raises(ValueError, match="config_hash"):
        ScenarioSpec(digest, 1)


@pytest.mark.parametrize("identifier", ["bad", "a" * 64 + ":-1", "a" * 64 + ":01"])
def test_invalid_scenario_id(identifier):
    with pytest.raises(ValueError, match="scenario_id"):
        ScenarioSpec.from_id(identifier, SIMULATOR_VERSION)


def test_unknown_simulator_version_is_rejected(config, spec):
    with pytest.raises(ValueError, match="unsupported simulator_version"):
        replace(spec, simulator_version="future-version")
    with pytest.raises(ValueError, match="unsupported simulator_version"):
        ScenarioSpec.from_id(spec.scenario_id, "future-version")


def test_committed_splits_are_disjoint_and_have_expected_owners():
    manifests = tuple(
        load_split_manifest(CONFIG_DIR / "splits" / f"{split}.json")
        for split in ("training", "validation", "final_test")
    )
    validate_split_manifests(*manifests)
    assert [len(manifest.scenarios) for manifest in manifests] == [10, 3, 3]
    assert [manifest.owner for manifest in manifests] == ["Nasir", "Nasir", "Saki"]
    identities = {spec.scenario_id for manifest in manifests for spec in manifest.scenarios}
    assert len(identities) == 16


def test_split_validator_rejects_overlap_and_missing_split():
    manifests = [
        load_split_manifest(CONFIG_DIR / "splits" / f"{split}.json")
        for split in ("training", "validation", "final_test")
    ]
    with pytest.raises(ValueError, match="exactly one"):
        validate_split_manifests(*manifests[:2])
    with pytest.raises(ValueError, match="exactly one"):
        validate_split_manifests(manifests[0], manifests[0], manifests[2])
    overlapping = replace(manifests[1], scenarios=(manifests[0].scenarios[0],))
    with pytest.raises(ValueError, match="overlap"):
        validate_split_manifests(manifests[0], overlapping, manifests[2])
    # Seeds cannot overlap even when the config hashes differ.
    changed = replace(manifests[1].config, initial_cash=200.0)
    overlapping = replace(
        manifests[1], config=changed,
        scenarios=(ScenarioSpec(changed.config_hash, manifests[0].scenarios[0].seed),),
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_split_manifests(manifests[0], overlapping, manifests[2])


def test_duplicate_seeds_within_split_are_rejected():
    manifest = load_split_manifest(CONFIG_DIR / "splits" / "training.json")
    with pytest.raises(ValueError, match="duplicate"):
        replace(manifest, scenarios=(manifest.scenarios[0], manifest.scenarios[0]))
    with pytest.raises(ValueError, match="nonempty"):
        replace(manifest, scenarios=())


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("split", "unknown", "split"),
        ("owner", "", "owner"),
        ("environment_config", "", "environment_config"),
        ("config_hash", "0" * 64, "config_hash"),
        ("simulator_version", "unknown", "simulator_version"),
        ("seeds", [], "nonempty"),
        ("seeds", [True], "seed"),
        ("seeds", [1001, 1001], "duplicate"),
        ("seeds", "1001", "array"),
    ],
)
def test_invalid_manifest(tmp_path, field, value, message):
    data = json.loads((CONFIG_DIR / "splits" / "training.json").read_text(encoding="utf-8"))
    data["environment_config"] = str(CONFIG_DIR / "environment.json")
    data[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match=message):
        load_split_manifest(path)


def test_training_load_does_not_require_final_test_manifest(tmp_path):
    data = json.loads((CONFIG_DIR / "splits" / "training.json").read_text(encoding="utf-8"))
    data["environment_config"] = str(CONFIG_DIR / "environment.json")
    path = tmp_path / "training.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_split_manifest(path).split == "training"
    assert not (tmp_path / "final_test.json").exists()
