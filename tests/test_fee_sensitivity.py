"""PF-23 adapter parity, common randomness and independent accounting checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import csv
import gzip
import json
import shutil

from packfolio.baselines import AlwaysOpenPolicy, BuyAndHoldPolicy, CashOnlyPolicy
from packfolio.config import load_environment_config
from packfolio.env import build_environment
from packfolio.evaluate import EvaluationMetadata, evaluate_episode
from packfolio.fee_sensitivity import FeeSensitivityEnv
from packfolio.scenarios import Scenario, ScenarioSpec
from packfolio.types import Action, MarketRegime


@pytest.fixture
def base():
    return load_environment_config(Path(__file__).resolve().parents[1] / "configs/env_frozen.yaml")


@pytest.mark.parametrize("seed", [0, 1001, 2001, 2**64 - 1])
@pytest.mark.parametrize("script", [
    [Action.HOLD],
    [Action.BUY_PACK],
    [Action.BUY_PACK, Action.OPEN_AND_SELL],
    [Action.SELL_PACK, Action.OPEN_AND_SELL, Action.BUY_PACK, Action.SELL_PACK],
    [Action.BUY_PACK] * 12 + [Action.OPEN_AND_SELL, Action.SELL_PACK] * 6,
])
def test_exact_base_equivalence(base, seed, script):
    original, adapter = build_environment(base), FeeSensitivityEnv(base)
    expected, info = original.reset(seed=seed)
    actual, actual_info = adapter.reset(seed=seed)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype == np.float32
    assert actual_info == info
    for index in range(100):
        action = script[index % len(script)]
        expected = original.step(action)
        actual = adapter.step(action)
        assert actual[0].tobytes() == expected[0].tobytes()
        assert actual[1:] == expected[1:]
        assert actual[2] == (index == 99)
    with pytest.raises(RuntimeError, match="ended"):
        adapter.step(Action.HOLD)


@pytest.mark.parametrize("seed", [1001, 2001, 2002])
def test_cross_fee_pairing_despite_different_opening_histories(base, seed):
    envs = [FeeSensitivityEnv(replace(base, selling_fee_rate=f)) for f in (0.025, 0.05, 0.1)]
    recipe = Scenario(base, ScenarioSpec(base.config_hash, seed))
    for env in envs:
        env.reset(seed=seed)
    histories = [[], [], []]
    shared = 0
    for index in range(100):
        # All buy/open at steps 0/1 in each ten-step block; only the low-fee
        # policy makes additional openings. Other policies exercise HOLD/SELL.
        actions = [Action.HOLD] * 3
        if index % 10 == 0:
            actions = [Action.BUY_PACK] * 3
        elif index % 10 == 1:
            actions = [Action.OPEN_AND_SELL] * 3
        elif index % 10 in (2, 3):
            actions[0] = Action.BUY_PACK if index % 10 == 2 else Action.OPEN_AND_SELL
            actions[2] = Action.SELL_PACK
        market_before = recipe.current_market
        latent = recipe.open_pack()
        market_after = recipe.advance()
        infos = [env.step(action)[4] for env, action in zip(envs, actions)]
        for history, info in zip(histories, infos):
            history.append(info["executed_action"])
            assert info["regime_before"] == market_before.regime.value
            assert info["regime_after"] == market_after.regime.value
            if info["pack_outcome_id"] is not None:
                assert info["pack_outcome_id"] == latent.outcome_id
                assert info["gross_opened_value"] == latent.gross_bundle_value
        if all(info["pack_outcome_id"] is not None for info in infos):
            shared += 1
    assert shared > 0
    assert histories[0] != histories[1]
    assert {env.provenance()["randomness_source_scenario_id"] for env in envs} == {
        ScenarioSpec(base.config_hash, seed).scenario_id}
    assert len({env.provenance()["evaluation_scenario_id"] for env in envs}) == 3


@pytest.mark.parametrize("fee", [0.025, 0.05, 0.1])
def test_independent_accounting_and_terminal_liquidation(base, fee):
    env = FeeSensitivityEnv(replace(base, selling_fee_rate=fee))
    env.reset(seed=2001)
    cash, inventory, rewards = 100.0, 0, []
    for index in range(100):
        action = {0: Action.BUY_PACK, 1: Action.SELL_PACK, 2: Action.BUY_PACK,
                  3: Action.OPEN_AND_SELL, 98: Action.BUY_PACK,
                  99: Action.BUY_PACK}.get(index, Action.HOLD)
        obs, reward, terminated, truncated, info = env.step(action)
        ask = base.market.quotes[MarketRegime(info["regime_before"])].pack_ask
        terminal_ask = base.market.quotes[MarketRegime(info["regime_after"])].pack_ask
        before = cash + inventory * ask * (1 - fee)
        paid = 0.0
        assert not info["action_was_infeasible"]
        if action is Action.BUY_PACK:
            cash -= ask
            inventory += 1
        elif action in (Action.SELL_PACK, Action.OPEN_AND_SELL):
            gross = ask if action is Action.SELL_PACK else info["gross_opened_value"]
            cash += gross * (1 - fee)
            inventory -= 1
            paid = gross * fee
        if index == 99:
            assert inventory == 2
            cash += inventory * terminal_ask * (1 - fee)
            paid += inventory * terminal_ask * fee
            inventory = 0
        after = cash + inventory * terminal_ask * (1 - fee)
        assert info["cash_after"] == cash
        assert info["sealed_count_after"] == inventory
        assert info["fee_paid"] == paid
        assert info["portfolio_value_before"] == before
        assert info["portfolio_value_after"] == after
        assert reward == (after - before) / 100
        assert obs[0] == np.float32(cash / 100)
        assert terminated == (index == 99) and not truncated
        rewards.append(reward)
    assert sum(rewards) == pytest.approx(cash / 100 - 1)


@pytest.mark.parametrize("fee", [0.025, 0.05, 0.1])
def test_infeasible_requests_and_replay(base, fee):
    env = FeeSensitivityEnv(replace(base, selling_fee_rate=fee))
    traces = []
    for _ in range(2):
        env.reset(seed=2001)
        trace = []
        for action in [Action.SELL_PACK, Action.OPEN_AND_SELL] + [Action.BUY_PACK] * 20:
            trace.append(env.step(action)[4])
        assert all(t["executed_action"] == Action.HOLD for t in trace[:2])
        assert all(t["action_was_infeasible"] for t in trace[:2])
        assert any(t["action_was_infeasible"] for t in trace[2:])
        traces.append(trace)
    assert traces[0] == traces[1]
    assert env.provenance()["randomness_source_config_hash"] == base.config_hash


@pytest.mark.parametrize("field,value", [("horizon", 52), ("initial_cash", 101)])
def test_rejects_non_fee_scalar_changes(base, field, value):
    with pytest.raises(ValueError, match="only"):
        FeeSensitivityEnv(replace(base, **{field: value}))


def test_rejects_market_and_pack_changes(base):
    changed = [replace(base, market=replace(base.market, initial_regime=MarketRegime.LOW)),
               replace(base, pack=replace(base.pack, pack_id="different"))]
    for config in changed:
        with pytest.raises(ValueError, match="only"):
            FeeSensitivityEnv(config)


def test_common_evaluator_and_provenance(base):
    variant = replace(base, selling_fee_rate=0.1)
    env = FeeSensitivityEnv(variant)
    with pytest.raises(RuntimeError, match="reset"):
        env.provenance()
    metadata = EvaluationMetadata(simulator_version="fixture", policy_id="CASH_ONLY",
        model_id=None, training_seed=None, scenario_id=ScenarioSpec(variant.config_hash, 2001).scenario_id,
        scenario_seed=2001, config_hash=variant.config_hash, git_commit="fixture")
    result = evaluate_episode(CashOnlyPolicy(), env, metadata)
    assert result.row.final_portfolio_value == 100
    assert result.row.episode_steps == 100
    assert result.row.config_hash == variant.config_hash
    assert env.provenance()["evaluation_scenario_id"] == metadata.scenario_id
    assert env.provenance()["randomness_source_scenario_id"] != metadata.scenario_id


@pytest.mark.parametrize("policy_type", [BuyAndHoldPolicy, AlwaysOpenPolicy])
@pytest.mark.parametrize("seed", [2001, 2002])
def test_configured_baselines_match_pf22_at_five_percent(base, policy_type, seed):
    settings = {"initial_cash": base.initial_cash,
                "reference_price": base.market.quotes[MarketRegime.NORMAL].pack_ask}
    policy = policy_type(**settings)
    metadata = EvaluationMetadata(simulator_version="fixture", policy_id=policy.policy_id,
        model_id=None, training_seed=None, scenario_id=ScenarioSpec(base.config_hash, seed).scenario_id,
        scenario_seed=seed, config_hash=base.config_hash, git_commit="fixture")
    expected = evaluate_episode(policy_type(**settings), build_environment(base), metadata)
    actual = evaluate_episode(policy, FeeSensitivityEnv(base), metadata)
    assert actual == expected
    assert actual.step_traces[0].requested_action is Action.BUY_PACK
    assert actual.step_traces[0].executed_action is Action.BUY_PACK
    assert actual.row.requested_buy_pack_count > 0
    assert actual.row.executed_buy_pack_count > 0
    if policy_type is AlwaysOpenPolicy:
        assert actual.step_traces[1].requested_action is Action.OPEN_AND_SELL
        assert actual.step_traces[1].executed_action is Action.OPEN_AND_SELL
        assert actual.row.executed_open_and_sell_count > 0
    else:
        assert actual.row.executed_open_and_sell_count == 0


@pytest.mark.parametrize("action,error", [(True, TypeError), (1.5, TypeError), (-1, ValueError), (4, ValueError)])
def test_invalid_actions(base, action, error):
    env = FeeSensitivityEnv(base)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.HOLD)
    env.reset(seed=2001)
    with pytest.raises(error):
        env.step(action)


# Reuse PF-22's two-scenario, handcrafted-checkpoint fixtures without training.
from test_held_out_integration import inputs, saved
from packfolio import fee_sensitivity as fees


@pytest.fixture(scope="module")
def fee_smoke(tmp_path_factory, inputs, saved):
    root = tmp_path_factory.mktemp("pf23-smoke")
    before = {str(p): fees._digest(p) for p in saved.rglob("*") if p.is_file()}
    checkpoints = {str(p): fees._digest(p) for p in (inputs / "models").rglob("checkpoint.pt")}
    calls = []
    original = fees.evaluate_episode
    def spy(policy, env, metadata):
        result = original(policy, env, metadata)
        calls.append((env.provenance()["selling_fee_rate"], metadata.policy_id,
                      metadata.training_seed, metadata.scenario_seed))
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fees, "evaluate_episode", spy)
        summary = fees.run_fee_sensitivity(reference_directory=saved,
            split_manifest=inputs / "split.json",
            replica_manifest=inputs / "models/final_replicas_manifest.json",
            output_directory=root / "evidence", summary_file=root / "fee_sensitivity.csv",
            bootstrap_resamples=50)
    assert len(calls) == len(set(calls)) == 36
    assert {c[0] for c in calls} == {0.025, 0.1}
    assert all(fees._digest(p) == d for p,d in {**before, **checkpoints}.items())
    return root, summary


def test_smoke_has_27_summaries_and_36_unique_new_episodes(fee_smoke):
    root, summary = fee_smoke
    with summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 27
    with gzip.open(root / "evidence/episodes.jsonl.gz", "rt") as stream:
        records = [json.loads(line) for line in stream]
    keys = {(r["selling_fee_rate"],r["row"]["policy_id"],r["row"]["training_seed"],
             r["row"]["scenario_seed"]) for r in records}
    assert len(records) == len(keys) == 36
    assert {r["selling_fee_rate"] for r in records} == {0.025, 0.1}
    assert {r["row"]["scenario_seed"] for r in records} == {9101, 9102}
    for record in records:
        episode = record["row"]
        if episode["policy_id"] in ("BUY_AND_HOLD", "ALWAYS_OPEN"):
            assert episode["requested_buy_pack_count"] > 0
            assert episode["executed_buy_pack_count"] > 0
            assert record["steps"][0]["requested_action"] == Action.BUY_PACK
            assert record["steps"][0]["executed_action"] == Action.BUY_PACK
        if episode["policy_id"] == "ALWAYS_OPEN":
            assert episode["requested_open_and_sell_count"] > 0
            assert episode["executed_open_and_sell_count"] > 0
            assert record["steps"][1]["requested_action"] == Action.OPEN_AND_SELL
            assert record["steps"][1]["executed_action"] == Action.OPEN_AND_SELL
    for row in rows:
        assert row["scenario_count"] == "2"
        if row["policy_id"] == "EV_ONE_STEP":
            assert row["information_class"] == "model-informed"
        if row["policy_id"] == "DQN":
            assert row["policy_status"] == "frozen weights; no exploration"
        if float(row["selling_fee_rate"]) == 0.05:
            assert float(row["paired_mean_change_vs_5pct"]) == 0
            assert float(row["paired_change_ci_low"]) == float(row["paired_change_ci_high"]) == 0
        if row["policy_id"] == "CASH_ONLY":
            assert float(row["mean_final_value"]) == 100
            assert float(row["paired_mean_change_vs_5pct"]) == 0


def test_summary_is_computed_from_saved_episode_values(fee_smoke, saved):
    root, summary = fee_smoke
    groups = {}
    manifest = fees._json(saved / "held_out_manifest.json")
    for group in manifest["groups"]:
        for row in fees.read_evaluation_rows_jsonl(saved / group["rows_file"]):
            groups.setdefault((0.05, row.policy_id, row.training_seed), {})[row.scenario_seed] = row.final_portfolio_value
    with gzip.open(root / "evidence/episodes.jsonl.gz", "rt") as stream:
        for line in stream:
            record = json.loads(line); row = record["row"]
            groups.setdefault((record["selling_fee_rate"],row["policy_id"],row["training_seed"]), {})[row["scenario_seed"]] = row["final_portfolio_value"]
    with summary.open(newline="") as stream:
        for row in csv.DictReader(stream):
            seed = int(row["training_seed"]) if row["training_seed"] else None
            values = groups[float(row["selling_fee_rate"]),row["policy_id"],seed]
            base_values = groups[0.05,row["policy_id"],seed]
            assert float(row["mean_final_value"]) == pytest.approx(sum(values.values()) / 2)
            assert float(row["paired_mean_change_vs_5pct"]) == pytest.approx(
                sum(values[s]-base_values[s] for s in values) / 2)


def test_aggregation_is_deterministic_and_never_runs_policies(fee_smoke, monkeypatch):
    root, summary = fee_smoke
    before = summary.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("aggregation must not run policies")
    monkeypatch.setattr(fees, "evaluate_episode", forbidden)
    fees.aggregate_fee_sensitivity(root / "evidence")
    assert summary.read_bytes() == before


@pytest.mark.parametrize("problem", ["missing", "duplicate", "provenance", "checksum", "cash"])
def test_invalid_saved_variant_evidence_is_rejected(fee_smoke, tmp_path, problem):
    root, _ = fee_smoke
    output = Path(shutil.copytree(root / "evidence", tmp_path / "evidence"))
    path = output / "episodes.jsonl.gz"
    with gzip.open(path, "rt") as stream:
        records = [json.loads(line) for line in stream]
    if problem == "missing": records.pop()
    elif problem == "duplicate": records.append(records[0])
    elif problem == "provenance": records[0]["row"]["config_hash"] = "0" * 64
    elif problem == "cash": records[0]["steps"][0]["cash_after"] += 1
    else: records.pop()
    with gzip.open(path, "wt") as stream:
        for record in records: stream.write(json.dumps(record) + "\n")
    manifest = fees._json(output / "manifest.json")
    manifest["summary_file"] = str(tmp_path / "summary.csv")
    if problem != "checksum": manifest["episodes_sha256"] = fees._digest(path)
    fees._write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError): fees.aggregate_fee_sensitivity(output)
    assert not (tmp_path / "summary.csv").exists()


def test_reference_validation_is_read_only_and_rejects_missing_rows(saved, inputs, tmp_path):
    reference = Path(shutil.copytree(saved, tmp_path / "reference"))
    manifest = fees._json(reference / "held_out_manifest.json")
    row_path = reference / manifest["groups"][0]["rows_file"]
    row_path.write_text(row_path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="missing"):
        fees.validate_reference(reference, inputs / "split.json", inputs / "models/final_replicas_manifest.json")


def test_variant_configuration_checked_before_outputs(saved, inputs, tmp_path, monkeypatch):
    original = fees.load_environment_config
    def changed(path):
        config = original(path)
        return replace(config, initial_cash=101) if str(path).endswith("env_frozen_fee_high.yaml") else config
    monkeypatch.setattr(fees, "load_environment_config", changed)
    with pytest.raises(ValueError, match="only"):
        fees.run_fee_sensitivity(reference_directory=saved, split_manifest=inputs / "split.json",
            replica_manifest=inputs / "models/final_replicas_manifest.json",
            output_directory=tmp_path / "evidence", summary_file=tmp_path / "summary.csv")
    assert not (tmp_path / "evidence").exists()
