"""PF-22: frozen-model evaluation and saved-record uncertainty summaries."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from math import fsum, isclose, isfinite
from pathlib import Path

import numpy as np

from packfolio.agents.dqn_agent import load_checkpoint_metadata, load_trained_agent
from packfolio.baselines import AlwaysOpenPolicy, BuyAndHoldPolicy, CashOnlyPolicy
from packfolio.config import load_environment_config
from packfolio.env import DEFAULT_INVENTORY_CAPACITY, build_environment
from packfolio.ev_baseline import EVOneStepPolicy, evaluate_ev_episode
from packfolio.evaluate import (
    EvaluationMetadata, evaluate_checkpoint_on_split, evaluate_episode,
    read_evaluation_rows_jsonl, read_step_traces_jsonl,
    write_evaluation_rows_jsonl, write_step_traces_jsonl,
)
from packfolio.market import snapshot_for
from packfolio.portfolio import apply_action, liquidate, liquidation_value
from packfolio.provenance import git_metadata
from packfolio.scenarios import SIMULATOR_VERSION, load_split_manifest
from packfolio.train import load_agent_config
from packfolio.tune import FINAL_REPLICA_SEEDS
from packfolio.types import Action, MarketRegime, PackOutcome, PortfolioSnapshot, TerminationReason


ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("CASH_ONLY", "BUY_AND_HOLD", "ALWAYS_OPEN", "EV_ONE_STEP")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def bootstrap_mean_interval(values, baseline_values=None, *, seed=22, resamples=2000,
                            confidence=0.95):
    """Percentile CI for a fixed model; subtract paired baseline before resampling."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires at least two finite scenario values")
    if baseline_values is not None:
        baseline = np.asarray(baseline_values, dtype=np.float64)
        if baseline.shape != values.shape or not np.isfinite(baseline).all():
            raise ValueError("baseline values must match paired scenarios")
        values = values - baseline
    if type(seed) is not int or seed < 0 or type(resamples) is not int or resamples < 1:
        raise ValueError("bootstrap seed/resamples must be nonnegative/positive integers")
    if isinstance(confidence, bool) or not isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    bounds = np.quantile(means, [tail, 1 - tail], method="linear")
    return float(bounds[0]), float(bounds[1])


def summarize_rows(rows, *, loss_tolerance=1e-9):
    """Per-model or baseline metrics; do not pool replicas as independent markets."""
    if not rows or not isfinite(loss_tolerance) or loss_tolerance < 0:
        raise ValueError("rows and a finite nonnegative loss tolerance are required")
    identities = {(r.policy_id, r.model_id, r.training_seed, r.config_hash) for r in rows}
    if len(identities) != 1 or len({r.scenario_id for r in rows}) != len(rows):
        raise ValueError("summary requires one model and unique scenarios")
    values = np.array([r.final_portfolio_value for r in rows])
    steps = sum(r.episode_steps for r in rows)
    result = {
        "scenario_count": len(rows), "mean_final_value": float(values.mean()),
        "median_final_value": float(np.median(values)),
        "sample_std_final_value": float(values.std(ddof=1)) if len(rows) > 1 else None,
        "loss_probability": sum(r.final_portfolio_value < r.initial_portfolio_value - loss_tolerance
                                for r in rows) / len(rows),
        "fifth_percentile_final_value": float(np.quantile(values, 0.05, method="linear")),
        "infeasible_action_rate": sum(r.infeasible_action_count for r in rows) / steps,
    }
    for kind in ("requested", "executed"):
        for action in Action:
            name = f"{kind}_{action.name.lower()}"
            result[name + "_rate"] = sum(getattr(r, name + "_count") for r in rows) / steps
    return result


def _preflight(split_path, replica_path, frozen_path, selected_path):
    split = load_split_manifest(split_path)
    config = load_environment_config(frozen_path)
    if split.split != "final_test" or split.config.config_hash != config.config_hash:
        raise ValueError("held-out split must be final_test and match the frozen configuration")
    if config.horizon != 100:
        raise ValueError("PF-22 preserves the 100-step horizon")
    if len(split.scenarios) < 2:
        raise ValueError("uncertainty summaries require at least two scenarios")
    selected = load_agent_config(selected_path)
    if selected.total_steps != 2000:
        raise ValueError("PF-22 preserves the 2000-step model budget")
    manifest = _json(replica_path)
    replicas = manifest["replicas"]
    seeds = [r["seed"] for r in replicas]
    if (manifest["final_replicas_version"] != 1 or len(seeds) != 5
            or set(seeds) != set(FINAL_REPLICA_SEEDS)
            or manifest["final_replica_seeds"] != list(FINAL_REPLICA_SEEDS)):
        raise ValueError("require all five unique final replica seeds")
    models = []
    sources = [split_path, replica_path, frozen_path, selected_path]
    for replica in sorted(replicas, key=lambda r: r["seed"]):
        run_name = Path(replica["run_directory"])
        if replica["status"] != "success" or len(run_name.parts) != 1 or run_name.name in (".", ".."):
            raise ValueError("require successful replicas with relative run directory names")
        directory = replica_path.parent / run_name
        checkpoint = directory / "checkpoint.pt"
        run_path = directory / "run_manifest.json"
        agent_path = directory / "agent_config.yaml"
        run = _json(run_path)
        agent = load_agent_config(agent_path)
        metadata = load_checkpoint_metadata(checkpoint)
        if (metadata.training_seed != replica["seed"] or run["training_seed"] != replica["seed"]
                or metadata.environment_config_hash != config.config_hash
                or run["environment_config_hash"] != config.config_hash
                or run["status"] != "complete" or run["actual_environment_steps"] != 2000
                or run["checkpoint_file"] != checkpoint.name
                or run["inventory_capacity"] != DEFAULT_INVENTORY_CAPACITY
                or run["observation_reference_price"] != config.market.quotes[MarketRegime.NORMAL].pack_ask
                or agent != replace(selected, seed=replica["seed"])):
            raise ValueError("incompatible replica configuration or training provenance")
        if set(run["environment_episode_seeds"]) & {s.seed for s in split.scenarios}:
            raise ValueError("held-out seeds overlap recorded model training scenarios")
        load_trained_agent(checkpoint)  # Validate weights before any episode or output creation.
        models.append({"model_id": directory.name, "training_seed": replica["seed"],
                       "checkpoint": str(checkpoint)})
        sources.extend((checkpoint, run_path, agent_path))
    return split, models, {str(p): _digest(p) for p in sources}


def _close(actual, expected, field):
    if not isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"saved trace/episode {field} mismatch")


def _validate_episode(row, steps, config):
    if len(steps) != config.horizon or row.episode_steps != config.horizon:
        raise ValueError("missing steps or incompatible horizon")
    portfolio = PortfolioSnapshot(cash=config.initial_cash, sealed_count=0)
    previous_regime = config.market.initial_regime
    for index, step in enumerate(steps):
        if (step.step_index != index or step.regime_before != previous_regime
                or step.sealed_count_before != portfolio.sealed_count):
            raise ValueError("step order/state continuity mismatch")
        _close(step.cash_before, portfolio.cash, "cash_before")
        market = snapshot_for(step.regime_before, config.market)
        _close(step.portfolio_value_before, liquidation_value(portfolio, market,
               selling_fee=config.selling_fee_rate), "value_before")
        feasible = True
        if step.requested_action is Action.BUY_PACK:
            feasible = portfolio.cash >= market.pack_ask and portfolio.sealed_count < DEFAULT_INVENTORY_CAPACITY
        elif step.requested_action in (Action.SELL_PACK, Action.OPEN_AND_SELL):
            feasible = portfolio.sealed_count > 0
        executed = step.requested_action if feasible else Action.HOLD
        if step.action_was_infeasible != (not feasible) or step.executed_action != executed:
            raise ValueError("requested/executed feasibility mismatch")
        outcome = None
        if executed is Action.OPEN_AND_SELL:
            configured = next((o for o in config.pack.outcomes if o.outcome_id == step.pack_outcome_id.value), None)
            if configured is None:
                raise ValueError("unknown pack outcome")
            _close(step.gross_opened_value, configured.base_bundle_value * market.card_value_multiplier,
                   "gross opening value")
            outcome = PackOutcome(outcome_id=step.pack_outcome_id,
                                  base_gross_value=configured.base_bundle_value,
                                  gross_value=step.gross_opened_value)
        update = apply_action(portfolio, executed, market, selling_fee=config.selling_fee_rate,
                              inventory_capacity=DEFAULT_INVENTORY_CAPACITY, pack_outcome=outcome)
        portfolio, fee = update.portfolio, update.fee_paid
        next_market = snapshot_for(step.regime_after, config.market)
        terminal = index == config.horizon - 1
        if step.termination_reason != (TerminationReason.HORIZON if terminal else None):
            raise ValueError("terminal timing mismatch")
        if terminal:
            update = liquidate(portfolio, next_market, selling_fee=config.selling_fee_rate)
            portfolio, fee = update.portfolio, fee + update.fee_paid
        if step.sealed_count_after != portfolio.sealed_count:
            raise ValueError("inventory/liquidation mismatch")
        _close(step.cash_after, portfolio.cash, "cash_after")
        _close(step.fee_paid, fee, "fee")
        _close(step.portfolio_value_after, liquidation_value(portfolio, next_market,
               selling_fee=config.selling_fee_rate), "value_after")
        _close(step.reward, (step.portfolio_value_after - step.portfolio_value_before)
               / config.initial_cash, "reward")
        previous_regime = step.regime_after
    _close(row.initial_portfolio_value, config.initial_cash, "initial value")
    _close(row.final_portfolio_value, portfolio.cash, "terminal value")
    _close(row.cumulative_reward, fsum(s.reward for s in steps), "cumulative reward")
    _close(row.cumulative_reward, row.final_portfolio_value / config.initial_cash - 1, "telescoping")
    for kind in ("requested", "executed"):
        for action in Action:
            count = sum(getattr(s, kind + "_action") == action for s in steps)
            if count != getattr(row, f"{kind}_{action.name.lower()}_count"):
                raise ValueError("action counts mismatch")
    if row.infeasible_action_count != sum(s.action_was_infeasible for s in steps):
        raise ValueError("infeasible counts mismatch")


def aggregate_results(output_directory):
    """Validate saved raw evidence and regenerate CSVs without policy predictions."""
    output = Path(output_directory)
    manifest = _json(output / "held_out_manifest.json")
    for source, digest in manifest["source_sha256"].items():
        if _digest(source) != digest:
            raise ValueError("input artifact changed since evaluation")
    split = load_split_manifest(manifest["split_manifest"])
    config = split.config
    model_seeds = [m["training_seed"] for m in manifest["models"]]
    if (manifest["manifest_version"] != 1 or split.split != "final_test"
            or manifest["scenario_version"] != SIMULATOR_VERSION
            or manifest["config_hash"] != config.config_hash
            or config.horizon != 100
            or manifest["inventory_capacity"] != DEFAULT_INVENTORY_CAPACITY
            or manifest["reference_price"] != config.market.quotes[MarketRegime.NORMAL].pack_ask
            or len(model_seeds) != 5 or set(model_seeds) != set(FINAL_REPLICA_SEEDS)
            or len({m["model_id"] for m in manifest["models"]}) != 5):
        raise ValueError("incompatible saved manifest settings or replica identities")
    specs = {s.scenario_id: s for s in split.scenarios}
    expected = {(p, None, None) for p in BASELINES} | {
        ("DQN", m["model_id"], m["training_seed"]) for m in manifest["models"]}
    groups, paths, episodes = {}, {}, []
    for group in manifest["groups"]:
        identity = (group["policy_id"], group["model_id"], group["training_seed"])
        if identity not in expected or identity in groups:
            raise ValueError("duplicate or unexpected model/baseline group")
        rows = read_evaluation_rows_jsonl(output / group["rows_file"])
        by_scenario = {}
        for row in rows:
            if row.scenario_id not in specs or row.scenario_id in by_scenario:
                raise ValueError("duplicate or unexpected scenario record")
            if ((row.policy_id, row.model_id, row.training_seed) != identity
                    or row.config_hash != config.config_hash
                    or row.scenario_seed != specs[row.scenario_id].seed
                    or row.git_commit != manifest["git_commit"]
                    or row.simulator_version != manifest["simulator_version"]):
                raise ValueError("incompatible episode provenance")
            trace_file = group["trace_files"][row.scenario_id]
            steps = read_step_traces_jsonl(output / trace_file)
            _validate_episode(row, steps, config)
            path = tuple((s.regime_before, s.regime_after) for s in steps)
            if path != paths.setdefault(row.scenario_id, path):
                raise ValueError("unmatched market paths")
            classification = "learned" if row.policy_id == "DQN" else "rule-based"
            if row.policy_id == "EV_ONE_STEP":
                metadata = _json(output / group["metadata_files"][row.scenario_id])
                if (metadata["information_class"] != "model-informed"
                        or metadata["policy_id"] != row.policy_id
                        or metadata["scenario_id"] != row.scenario_id
                        or metadata["config_hash"] != row.config_hash
                        or metadata["inventory_capacity"] != DEFAULT_INVENTORY_CAPACITY
                        or metadata["observation_reference_price"] != config.market.quotes[MarketRegime.NORMAL].pack_ask):
                    raise ValueError("EV model-informed metadata mismatch")
                classification = "model-informed"
            episodes.append({**row.to_dict(), "information_class": classification,
                             "episode_file": group["rows_file"], "trace_file": trace_file})
            by_scenario[row.scenario_id] = row
        if set(by_scenario) != set(specs):
            raise ValueError("missing scenario records")
        groups[identity] = by_scenario
    if set(groups) != expected:
        raise ValueError("missing model/baseline groups")
    rules = manifest["statistics"]
    order = sorted(specs)
    summaries = []
    for identity in sorted(groups, key=lambda k: (k[0], k[2] or -1)):
        rows = [groups[identity][sid] for sid in order]
        values = [r.final_portfolio_value for r in rows]
        ci = bootstrap_mean_interval(values, seed=rules["bootstrap_seed"],
            resamples=rules["bootstrap_resamples"], confidence=rules["confidence"])
        summary = {"policy_id": identity[0], "model_id": identity[1], "training_seed": identity[2],
                   "information_class": "model-informed" if identity[0] == "EV_ONE_STEP" else
                   ("learned" if identity[0] == "DQN" else "rule-based"),
                   **summarize_rows(rows, loss_tolerance=rules["loss_tolerance"]),
                   "mean_final_value_ci_low": ci[0], "mean_final_value_ci_high": ci[1],
                   "ci_sampling_unit": "scenario; fixed policy/model",
                   "small_sample_warning": len(rows) < 30}
        for baseline in BASELINES:
            low, high, difference = None, None, None
            if identity[0] == "DQN":
                baseline_values = [groups[(baseline, None, None)][sid].final_portfolio_value for sid in order]
                difference = fsum(v - b for v, b in zip(values, baseline_values)) / len(values)
                low, high = bootstrap_mean_interval(values, baseline_values,
                    seed=rules["bootstrap_seed"], resamples=rules["bootstrap_resamples"],
                    confidence=rules["confidence"])
            summary.update({baseline + "_paired_mean_difference": difference,
                            baseline + "_paired_ci_low": low, baseline + "_paired_ci_high": high})
        summaries.append(summary)
    episodes.sort(key=lambda r: (r["policy_id"], r["training_seed"] or -1, r["scenario_id"]))
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "summary.csv", summaries)
    model_means = [s["mean_final_value"] for s in summaries if s["policy_id"] == "DQN"]
    _write_json(output / "uncertainty.json", {**rules,
        "method": "percentile bootstrap of scenario means; linear quantiles",
        "pairing": "same scenario index for model and baseline; common resamples across models",
        "sampling_unit": "scenario, conditional on each fixed checkpoint",
        "training_uncertainty_included": False,
        "replica_mean_final_value": float(np.mean(model_means)),
        "replica_sample_std_of_means": float(np.std(model_means, ddof=1)),
        "replica_summary_label": "descriptive across five models on shared scenarios; not independent market samples",
        "scenario_count": len(specs), "small_sample_warning": len(specs) < 30,
        "limitations": manifest["limitations"]})
    return output / "summary.csv"


def run_held_out(*, split_manifest=ROOT / "configs/splits/final_test_pf22.json",
                 replica_manifest=ROOT / "runs/pf21-final-replicas/final_replicas_manifest.json",
                 frozen_config=ROOT / "configs/env_frozen.yaml",
                 selected_agent_config=ROOT / "configs/agent_selected.yaml",
                 output_directory=ROOT / "results", bootstrap_seed=22,
                 bootstrap_resamples=2000, confidence=0.95, loss_tolerance=1e-9):
    """Explicit execution entry point. Never train or select models."""
    split_path, replica_path, frozen_path, selected_path = (
        Path(p).resolve() for p in (split_manifest, replica_manifest, frozen_config, selected_agent_config))
    bootstrap_mean_interval([0, 0], seed=bootstrap_seed, resamples=bootstrap_resamples, confidence=confidence)
    if isinstance(loss_tolerance, bool) or not isfinite(loss_tolerance) or loss_tolerance < 0:
        raise ValueError("loss_tolerance must be finite and nonnegative")
    output = Path(output_directory).resolve()
    if output.exists():
        raise ValueError("output directory already exists; preserve prior evidence")
    split, models, sources = _preflight(split_path, replica_path, frozen_path, selected_path)
    config = split.config
    commit, dirty = git_metadata()
    version = importlib.metadata.version("packfolio")
    output.mkdir(parents=True)
    groups = []
    for model in models:
        directory = output / "raw/DQN" / str(model["training_seed"])
        evaluate_checkpoint_on_split(checkpoint_path=model["checkpoint"],
            split_manifest_path=split_path, output_directory=directory)
        groups.append({"policy_id": "DQN", "model_id": model["model_id"],
            "training_seed": model["training_seed"],
            "rows_file": str((directory / "evaluation_rows.jsonl").relative_to(output)),
            "trace_files": {s.scenario_id: str((directory / "step_traces" /
                (s.scenario_id.replace(":", "_") + ".jsonl")).relative_to(output)) for s in split.scenarios}})
    reference = config.market.quotes[MarketRegime.NORMAL].pack_ask
    policies = (CashOnlyPolicy(), BuyAndHoldPolicy(initial_cash=config.initial_cash, reference_price=reference),
                AlwaysOpenPolicy(initial_cash=config.initial_cash, reference_price=reference), EVOneStepPolicy(config))
    for policy in policies:
        directory = output / "raw" / policy.policy_id
        directory.mkdir(parents=True)
        rows, trace_files, metadata_files = [], {}, {}
        for spec in split.scenarios:
            metadata = EvaluationMetadata(simulator_version=version, policy_id=policy.policy_id,
                model_id=None, training_seed=None, scenario_id=spec.scenario_id,
                scenario_seed=spec.seed, config_hash=config.config_hash, git_commit=commit)
            env = build_environment(config)
            trace = directory / "step_traces" / (str(spec.seed) + ".jsonl")
            if isinstance(policy, EVOneStepPolicy):
                episode_directory = directory / str(spec.seed)
                result = evaluate_ev_episode(policy, env, metadata, output_directory=episode_directory)
                trace = episode_directory / "step_traces.jsonl"
                metadata_files[spec.scenario_id] = str((episode_directory / "policy_metadata.json").relative_to(output))
            else:
                result = evaluate_episode(policy, env, metadata)
                trace.parent.mkdir(exist_ok=True)
                write_step_traces_jsonl(trace, result.step_traces)
            rows.append(result.row)
            trace_files[spec.scenario_id] = str(trace.relative_to(output))
        write_evaluation_rows_jsonl(directory / "evaluation_rows.jsonl", rows)
        groups.append({"policy_id": policy.policy_id, "model_id": None, "training_seed": None,
            "rows_file": str((directory / "evaluation_rows.jsonl").relative_to(output)),
            "trace_files": trace_files, "metadata_files": metadata_files})
    _write_json(output / "held_out_manifest.json", {
        "manifest_version": 1, "split_manifest": str(split_path), "source_sha256": sources,
        "scenario_version": SIMULATOR_VERSION, "simulator_version": version,
        "config_hash": config.config_hash, "inventory_capacity": DEFAULT_INVENTORY_CAPACITY,
        "reference_price": reference, "git_commit": commit, "git_working_tree_dirty": dirty,
        "models": models, "groups": groups,
        "statistics": {"bootstrap_seed": bootstrap_seed, "bootstrap_resamples": bootstrap_resamples,
                       "confidence": confidence, "loss_tolerance": loss_tolerance,
                       "quantile_method": "linear", "standard_deviation_ddof": 1},
        "limitations": ["Historical PF-21 tuning records are incomplete locally.",
                        "Models trained for 2000 steps / 20 episodes; convergence not established.",
                        "PF-12/PF-13 float32 affordability ambiguity is retained.",
                        "Scenario bootstrap does not measure training variability or model-selection uncertainty."]})
    return aggregate_results(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "configs/splits/final_test_pf22.json")
    parser.add_argument("--replica-manifest", type=Path,
                        default=ROOT / "runs/pf21-final-replicas/final_replicas_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--bootstrap-seed", type=int, default=22)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    result = aggregate_results(args.output) if args.aggregate_only else run_held_out(
        split_manifest=args.split_manifest, replica_manifest=args.replica_manifest,
        output_directory=args.output, bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
