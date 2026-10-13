"""PF-23 adapter: variant accounting with the frozen base scenario recipe.

Only orchestration and observation construction mirror PackfolioEnv. Randomness
and transactions remain owned by the public Scenario and portfolio interfaces.
"""

from dataclasses import replace
import argparse
import csv
import gzip
import importlib.metadata
import json
from math import fsum
from numbers import Integral
from pathlib import Path
from time import perf_counter

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from packfolio.config import EnvironmentConfig, REGIME_ORDER, load_environment_config
from packfolio.env import DEFAULT_INVENTORY_CAPACITY
from packfolio.agents.dqn_agent import load_trained_agent
from packfolio.baselines import AlwaysOpenPolicy, BuyAndHoldPolicy, CashOnlyPolicy
from packfolio.ev_baseline import EVOneStepPolicy
from packfolio.evaluate import EvaluationMetadata, evaluate_episode, read_evaluation_rows_jsonl, read_step_traces_jsonl
from packfolio.held_out import (
    BASELINES, ROOT, _digest, _json, _preflight, _validate_episode, _write_csv,
    _write_json, bootstrap_mean_interval, summarize_rows,
)
from packfolio.provenance import git_metadata
from packfolio.portfolio import apply_action, liquidate, liquidation_value
from packfolio.scenarios import Scenario, ScenarioSpec
from packfolio.types import (
    Action, MarketRegime, OBSERVATION_SIZE, PackOutcome, PackOutcomeId,
    EvaluationRow, PortfolioSnapshot, StepInfo, TerminationReason,
)


class FeeSensitivityEnv(gym.Env):
    """Keep base latent draws, but use the evaluation fee for every transaction.

    Policies receive only observations. The evaluator must save ``provenance``
    separately: an evaluation scenario ID is not the RNG source scenario ID.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvironmentConfig) -> None:
        if not isinstance(config, EnvironmentConfig):
            raise TypeError("config must be EnvironmentConfig")
        frozen = Path(__file__).resolve().parents[2] / "configs/env_frozen.yaml"
        base = load_environment_config(frozen)
        if base.horizon != 100 or base.selling_fee_rate != 0.05:
            raise ValueError("require the frozen 100-step, 5% base configuration")
        if replace(config, selling_fee_rate=base.selling_fee_rate) != base:
            raise ValueError("evaluation configuration may differ only in selling_fee_rate")
        for outcome in config.pack.outcomes:
            PackOutcomeId(outcome.outcome_id)
        self._config = config
        self._base = base
        self._capacity = DEFAULT_INVENTORY_CAPACITY
        self._reference_price = base.market.quotes[MarketRegime.NORMAL].pack_ask
        self._scenario = None
        self._spec = None
        self._portfolio = None
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            high=np.array([np.inf, 1, np.inf, np.inf, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dict or None")
        if options:
            raise ValueError("reset options are not supported")
        spec = ScenarioSpec(self._base.config_hash, seed) if seed is not None else None
        super().reset(seed=seed)
        if spec is None:
            episode_seed = int(self.np_random.integers(2**64, dtype=np.uint64))
            spec = ScenarioSpec(self._base.config_hash, episode_seed)
        self._spec = spec
        self._scenario = Scenario(self._base, spec)
        self._portfolio = PortfolioSnapshot(cash=self._config.initial_cash, sealed_count=0)
        return self._get_observation(), {}

    def provenance(self):
        """Return genuine evaluation and randomness-source identities after reset."""
        if self._spec is None:
            raise RuntimeError("call reset() before requesting episode provenance")
        evaluation_spec = ScenarioSpec(self._config.config_hash, self._spec.seed)
        return {
            "adapter_version": 1,
            "scenario_version": self._spec.simulator_version,
            "scenario_seed": self._spec.seed,
            "randomness_source_config_hash": self._base.config_hash,
            "randomness_source_scenario_id": self._spec.scenario_id,
            "evaluation_config_hash": self._config.config_hash,
            "evaluation_scenario_id": evaluation_spec.scenario_id,
            "selling_fee_rate": self._config.selling_fee_rate,
            "inventory_capacity": self._capacity,
            "observation_reference_price": self._reference_price,
        }

    def step(self, action):
        if self._scenario is None or self._portfolio is None:
            raise RuntimeError("call reset() before step()")
        if self._scenario.done:
            raise RuntimeError("episode has ended; call reset() before step()")
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        if not 0 <= action < len(Action):
            raise ValueError("action must be in 0..3")
        requested = Action(int(action))
        index = self._scenario.timestep
        market = self._scenario.current_market
        before = self._portfolio
        fee = self._config.selling_fee_rate
        value_before = liquidation_value(before, market, selling_fee=fee)
        feasible = True
        if requested is Action.BUY_PACK:
            feasible = before.cash >= market.pack_ask and before.sealed_count < self._capacity
        elif requested in (Action.OPEN_AND_SELL, Action.SELL_PACK):
            feasible = before.sealed_count > 0
        executed = requested if feasible else Action.HOLD
        outcome = None
        if executed is Action.OPEN_AND_SELL:
            draw = self._scenario.open_pack()
            configured = next(o for o in self._config.pack.outcomes if o.outcome_id == draw.outcome_id)
            outcome = PackOutcome(outcome_id=PackOutcomeId(draw.outcome_id),
                                  base_gross_value=configured.base_bundle_value,
                                  gross_value=draw.gross_bundle_value)
        update = apply_action(before, executed, market, selling_fee=fee,
                              inventory_capacity=self._capacity, pack_outcome=outcome)
        next_market = self._scenario.advance()
        self._portfolio = update.portfolio
        terminated = self._scenario.done
        fee_paid = update.fee_paid
        if terminated:
            update = liquidate(self._portfolio, next_market, selling_fee=fee)
            self._portfolio = update.portfolio
            fee_paid += update.fee_paid
        value_after = liquidation_value(self._portfolio, next_market, selling_fee=fee)
        reward = (value_after - value_before) / self._config.initial_cash
        info = StepInfo(
            step_index=index, requested_action=requested, executed_action=executed,
            action_was_infeasible=not feasible, regime_before=market.regime,
            regime_after=next_market.regime, cash_before=before.cash,
            cash_after=self._portfolio.cash, sealed_count_before=before.sealed_count,
            sealed_count_after=self._portfolio.sealed_count,
            pack_outcome_id=outcome.outcome_id if outcome is not None else None,
            gross_opened_value=outcome.gross_value if outcome is not None else None,
            fee_paid=fee_paid, portfolio_value_before=value_before,
            portfolio_value_after=value_after, reward=reward,
            termination_reason=TerminationReason.HORIZON if terminated else None,
        )
        return self._get_observation(), reward, terminated, False, info.to_dict()

    def _get_observation(self):
        market = self._scenario.current_market
        remaining = self._config.horizon - self._scenario.timestep
        with np.errstate(over="ignore"):
            observation = np.array([
                self._portfolio.cash / self._config.initial_cash,
                self._portfolio.sealed_count / self._capacity,
                market.pack_ask / self._reference_price, market.card_value_multiplier,
                remaining / self._config.horizon,
                *(float(market.regime is regime) for regime in REGIME_ORDER),
            ], dtype=np.float32)
        if not np.isfinite(observation).all():
            raise ValueError("observation values must be finite as float32")
        return observation


def _identity(row):
    return row.policy_id, row.model_id, row.training_seed


def _check_evidence(row, steps, config, paths, openings):
    _validate_episode(row, steps, config)
    path = tuple((s.regime_before, s.regime_after) for s in steps)
    if path != paths.setdefault(row.scenario_seed, path):
        raise ValueError("unmatched exogenous market path")
    for step in steps:
        if step.pack_outcome_id is not None:
            key = row.scenario_seed, step.step_index
            draw = step.pack_outcome_id, step.gross_opened_value
            if draw != openings.setdefault(key, draw):
                raise ValueError("unmatched shared-timestep pack draw")


def validate_reference(reference_directory, split_manifest, replica_manifest):
    """Read and validate PF-22 evidence without rewriting it or running policies."""
    reference = Path(reference_directory).resolve()
    split_path, replica_path = Path(split_manifest).resolve(), Path(replica_manifest).resolve()
    split, models, sources = _preflight(split_path, replica_path,
        ROOT / "configs/env_frozen.yaml", ROOT / "configs/agent_selected.yaml")
    manifest = _json(reference / "held_out_manifest.json")
    if (manifest["config_hash"] != split.config.config_hash
            or manifest["scenario_version"] != split.scenarios[0].simulator_version
            or manifest["manifest_version"] != 1
            or manifest["inventory_capacity"] != DEFAULT_INVENTORY_CAPACITY
            or manifest["reference_price"] != 10.0
            or manifest["models"] != models
            or _digest(manifest["split_manifest"]) != _digest(split_path)):
        raise ValueError("incompatible PF-22 reference provenance")
    for source, digest in manifest["source_sha256"].items():
        if _digest(source) != digest:
            raise ValueError("PF-22 source artifact changed")
    expected = {(p, None, None) for p in BASELINES} | {
        ("DQN", m["model_id"], m["training_seed"]) for m in models}
    seeds = {s.seed for s in split.scenarios}
    groups, paths, openings = {}, {}, {}
    evidence = {str(reference / "held_out_manifest.json"): _digest(reference / "held_out_manifest.json")}
    for group in manifest["groups"]:
        identity = group["policy_id"], group["model_id"], group["training_seed"]
        if identity not in expected or identity in groups:
            raise ValueError("duplicate or unexpected reference strategy")
        row_path = reference / group["rows_file"]
        evidence[str(row_path)] = _digest(row_path)
        rows = {}
        for row in read_evaluation_rows_jsonl(row_path):
            if (row.scenario_seed not in seeds or row.scenario_seed in rows
                    or _identity(row) != identity or row.config_hash != split.config.config_hash
                    or row.scenario_id != ScenarioSpec(row.config_hash, row.scenario_seed).scenario_id
                    or row.git_commit != manifest["git_commit"]
                    or row.simulator_version != manifest["simulator_version"]):
                raise ValueError("duplicate or incompatible reference episode")
            trace = reference / group["trace_files"][row.scenario_id]
            evidence[str(trace)] = _digest(trace)
            steps = read_step_traces_jsonl(trace)
            _check_evidence(row, steps, split.config, paths, openings)
            if row.policy_id == "EV_ONE_STEP":
                sidecar = reference / group["metadata_files"][row.scenario_id]
                label = _json(sidecar)
                evidence[str(sidecar)] = _digest(sidecar)
                if (label["information_class"] != "model-informed"
                        or label["policy_id"] != row.policy_id
                        or label["config_hash"] != row.config_hash
                        or label["scenario_id"] != row.scenario_id
                        or label["inventory_capacity"] != DEFAULT_INVENTORY_CAPACITY
                        or label["observation_reference_price"] != manifest["reference_price"]):
                    raise ValueError("invalid reference EV classification")
            rows[row.scenario_seed] = row
        if set(rows) != seeds:
            raise ValueError("missing reference scenarios")
        groups[identity] = rows
    if set(groups) != expected:
        raise ValueError("missing reference strategies")
    # The published CSV must represent these same raw rows, not another run.
    seen = set()
    with (reference / "episodes.csv").open(encoding="utf-8", newline="") as stream:
        for entry in csv.DictReader(stream):
            identity = entry["policy_id"], entry["model_id"] or None, (
                int(entry["training_seed"]) if entry["training_seed"] else None)
            seed = int(entry["scenario_seed"])
            key = identity, seed
            if identity not in groups or seed not in groups[identity] or key in seen:
                raise ValueError("duplicate or incompatible reference CSV")
            seen.add(key)
            if any(entry[k] != ("" if v is None else str(v))
                   for k, v in groups[identity][seed].to_dict().items()):
                raise ValueError("reference CSV differs from raw episodes")
    if len(seen) != len(expected) * len(seeds):
        raise ValueError("missing reference CSV episodes")
    for name in ("episodes.csv", "summary.csv", "uncertainty.json"):
        evidence[str(reference / name)] = _digest(reference / name)
    return split, models, groups, paths, openings, {**sources, **evidence}


def aggregate_fee_sensitivity(output_directory):
    """Build 27 summaries from saved evidence; never call a policy here."""
    output = Path(output_directory)
    manifest = _json(output / "manifest.json")
    for source, digest in manifest["source_sha256"].items():
        if _digest(source) != digest:
            raise ValueError("fee-sensitivity input artifact changed")
    if _digest(output / "episodes.jsonl.gz") != manifest["episodes_sha256"]:
        raise ValueError("fee-sensitivity episode artifact changed")
    split, models, reference, paths, openings, _ = validate_reference(
        manifest["reference_directory"], manifest["split_manifest"], manifest["replica_manifest"])
    configs = {float(f): load_environment_config(p) for f, p in manifest["fee_configs"].items()}
    if set(configs) != {0.025, 0.05, 0.1}:
        raise ValueError("require the three predeclared fees")
    for fee, config in configs.items():
        FeeSensitivityEnv(config)
        if config.selling_fee_rate != fee:
            raise ValueError("fee/config mismatch")
    expected_provenance = {}
    for fee, config in configs.items():
        env = FeeSensitivityEnv(config)
        env.reset(seed=split.scenarios[0].seed)
        data = env.provenance()
        for key in ("scenario_seed", "randomness_source_scenario_id", "evaluation_scenario_id"):
            data.pop(key)
        expected_provenance[str(fee)] = data
    if manifest["variant_provenance"] != expected_provenance:
        raise ValueError("invalid randomness-source/evaluation provenance")
    groups = {(0.05, identity): rows for identity, rows in reference.items()}
    for fee in (0.025, 0.1):
        for identity in reference:
            groups[fee, identity] = {}
    with gzip.open(output / "episodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            row = EvaluationRow.from_dict(record["row"])
            fee = record["selling_fee_rate"]
            key = fee, _identity(row)
            if fee == 0.05 or key not in groups or row.scenario_seed in groups[key]:
                raise ValueError("duplicate or unexpected variant episode")
            config = configs[fee]
            if (row.scenario_seed not in paths or row.config_hash != config.config_hash
                    or row.scenario_id != ScenarioSpec(config.config_hash, row.scenario_seed).scenario_id
                    or row.git_commit != manifest["git_commit"]
                    or row.simulator_version != manifest["simulator_version"]):
                raise ValueError("incompatible variant episode provenance")
            steps = tuple(StepInfo.from_dict(s) for s in record["steps"])
            _check_evidence(row, steps, config, paths, openings)
            groups[key][row.scenario_seed] = row
    rules = manifest["statistics"]
    kwargs = {"seed": rules["bootstrap_seed"], "resamples": rules["bootstrap_resamples"],
              "confidence": rules["confidence"]}
    summaries = []
    for (fee, identity), by_seed in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1][0], item[0][1][2] or -1)):
        if set(by_seed) != set(paths):
            raise ValueError("missing variant scenarios")
        rows = [by_seed[s] for s in sorted(paths)]
        values = [r.final_portfolio_value for r in rows]
        base_values = [reference[identity][s].final_portfolio_value for s in sorted(paths)]
        interval = bootstrap_mean_interval(values, **kwargs)
        paired = bootstrap_mean_interval(values, base_values, **kwargs)
        summaries.append({"selling_fee_rate": fee, "policy_id": identity[0],
            "model_id": identity[1], "training_seed": identity[2],
            "policy_status": "frozen weights; no exploration" if identity[0] == "DQN" else (
                "fee-recomputed" if identity[0] == "EV_ONE_STEP" else "unchanged rule"),
            "information_class": "model-informed" if identity[0] == "EV_ONE_STEP" else (
                "learned" if identity[0] == "DQN" else "rule-based"),
            "evaluation_config_hash": configs[fee].config_hash,
            "randomness_source_config_hash": split.config.config_hash,
            **summarize_rows(rows, loss_tolerance=rules["loss_tolerance"]),
            "infeasible_action_count": sum(r.infeasible_action_count for r in rows),
            "mean_final_value_ci_low": interval[0], "mean_final_value_ci_high": interval[1],
            "paired_mean_change_vs_5pct": fsum(v-b for v,b in zip(values, base_values)) / len(values),
            "paired_change_ci_low": paired[0], "paired_change_ci_high": paired[1],
            "ci_sampling_unit": "scenario; fixed model; paired fee minus 5%",
            "small_sample_warning": len(rows) < 30})
    destination = Path(manifest["summary_file"])
    _write_csv(destination, summaries)
    return destination


def run_fee_sensitivity(*, reference_directory=ROOT / "results",
                       split_manifest=ROOT / "configs/splits/final_test_pf22.json",
                       replica_manifest=ROOT / "runs/pf21-final-replicas/final_replicas_manifest.json",
                       output_directory=ROOT / "outputs/pf23", summary_file=ROOT / "results/fee_sensitivity.csv",
                       bootstrap_seed=23, bootstrap_resamples=2000):
    """Explicit execution only; preserve PF-22 and refuse existing output paths."""
    output, summary = Path(output_directory).resolve(), Path(summary_file).resolve()
    if output.exists() or summary.exists():
        raise ValueError("output already exists; preserve prior evidence")
    bootstrap_mean_interval([0, 0], seed=bootstrap_seed, resamples=bootstrap_resamples)
    split, models, reference, paths, openings, sources = validate_reference(
        reference_directory, split_manifest, replica_manifest)
    fee_paths = {0.025: ROOT / "configs/env_frozen_fee_low.yaml",
                 0.05: ROOT / "configs/env_frozen.yaml", 0.1: ROOT / "configs/env_frozen_fee_high.yaml"}
    configs = {fee: load_environment_config(path) for fee, path in fee_paths.items()}
    for fee, config in configs.items():
        FeeSensitivityEnv(config)
        if config.selling_fee_rate != fee:
            raise ValueError("fee/config mismatch")
        sources[str(fee_paths[fee])] = _digest(fee_paths[fee])
    commit, dirty = git_metadata()
    version = importlib.metadata.version("packfolio")
    output.mkdir(parents=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    provenance = {}
    with gzip.open(output / "episodes.jsonl.gz", "wt", encoding="utf-8") as stream:
        for fee, config in configs.items():
            env = FeeSensitivityEnv(config)
            env.reset(seed=split.scenarios[0].seed)
            data = env.provenance()
            for key in ("scenario_seed", "randomness_source_scenario_id", "evaluation_scenario_id"):
                data.pop(key)
            provenance[str(fee)] = data
            if fee == 0.05:
                continue
            policies = [(load_trained_agent(m["checkpoint"]), "DQN", m["model_id"], m["training_seed"])
                        for m in models]
            policies += [(p, p.policy_id, None, None) for p in (
                CashOnlyPolicy(), BuyAndHoldPolicy(initial_cash=config.initial_cash,
                    reference_price=config.market.quotes[MarketRegime.NORMAL].pack_ask),
                AlwaysOpenPolicy(initial_cash=config.initial_cash,
                    reference_price=config.market.quotes[MarketRegime.NORMAL].pack_ask), EVOneStepPolicy(config))]
            for policy, policy_id, model_id, training_seed in policies:
                for spec in split.scenarios:
                    metadata = EvaluationMetadata(simulator_version=version, policy_id=policy_id,
                        model_id=model_id, training_seed=training_seed,
                        scenario_id=ScenarioSpec(config.config_hash, spec.seed).scenario_id,
                        scenario_seed=spec.seed, config_hash=config.config_hash, git_commit=commit)
                    result = evaluate_episode(policy, env, metadata)
                    _check_evidence(result.row, result.step_traces, config, paths, openings)
                    stream.write(json.dumps({"selling_fee_rate": fee, "row": result.row.to_dict(),
                        "steps": [s.to_dict() for s in result.step_traces]}, allow_nan=False) + "\n")
    _write_json(output / "manifest.json", {
        "reference_directory": str(Path(reference_directory).resolve()),
        "split_manifest": str(Path(split_manifest).resolve()),
        "replica_manifest": str(Path(replica_manifest).resolve()),
        "fee_configs": {str(f): str(p) for f,p in fee_paths.items()},
        "variant_provenance": provenance, "source_sha256": sources,
        "episodes_sha256": _digest(output / "episodes.jsonl.gz"),
        "summary_file": str(summary), "git_commit": commit, "git_working_tree_dirty": dirty,
        "simulator_version": version, "evaluation_and_recording_seconds": perf_counter()-started,
        "statistics": {"bootstrap_seed": bootstrap_seed, "bootstrap_resamples": bootstrap_resamples,
                       "confidence": 0.95, "loss_tolerance": 1e-9},
        "limitations": ["Fixed-model scenario bootstrap excludes training and selection uncertainty.",
                        "Intervals are not adjusted for multiple comparisons.",
                        "Historical tuning records incomplete; 2000-step models have no convergence guarantee.",
                        "PF-12/PF-13 float32 affordability convention is unchanged."]})
    return aggregate_fee_sensitivity(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description="PF-23 frozen-policy fee sensitivity")
    parser.add_argument("--reference", type=Path, default=ROOT / "results")
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "configs/splits/final_test_pf22.json")
    parser.add_argument("--replica-manifest", type=Path,
                        default=ROOT / "runs/pf21-final-replicas/final_replicas_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pf23")
    parser.add_argument("--summary", type=Path, default=ROOT / "results/fee_sensitivity.csv")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(argv)
    result = aggregate_fee_sensitivity(args.output) if args.aggregate_only else run_fee_sensitivity(
        reference_directory=args.reference, split_manifest=args.split_manifest,
        replica_manifest=args.replica_manifest, output_directory=args.output, summary_file=args.summary)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
