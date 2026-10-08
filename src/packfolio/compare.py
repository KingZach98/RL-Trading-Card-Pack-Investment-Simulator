"""PF-19 development comparison, built from validated saved episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from math import fsum, isclose
from pathlib import Path
from time import perf_counter

from packfolio.agents.dqn_agent import load_checkpoint_metadata
from packfolio.baselines import AlwaysOpenPolicy, BuyAndHoldPolicy, CashOnlyPolicy
from packfolio.env import DEFAULT_INVENTORY_CAPACITY, build_environment
from packfolio.ev_baseline import EVOneStepPolicy, evaluate_ev_episode
from packfolio.evaluate import (
    EvaluationMetadata, evaluate_checkpoint_on_split, evaluate_episode,
    read_evaluation_rows_jsonl, read_step_traces_jsonl,
    write_evaluation_rows_jsonl, write_step_traces_jsonl,
)
from packfolio.provenance import git_metadata
from packfolio.scenarios import SIMULATOR_VERSION, load_split_manifest
from packfolio.types import Action, MarketRegime, StepInfo, TerminationReason


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs"
POLICY_IDS = ("CASH_ONLY", "BUY_AND_HOLD", "ALWAYS_OPEN", "EV_ONE_STEP", "DQN")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _development_split(path: Path):
    split = load_split_manifest(path)
    if split.split != "validation":
        raise ValueError("PF-19 requires a development validation split, never final_test")
    return split


def _close(actual: float, expected: float, label: str) -> None:
    if not isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"saved episode {label} disagrees with its trace")


def build_comparison(output_directory: str | Path) -> Path:
    """Read raw files, validate every trace, then write the per-episode table."""
    output = Path(output_directory)
    manifest = json.loads((output / "comparison_manifest.json").read_text(encoding="utf-8"))
    split = _development_split(Path(manifest["split_manifest"]))
    config = split.config
    if (manifest["config_hash"] != config.config_hash
            or manifest["scenario_version"] != SIMULATOR_VERSION
            or manifest["inventory_capacity"] != DEFAULT_INVENTORY_CAPACITY
            or manifest["reference_price"] != config.market.quotes[MarketRegime.NORMAL].pack_ask):
        raise ValueError("comparison settings disagree with the shared environment")
    checkpoint = Path(manifest["checkpoint"])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint content changed")
    checkpoint_metadata = load_checkpoint_metadata(checkpoint)
    if checkpoint_metadata.environment_config_hash != config.config_hash:
        raise ValueError("checkpoint configuration mismatch")

    expected = {(policy, spec.scenario_id) for policy in POLICY_IDS for spec in split.scenarios}
    seen = set()
    market_paths = {}
    records = []
    for entry in manifest["episodes"]:
        rows = read_evaluation_rows_jsonl(output / entry["rows_file"])
        matches = [row for row in rows if row.scenario_id == entry["scenario_id"]]
        if len(matches) != 1:
            raise ValueError("episode row must have one unambiguous saved source")
        row = matches[0]
        key = (row.policy_id, row.scenario_id)
        if key in seen or key not in expected or row.policy_id != entry["policy_id"]:
            raise ValueError("duplicate or unexpected policy/scenario episode")
        seen.add(key)
        spec = next(spec for spec in split.scenarios if spec.scenario_id == row.scenario_id)
        if (row.scenario_seed != spec.seed or row.config_hash != config.config_hash
                or row.simulator_version != manifest["simulator_version"]
                or row.git_commit != manifest["git_commit"]):
            raise ValueError("episode identity/provenance mismatch")
        if row.policy_id == "DQN":
            if row.training_seed != checkpoint_metadata.training_seed or row.model_id != checkpoint.parent.name:
                raise ValueError("DQN model identity mismatch")
        elif row.model_id is not None or row.training_seed is not None:
            raise ValueError("baseline must not have a trained model identity")

        information_class = "learned" if row.policy_id == "DQN" else "rule-based"
        if row.policy_id == "EV_ONE_STEP":
            metadata = json.loads((output / entry["policy_metadata_file"]).read_text(encoding="utf-8"))
            if (metadata["information_class"] != "model-informed"
                    or metadata["policy_id"] != row.policy_id
                    or metadata["scenario_id"] != row.scenario_id
                    or metadata["config_hash"] != config.config_hash
                    or metadata["inventory_capacity"] != manifest["inventory_capacity"]
                    or metadata["observation_reference_price"] != manifest["reference_price"]):
                raise ValueError("EV model-informed metadata mismatch")
            information_class = metadata["information_class"]

        steps = read_step_traces_jsonl(output / entry["trace_file"])
        if len(steps) != config.horizon or row.episode_steps != config.horizon:
            raise ValueError("episode violates the shared horizon/action limit")
        # Replay requested actions, not policy decisions: this checks all ledger,
        # fee, draw-slot and terminal fields using the authoritative simulator.
        env = build_environment(config)
        env.reset(seed=spec.seed)
        for index, step in enumerate(steps):
            _, _, terminated, truncated, info = env.step(step.requested_action)
            if step != StepInfo.from_dict(info) or step.step_index != index:
                raise ValueError(f"saved trace differs from environment replay at step {index}")
            if truncated or terminated != (index == config.horizon - 1):
                raise ValueError("incorrect terminal timing")
            # Independently reconcile recorded cash flows, including the extra
            # terminal sale, so matching a replay alone is not the accounting proof.
            ask = config.market.quotes[step.regime_before].pack_ask
            next_ask = config.market.quotes[step.regime_after].pack_ask
            inventory = step.sealed_count_before
            purchase = ask if step.executed_action is Action.BUY_PACK else 0.0
            gross_sales = 0.0
            if step.executed_action is Action.BUY_PACK:
                inventory += 1
            elif step.executed_action in (Action.SELL_PACK, Action.OPEN_AND_SELL):
                inventory -= 1
                gross_sales = ask if step.executed_action is Action.SELL_PACK else step.gross_opened_value
            if terminated:
                gross_sales += inventory * next_ask
                inventory = 0
            if step.sealed_count_after != inventory:
                raise ValueError("inventory accounting mismatch")
            _close(step.fee_paid, gross_sales * config.selling_fee_rate, "single sale fee")
            _close(step.cash_after, step.cash_before - purchase + gross_sales - step.fee_paid, "cash flow")
            net_fraction = 1 - config.selling_fee_rate
            _close(step.portfolio_value_before,
                   step.cash_before + step.sealed_count_before * ask * net_fraction, "pre-action value")
            _close(step.portfolio_value_after,
                   step.cash_after + inventory * next_ask * net_fraction, "post-action value")
            _close(step.reward, (step.portfolio_value_after - step.portfolio_value_before)
                   / config.initial_cash, "step reward")
        last = steps[-1]
        if last.sealed_count_after != 0 or last.termination_reason is not TerminationReason.HORIZON:
            raise ValueError("terminal liquidation is incomplete")
        if not row.terminated or row.truncated or row.termination_reason is not TerminationReason.HORIZON:
            raise ValueError("episode termination mismatch")
        _close(row.initial_portfolio_value, config.initial_cash, "initial value")
        _close(row.final_portfolio_value, last.cash_after, "final liquidated value")
        _close(row.final_portfolio_value, last.portfolio_value_after, "final value")
        _close(row.cumulative_reward, fsum(step.reward for step in steps), "reward sum")
        _close(row.cumulative_reward, row.final_portfolio_value / config.initial_cash - 1, "telescoping")
        for action in ("hold", "buy_pack", "open_and_sell", "sell_pack"):
            for kind in ("requested", "executed"):
                count = sum(getattr(step, f"{kind}_action").name.lower() == action for step in steps)
                if getattr(row, f"{kind}_{action}_count") != count:
                    raise ValueError("action count mismatch")
        if row.infeasible_action_count != sum(step.action_was_infeasible for step in steps):
            raise ValueError("infeasible action count mismatch")
        path = tuple((step.regime_before, step.regime_after) for step in steps)
        if path != market_paths.setdefault(spec.scenario_id, path):
            raise ValueError("strategies have unmatched market paths")
        records.append({**row.to_dict(), "information_class": information_class,
                        "episode_file": entry["rows_file"], "trace_file": entry["trace_file"]})
    if seen != expected:
        raise ValueError("missing policy/scenario episodes")
    table = output / "comparison.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return table


def run_comparison(*, split_manifest: str | Path, output_directory: str | Path,
                   checkpoint: str | Path | None = None) -> Path:
    """Run five strategies; optionally train the unchanged development pilot."""
    started = perf_counter()
    split_path = Path(split_manifest).resolve()
    split = _development_split(split_path)
    output = Path(output_directory).resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        raise ValueError("generated artifacts must be in a subdirectory of outputs/")
    if output.exists():
        raise ValueError("use a new output directory; do not overwrite comparison evidence")
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        if load_checkpoint_metadata(checkpoint).environment_config_hash != split.config.config_hash:
            raise ValueError("checkpoint configuration mismatch")
    output.mkdir(parents=True)
    timings = {}
    if checkpoint is None:
        from packfolio.train import train

        raw_split = json.loads(split_path.read_text(encoding="utf-8"))
        stage = perf_counter()
        run = train(split_path.parent / raw_split["environment_config"],
                    ROOT / "configs" / "agent.yaml", output / "pilot")
        timings["pilot_training_seconds"] = perf_counter() - stage
        checkpoint = run / "checkpoint.pt"
    checkpoint = Path(checkpoint).resolve()
    stage = perf_counter()
    dqn_output = evaluate_checkpoint_on_split(checkpoint_path=checkpoint,
        split_manifest_path=split_path, output_directory=output / "DQN")
    timings["DQN_evaluation_seconds"] = perf_counter() - stage
    config = split.config
    commit, dirty = git_metadata()
    version = importlib.metadata.version("packfolio")
    reference_price = config.market.quotes[MarketRegime.NORMAL].pack_ask
    policies = (CashOnlyPolicy(), BuyAndHoldPolicy(initial_cash=config.initial_cash,
                reference_price=reference_price), AlwaysOpenPolicy(initial_cash=config.initial_cash,
                reference_price=reference_price), EVOneStepPolicy(config))
    entries = []
    for policy in policies:
        stage = perf_counter()
        for spec in split.scenarios:
            directory = output / policy.policy_id / str(spec.seed)
            metadata = EvaluationMetadata(simulator_version=version, policy_id=policy.policy_id,
                model_id=None, training_seed=None, scenario_id=spec.scenario_id,
                scenario_seed=spec.seed, config_hash=config.config_hash, git_commit=commit)
            env = build_environment(config)
            if isinstance(policy, EVOneStepPolicy):
                evaluate_ev_episode(policy, env, metadata, output_directory=directory)
            else:
                result = evaluate_episode(policy, env, metadata)
                directory.mkdir(parents=True)
                write_evaluation_rows_jsonl(directory / "evaluation_rows.jsonl", (result.row,))
                write_step_traces_jsonl(directory / "step_traces.jsonl", result.step_traces)
            entry = {"policy_id": policy.policy_id, "scenario_id": spec.scenario_id,
                     "rows_file": str((directory / "evaluation_rows.jsonl").relative_to(output)),
                     "trace_file": str((directory / "step_traces.jsonl").relative_to(output))}
            if isinstance(policy, EVOneStepPolicy):
                entry["policy_metadata_file"] = str((directory / "policy_metadata.json").relative_to(output))
            entries.append(entry)
        timings[f"{policy.policy_id}_evaluation_seconds"] = perf_counter() - stage
    for spec in split.scenarios:
        entries.append({"policy_id": "DQN", "scenario_id": spec.scenario_id,
            "rows_file": str((dqn_output / "evaluation_rows.jsonl").relative_to(output)),
            "trace_file": str((dqn_output / "step_traces" /
                (spec.scenario_id.replace(":", "_") + ".jsonl")).relative_to(output))})
    _write_json(output / "comparison_manifest.json", {
        "purpose": "development pilot; not final-test evidence",
        "checkpoint_role": "development pilot" if "pilot_training_seconds" in timings else "supplied development checkpoint",
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split_manifest": str(split_path), "config_hash": config.config_hash,
        "scenario_version": SIMULATOR_VERSION, "simulator_version": version,
        "inventory_capacity": DEFAULT_INVENTORY_CAPACITY, "reference_price": reference_price,
        "git_commit": commit, "git_working_tree_dirty": dirty, "episodes": entries})
    stage = perf_counter()
    table = build_comparison(output)
    timings["saved_file_validation_and_table_seconds"] = perf_counter() - stage
    timings["total_workflow_seconds"] = perf_counter() - started
    _write_json(output / "runtime_budget.json", {
        "purpose": "measured development runtime budget, not a final experiment commitment",
        "scenarios": len(split.scenarios), "strategies": len(POLICY_IDS),
        "evaluation_environment_steps": len(split.scenarios) * len(POLICY_IDS) * config.horizon,
        "validation_replay_steps": len(split.scenarios) * len(POLICY_IDS) * config.horizon,
        "timings_seconds": timings})
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=ROOT / "configs/splits/validation.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run_comparison(split_manifest=args.split_manifest, checkpoint=args.checkpoint,
                         output_directory=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
