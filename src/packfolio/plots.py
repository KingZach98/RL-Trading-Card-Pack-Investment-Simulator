"""PF-24 saved-data figures; no training or policy evaluation.

Reproduce: python -B -m packfolio.plots --output results/figures
Examples: nearest to each strategy's median final value, lowest seed on ties.
Training lines are exploratory 2000-step histories, not convergence evidence.
"""

import argparse
import gzip
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from packfolio.evaluate import read_evaluation_rows_jsonl, read_step_traces_jsonl
from packfolio.held_out import BASELINES, ROOT, _digest, _json, bootstrap_mean_interval, summarize_rows
from packfolio.types import Action, EvaluationRow


FIGURES = ("training_history", "final_values", "downside", "action_behaviour",
           "fee_sensitivity", "episode_examples")


def select_episode(rows):
    """Nearest final value to the strategy median; lowest scenario seed on ties."""
    if not rows:
        raise ValueError("episode selection requires records")
    median = float(np.median([r.final_portfolio_value for r in rows]))
    return min(rows, key=lambda r: (abs(r.final_portfolio_value - median), r.scenario_seed))


def metrics(rows, rules, reference=None):
    rows = sorted(rows, key=lambda r: r.scenario_seed)
    values = [r.final_portfolio_value for r in rows]
    kwargs = {"seed": rules["bootstrap_seed"], "resamples": rules["bootstrap_resamples"],
              "confidence": rules["confidence"]}
    low, high = bootstrap_mean_interval(values, **kwargs)
    result = {**summarize_rows(rows, loss_tolerance=rules["loss_tolerance"]),
              "ci_low": low, "ci_high": high}
    if reference is not None:
        reference = sorted(reference, key=lambda r: r.scenario_seed)
        if [r.scenario_seed for r in rows] != [r.scenario_seed for r in reference]:
            raise ValueError("paired scenarios do not match")
        base = [r.final_portfolio_value for r in reference]
        result["paired_change"] = float(np.mean(np.array(values) - base))
        result["paired_low"], result["paired_high"] = bootstrap_mean_interval(values, base, **kwargs)
    return result


def _history(directory):
    path = directory / "training_metrics.jsonl"
    # Training logs lack an evaluation-record digest; verify their committed
    # bytes with newline normalization, not a hash invented after reading them.
    relative = path.relative_to(ROOT).as_posix()
    original = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=ROOT)
    if path.read_bytes().replace(b"\r\n", b"\n") != original.replace(b"\r\n", b"\n"):
        raise ValueError("training log differs from committed evidence")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if (len(rows) != 20 or [r["episode"] for r in rows] != list(range(1, 21))
            or sum(r["episode_steps"] for r in rows) != 2000
            or not all(r["completed"] for r in rows)
            or not all(np.isfinite([r["episode_reward"], r["mean_loss"]]).all() for r in rows)):
        raise ValueError("incomplete or invalid training history")
    return rows


def load_data(root=ROOT):
    """Validate recorded hashes and complete groups before creating any figures.

    Historical absolute paths must remain accessible in Phase 1. No manifests
    are relocated or rewritten; portable release work is a separate phase.
    """
    root = Path(root)
    publication = _json(root / "results/fee_sensitivity_manifest.json")
    corrected = root / publication["corrected_run"]["directory"]
    for path, digest in (
        (root / publication["official_csv"]["path"], publication["official_csv"]["sha256"]),
        (corrected / "manifest.json", publication["corrected_run"]["manifest_sha256"]),
        (corrected / "episodes.jsonl.gz", publication["corrected_run"]["episodes_sha256"]),
        (root / publication["evidence_archive"]["path"], publication["evidence_archive"]["sha256"]),
    ):
        if _digest(path) != digest:
            raise ValueError("publication artifact hash mismatch")
    fee_manifest = _json(corrected / "manifest.json")
    for path, digest in fee_manifest["source_sha256"].items():
        if _digest(path) != digest:
            raise ValueError("recorded input hash mismatch")
    reference = root / "results"
    manifest = _json(reference / "held_out_manifest.json")
    seeds = set(_json(manifest["split_manifest"])["seeds"])
    expected = {(p, None) for p in BASELINES} | {("DQN", m["training_seed"]) for m in manifest["models"]}
    groups, traces = {}, {}
    for group in manifest["groups"]:
        key = group["policy_id"], group["training_seed"]
        if key not in expected or (0.05, key) in groups:
            raise ValueError("unexpected or duplicate strategy")
        rows = read_evaluation_rows_jsonl(reference / group["rows_file"])
        groups[0.05, key] = list(rows)
        for row in rows:
            if (row.policy_id, row.training_seed) != key or row.config_hash != manifest["config_hash"]:
                raise ValueError("incompatible reference row")
        traces[key] = {sid: reference / path for sid, path in group["trace_files"].items()}
    for fee in (0.025, 0.1):
        for key in expected:
            groups[fee, key] = []
    with gzip.open(corrected / "episodes.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            row = EvaluationRow.from_dict(record["row"])
            key = record["selling_fee_rate"], (row.policy_id, row.training_seed)
            if key not in groups or key[0] == 0.05:
                raise ValueError("unexpected fee episode")
            groups[key].append(row)
    if len(groups) != 27:
        raise ValueError("missing strategies")
    for (fee, key), rows in groups.items():
        if len(rows) != len(seeds) or {r.scenario_seed for r in rows} != seeds:
            raise ValueError("missing or duplicate scenarios")
        model = next((m for m in manifest["models"] if m["training_seed"] == key[1]), None)
        if any(r.model_id != (model["model_id"] if model else None) for r in rows):
            raise ValueError("model identity mismatch")
        config_hash = fee_manifest["variant_provenance"][str(fee)]["evaluation_config_hash"]
        if any(r.config_hash != config_hash or r.episode_steps != 100
               or r.scenario_id != f"{config_hash}:{r.scenario_seed}" for r in rows):
            raise ValueError("configuration or episode identity mismatch")
    histories = {}
    for model in manifest["models"]:
        directory = root / "runs/pf21-final-replicas" / model["model_id"]
        if _digest(directory / "checkpoint.pt") != publication["checkpoint_sha256"][model["model_id"]]:
            raise ValueError("checkpoint hash mismatch")
        histories[model["training_seed"]] = _history(directory)
    return groups, traces, histories, manifest["statistics"], fee_manifest["statistics"]


def _label(key):
    return f"DQN {key[1]}" if key[0] == "DQN" else (
        "EV_ONE_STEP\n(model-informed)" if key[0] == "EV_ONE_STEP" else key[0])


def generate_figures(root=ROOT, output=None):
    """Generate six PNGs from verified saved evidence; never overwrite outputs."""
    output = Path(output) if output is not None else Path(root) / "results/figures"
    if output.exists():
        raise ValueError("figure directory exists; use a new output directory")
    groups, traces, histories, base_rules, fee_rules = load_data(root)
    keys = sorted((k for f, k in groups if f == 0.05), key=lambda k: (k[0], k[1] or -1))
    base = {k: metrics(groups[0.05, k], base_rules) for k in keys}
    fees = {(f, k): metrics(groups[f, k], fee_rules, groups[0.05, k])
            for f in (0.025, 0.05, 0.1) for k in keys}
    examples = {}
    for key in keys:
        row = select_episode(groups[0.05, key])
        steps = read_step_traces_jsonl(traces[key][row.scenario_id])
        if (len(steps) != row.episode_steps or steps[-1].portfolio_value_after != row.final_portfolio_value
                or [s.step_index for s in steps] != list(range(row.episode_steps))):
            raise ValueError("selected example trace mismatch")
        examples[key] = row, steps
    output.mkdir(parents=True)
    colors = plt.get_cmap("tab10")(np.arange(len(keys)))
    labels = [_label(k) for k in keys]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        def save(fig, name):
            fig.savefig(output / (name + ".png"), dpi=150, metadata={"Software": "packfolio PF-24"})
            plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for seed, rows in sorted(histories.items()):
            steps = np.cumsum([r["episode_steps"] for r in rows])
            axes[0].plot(steps, [r["episode_reward"] for r in rows], label=str(seed))
            axes[1].plot(steps, [r["mean_loss"] for r in rows])
        axes[0].set_ylabel("Recorded exploratory episode return")
        axes[1].set_ylabel("Recorded episode-mean training loss")
        axes[0].legend(title="Training seed", fontsize=8)
        for ax in axes: ax.set_xlabel("Cumulative environment steps")
        fig.suptitle("Exploratory 2,000-step training histories - not convergence evidence")
        save(fig, "training_history")

        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        means = np.array([base[k]["mean_final_value"] for k in keys])
        ax.barh(labels, means, color=colors)
        ax.errorbar(means, np.arange(len(keys)),
            xerr=[means-[base[k]["ci_low"] for k in keys], [base[k]["ci_high"] for k in keys]-means],
            fmt="none", color="black", capsize=3)
        ax.axvline(100, color="grey", linestyle="--")
        ax.set(xlabel="Mean final portfolio value", title="5% fee: 95% scenario-bootstrap intervals; fixed models")
        save(fig, "final_values")

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
        for ax, metric, title in zip(axes, ["loss_probability", "fifth_percentile_final_value"],
                                     ["Loss probability", "Fifth-percentile final value (linear quantile)"]):
            ax.barh(labels, [base[k][metric] for k in keys], color=colors)
            ax.set_title(title)
        axes[0].set_xlabel("Fraction ending below initial cash (tolerance 1e-9)")
        axes[1].set_xlabel("Final portfolio value")
        fig.suptitle("5% fee: downside for every strategy and replica")
        save(fig, "downside")

        fig, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
        for ax, kind in zip(axes[:2], ["requested", "executed"]):
            left = np.zeros(len(keys))
            for action, color in zip(Action, plt.get_cmap("Set2").colors):
                values = np.array([base[k][kind+"_"+action.name.lower()+"_rate"] for k in keys])
                ax.barh(labels, values, left=left, color=color, label=action.name)
                left += values
            ax.set(xlim=(0, 1), xlabel="Fraction of action steps", title=kind.title()+" actions")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=4, fontsize=9)
        axes[2].barh(labels, [base[k]["infeasible_action_rate"] for k in keys], color=colors)
        axes[2].set(xlabel="Fraction of requests", title="Infeasible requests executed as HOLD", xlim=(0, 1))
        fig.suptitle("5% fee: requested actions are not executed actions")
        save(fig, "action_behaviour")

        fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
        rates = [2.5, 5, 10]
        for key, color in zip(keys, colors):
            rows = [fees[f, key] for f in (0.025, 0.05, 0.1)]
            axes[0].plot(rates, [r["mean_final_value"] for r in rows], marker="o", color=color, label=_label(key))
            change = np.array([r["paired_change"] for r in rows])
            axes[1].errorbar(rates, change,
                yerr=[change-[r["paired_low"] for r in rows], [r["paired_high"] for r in rows]-change],
                marker="o", color=color, capsize=2)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9)
        axes[0].set_ylabel("Mean final portfolio value")
        axes[1].set_ylabel("Paired mean change versus 5% (95% scenario CI)")
        axes[1].axhline(0, color="grey", linestyle="--")
        for ax in axes: ax.set_xticks(rates); ax.set_xlabel("Selling fee (%)")
        fig.suptitle("Frozen DQN weights; EV_ONE_STEP model-informed and fee-recomputed")
        save(fig, "fee_sensitivity")

        fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout="constrained")
        for ax, key, color in zip(axes.flat, keys, colors):
            row, steps = examples[key]
            ax.plot(range(len(steps)+1), [steps[0].portfolio_value_before]+[s.portfolio_value_after for s in steps], color=color)
            ax.axhline(row.initial_portfolio_value, color="grey", linestyle="--")
            ax.set(title=_label(key)+f" | seed {row.scenario_seed}", xlabel="Step", ylabel="Net portfolio value")
        fig.suptitle("Per-strategy nearest-to-median episode; lowest seed breaks ties (not paired examples)")
        save(fig, "episode_examples")
    return [output / (name + ".png") for name in FIGURES]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args(argv)
    for path in generate_figures(output=args.output): print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
