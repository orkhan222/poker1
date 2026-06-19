from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.evaluator import evaluate_policy
from poker_agent.features import load_training_examples
from poker_agent.model import load_policy
from poker_agent.policy_acceptance import EVCalibratedPolicy, evaluate_human_likeness, parse_seed_list, run_simulations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy acceptance gates from the project scope")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/policy_acceptance.json"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/policy_acceptance.md"))
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--allow-missing-hole-cards", action="store_true")
    parser.add_argument("--missing-hole-cards", choices=("drop", "flag", "keep"), default="drop")
    parser.add_argument("--keep-all-in-class", action="store_true")
    parser.add_argument("--simulation-seeds", default="11,23,37,41,53,67,79,83,97,101")
    parser.add_argument("--simulation-hands-per-seed", type=int, default=500)
    parser.add_argument("--simulation-player-count", type=int, default=6)
    parser.add_argument("--min-policy-macro-f1", type=float, default=0.50)
    parser.add_argument("--min-policy-lift", type=float, default=0.0)
    parser.add_argument("--max-js-divergence", type=float, default=0.08)
    parser.add_argument("--max-total-variation", type=float, default=0.20)
    parser.add_argument("--min-bucket-examples", type=int, default=50)
    parser.add_argument("--min-win-rate-high-confidence", type=float, default=0.505)
    parser.add_argument("--min-win-rate-median", type=float, default=0.52)
    parser.add_argument("--strategy-selector", choices=("base", "ev_calibrated"), default="ev_calibrated")
    parser.add_argument("--ev-weight", type=float, default=0.72)
    parser.add_argument("--ev-temperature", type=float, default=1.6)
    parser.add_argument("--min-ev-gain", type=float, default=0.02)
    return parser.parse_args()


def gate_status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    alignment = report["human_action_alignment"]
    base_alignment = report["base_human_action_alignment"]
    simulation = report["simulation"]
    likeness = report["human_likeness"]
    lines = [
        "# Policy Acceptance Report",
        "",
        "## Status",
        "",
        f"- Overall status: `{report['overall_status']}`",
        f"- Model: `{report['model']}`",
        f"- Dataset: `{report['dataset']}`",
        f"- Generated at: `{report['generated_at']}`",
        "",
        "## Human Action Alignment",
        "",
        "### Strategy Selector",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Strategy selector | {report['strategy_selector']} |",
        f"| Examples | {int(alignment['examples'])} |",
        f"| Accuracy | {alignment['accuracy']:.4f} |",
        f"| Macro F1 | {alignment['macro_f1']:.4f} |",
        f"| Weighted F1 | {alignment['weighted_f1']:.4f} |",
        f"| Cross entropy | {alignment['cross_entropy']:.4f} |",
        f"| Lift vs majority | {alignment['lift_vs_majority']:.4f} |",
        "",
        "### Base Model Reference",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {base_alignment['accuracy']:.4f} |",
        f"| Macro F1 | {base_alignment['macro_f1']:.4f} |",
        f"| Lift vs majority | {base_alignment['lift_vs_majority']:.4f} |",
        "",
        "## Synthetic Policy Simulation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Status | {simulation['status']} |",
        f"| Runs | {simulation.get('run_count', 0)} |",
        f"| Mean policy win rate | {float(simulation.get('mean_policy_win_rate', 0.0)):.4f} |",
        f"| Mean EV delta vs baseline | {float(simulation.get('mean_ev_delta_vs_baseline', 0.0)):.4f} |",
        f"| P(win rate >= 50.5%) | {float(simulation.get('probability_win_rate_at_least_50_5', 0.0)):.4f} |",
        f"| P(win rate >= 52%) | {float(simulation.get('probability_win_rate_at_least_52', 0.0)):.4f} |",
        "",
        "## Human-Likeness",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Status | {likeness['status']} |",
        f"| JS divergence | {likeness['js_divergence']:.4f} |",
        f"| Total variation | {likeness['total_variation']:.4f} |",
        f"| Timing / bet-size status | {likeness['timing_and_bet_size_status']} |",
        "",
        "## Interpretation",
        "",
        "This report separates measurable engineering readiness from strategy approval. "
        "The simulation is a synthetic policy proxy and is not a full poker engine. "
        "A true production claim still requires a validated self-play environment, "
        "bet-size labels, and timing-distribution labels.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    base_model = load_policy(args.model)
    strategy_model = (
        EVCalibratedPolicy(
            base_model,
            ev_weight=args.ev_weight,
            temperature=args.ev_temperature,
            min_ev_gain=args.min_ev_gain,
        )
        if args.strategy_selector == "ev_calibrated"
        else base_model
    )
    examples = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        require_hole_cards=not args.allow_missing_hole_cards,
        missing_hole_cards="flag" if args.allow_missing_hole_cards and args.missing_hole_cards == "drop" else args.missing_hole_cards,
        merge_all_in=not args.keep_all_in_class,
    )
    if not examples:
        raise SystemExit(f"No evaluation examples found in {args.dataset}")

    base_alignment = evaluate_policy(base_model, examples)
    alignment = evaluate_policy(strategy_model, examples)
    alignment_status = gate_status(
        float(alignment.get("macro_f1", 0.0)) >= args.min_policy_macro_f1
        and float(alignment.get("lift_vs_majority", -999.0)) >= args.min_policy_lift
    )
    human_likeness = evaluate_human_likeness(
        strategy_model,
        examples,
        max_js_divergence=args.max_js_divergence,
        max_total_variation=args.max_total_variation,
        min_bucket_examples=args.min_bucket_examples,
    )
    simulation = run_simulations(
        strategy_model,
        seeds=parse_seed_list(args.simulation_seeds),
        hands_per_seed=args.simulation_hands_per_seed,
        player_count=args.simulation_player_count,
        min_win_rate_high_confidence=args.min_win_rate_high_confidence,
        min_win_rate_median=args.min_win_rate_median,
    )

    statuses = [alignment_status, human_likeness["status"], simulation["status"]]
    overall_status = "PASS" if all(status == "PASS" for status in statuses) else "FAIL"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "model": str(args.model),
        "overall_status": overall_status,
        "strategy_selector": args.strategy_selector,
        "strategy_settings": {
            "ev_weight": args.ev_weight,
            "ev_temperature": args.ev_temperature,
            "min_ev_gain": args.min_ev_gain,
        },
        "human_action_alignment_status": alignment_status,
        "base_human_action_alignment": base_alignment,
        "human_action_alignment": alignment,
        "simulation": simulation,
        "human_likeness": human_likeness,
        "thresholds": {
            "min_policy_macro_f1": args.min_policy_macro_f1,
            "min_policy_lift": args.min_policy_lift,
            "max_js_divergence": args.max_js_divergence,
            "max_total_variation": args.max_total_variation,
            "min_win_rate_high_confidence": args.min_win_rate_high_confidence,
            "min_win_rate_median": args.min_win_rate_median,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.report_out)

    print(f"policy_acceptance_status={overall_status}")
    print(f"policy_alignment_status={alignment_status}")
    print(f"policy_alignment_accuracy={alignment['accuracy']:.4f}")
    print(f"policy_alignment_macro_f1={alignment['macro_f1']:.4f}")
    print(f"policy_simulation_status={simulation['status']}")
    print(f"policy_simulation_mean_win_rate={float(simulation.get('mean_policy_win_rate', 0.0)):.4f}")
    print(f"policy_human_likeness_status={human_likeness['status']}")
    print(f"policy_human_likeness_js={human_likeness['js_divergence']:.4f}")
    print(f"policy_acceptance_report={args.report_out}")
    print(f"policy_acceptance_json={args.out}")


if __name__ == "__main__":
    main()
