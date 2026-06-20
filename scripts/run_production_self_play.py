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

from poker_agent.holdem_self_play import run_holdem_self_play
from poker_agent.model import load_policy
from poker_agent.policy_acceptance import DeploymentGatedPolicy, EVCalibratedPolicy, parse_seed_list


DEFAULT_PRODUCTION_SEEDS = "11,23,37,41,53,67,79,83,97,101,113,127,131,149,157,163,179,191,197,211"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-scale validated Hold'em self-play")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/production_self_play.json"))
    parser.add_argument("--report-out", type=Path, default=Path("reports/production_self_play.md"))
    parser.add_argument("--strategy-selector", choices=("base", "ev_calibrated", "deployment_gated"), default="deployment_gated")
    parser.add_argument("--seeds", default=DEFAULT_PRODUCTION_SEEDS)
    parser.add_argument("--hands-per-seed", type=int, default=250)
    parser.add_argument("--player-count", type=int, default=6)
    parser.add_argument("--min-paired-hands", type=int, default=5000)
    parser.add_argument("--min-win-rate-high-confidence", type=float, default=0.505)
    parser.add_argument("--min-win-rate-median", type=float, default=0.52)
    return parser.parse_args()


def build_strategy_model(args: argparse.Namespace) -> Any:
    base_model = load_policy(args.model)
    if args.strategy_selector == "ev_calibrated":
        return EVCalibratedPolicy(base_model)
    if args.strategy_selector == "deployment_gated":
        return DeploymentGatedPolicy(base_model)
    return base_model


def paired_hands(report: dict[str, Any]) -> int:
    total = 0
    for row in report.get("runs", []):
        total += int(row.get("hands", 0))
    return total


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Production Self-Play Report",
        "",
        "## Status",
        "",
        f"- Status: `{payload['status']}`",
        f"- Production scale status: `{payload['production_scale_status']}`",
        f"- Strategy selector: `{payload['strategy_selector']}`",
        f"- Paired hands: `{payload['paired_hands']}`",
        f"- Required paired hands: `{payload['min_paired_hands']}`",
        f"- Mean policy win rate: `{payload['mean_policy_win_rate']:.4f}`",
        f"- P(win rate >= 50.5%): `{payload['probability_win_rate_at_least_50_5']:.4f}`",
        f"- P(win rate >= 52%): `{payload['probability_win_rate_at_least_52']:.4f}`",
        "",
        "## Environment",
        "",
        f"- Simulation type: `{payload['simulation_type']}`",
        f"- Environment: `{payload.get('environment', {}).get('name', 'unknown')}`",
        f"- Showdown evaluator: `{payload.get('environment', {}).get('showdown_evaluator', 'unknown')}`",
        "",
        "## Interpretation",
        "",
        "This report validates the strategy selector in a deterministic bounded Hold'em self-play environment. "
        "It is a production-scale simulation gate, but it does not replace the raw supervised model gate or live monitoring.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    seeds = parse_seed_list(args.seeds)
    model = build_strategy_model(args)
    report = run_holdem_self_play(
        model,
        seeds=seeds,
        hands_per_seed=args.hands_per_seed,
        player_count=args.player_count,
        min_win_rate_high_confidence=args.min_win_rate_high_confidence,
        min_win_rate_median=args.min_win_rate_median,
    )
    total_hands = paired_hands(report)
    scale_passed = total_hands >= args.min_paired_hands
    production_scale_status = "PASS" if scale_passed and report.get("status") == "PASS" else "FAIL"
    payload = {
        **report,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "strategy_selector": args.strategy_selector,
        "seeds": seeds,
        "hands_per_seed": args.hands_per_seed,
        "paired_hands": total_hands,
        "min_paired_hands": args.min_paired_hands,
        "production_scale_status": production_scale_status,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(payload, args.report_out)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "production_scale_status": production_scale_status,
                "paired_hands": total_hands,
                "mean_policy_win_rate": payload["mean_policy_win_rate"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if production_scale_status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
