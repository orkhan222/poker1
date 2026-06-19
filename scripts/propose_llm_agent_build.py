from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.llm_agent_proposal import AgentBuildRecommendation, build_recommendation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the stakeholder approval package for the selected LLM agent")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--architecture-decision",
        type=Path,
        default=Path("reports/llm_agent_architecture_decision.json"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/qwen25_qlora"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/llm_agent_build_proposal.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/llm_agent_build_proposal.md"))
    parser.add_argument("--approval-status", default="proposed_for_stakeholder_approval")
    parser.add_argument(
        "--require-adapter",
        action="store_true",
        help="Fail if the selected model adapter artifact is not present.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be a mapping: {path}")
    return payload


def format_metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(recommendation: AgentBuildRecommendation, path: Path) -> None:
    payload = recommendation.to_dict()
    metrics = recommendation.architecture_metrics
    lines = [
        "# LLM Agent Build Proposal",
        "",
        "## Decision For Approval",
        "",
        f"- Recommended agent: `{recommendation.selected_architecture}`",
        f"- Recommended model: `{recommendation.selected_model}`",
        f"- Approval status: `{recommendation.approval_status}`",
        f"- Adapter artifact available: `{str(recommendation.adapter_available).lower()}`",
        f"- Ready for stakeholder review: `{str(recommendation.ready_for_stakeholder_review).lower()}`",
        "",
        "The recommended implementation is a bounded event-normalization agent, not an autonomous poker-playing agent. "
        "It converts noisy OCR/dealer-log text into schema-valid events and keeps the poker decision policy as a separate model.",
        "",
        "## Selected Architecture Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {format_metric(metrics, 'accuracy')} |",
        f"| Macro F1 | {format_metric(metrics, 'macro_f1')} |",
        f"| Schema validity | {format_metric(metrics, 'schema_validity_rate')} |",
        f"| Simulation readiness | {format_metric(metrics, 'simulation_readiness_rate')} |",
        f"| Average latency ms | {format_metric(metrics, 'average_latency_ms')} |",
        f"| Unmatched rate | {format_metric(metrics, 'unmatched_rate')} |",
        "",
        "## Architecture Options",
        "",
        "| Architecture | Production Fit | Operational Risk |",
        "| --- | --- | --- |",
    ]
    for option in payload["architecture_options"]:
        lines.append(f"| `{option['name']}` | {option['production_fit']} | {option['operational_risk']} |")

    lines.extend(
        [
            "",
            "## Model Candidates",
            "",
            "| Priority | Model | Params | Role | Expected Use |",
            "| ---: | --- | ---: | --- | --- |",
        ]
    )
    for model in payload["model_candidates"]:
        lines.append(
            f"| {model['priority']} | `{model['name']}` | {model['parameter_count']} | "
            f"{model['role']} | {model['expected_use']} |"
        )

    lines.extend(
        [
            "",
            "## Implementation Plan",
            "",
        ]
    )
    for item in recommendation.implementation_plan:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Acceptance Gates",
            "",
        ]
    )
    for item in recommendation.acceptance_gates:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Stakeholder Alignment",
            "",
            "Before building the production service path, the recommended agent type and model family should be approved. "
            "If the stakeholder wants a different model family, the same benchmark and schema gates should be rerun before implementation proceeds.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    decision_payload = load_json(args.architecture_decision)
    recommendation = build_recommendation(
        config=config,
        decision_payload=decision_payload,
        model_dir=args.model_dir,
        approval_status=args.approval_status,
    )

    if args.require_adapter and not recommendation.adapter_available:
        raise SystemExit(f"Adapter artifact is required but missing in {args.model_dir}")

    payload = recommendation.to_dict()
    payload.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": str(args.config),
            "architecture_decision": str(args.architecture_decision),
            "model_dir": str(args.model_dir),
        }
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(recommendation, args.out_md)

    print(f"llm_agent_recommended_architecture={recommendation.selected_architecture}")
    print(f"llm_agent_recommended_model={recommendation.selected_model}")
    print(f"llm_agent_approval_status={recommendation.approval_status}")
    print(f"llm_agent_adapter_available={str(recommendation.adapter_available).lower()}")
    print(f"llm_agent_ready_for_stakeholder_review={str(recommendation.ready_for_stakeholder_review).lower()}")
    print(f"llm_agent_build_proposal={args.out_md}")
    print(f"llm_agent_build_proposal_json={args.out_json}")


if __name__ == "__main__":
    main()
