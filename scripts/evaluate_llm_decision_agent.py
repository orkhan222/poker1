from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.decision_context import describe_prompt_profile
from poker_agent.features import load_training_examples
from poker_agent.llm_decision import (
    DECISION_ACTIONS,
    HeuristicTextProvider,
    LLMDecisionAgent,
    TransformersTextProvider,
    request_to_jsonable,
)
from poker_agent.schemas import PredictionRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the text-model poker decision baseline")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", default=Path("reports/llm_decision_agent_eval.json"), type=Path)
    parser.add_argument("--predictions-out", default=Path("reports/llm_decision_agent_predictions.jsonl"), type=Path)
    parser.add_argument("--report-out", default=Path("reports/llm_decision_agent_report.md"), type=Path)
    parser.add_argument("--prompt-report-out", default=Path("reports/llm_decision_prompt_contract.md"), type=Path)
    parser.add_argument(
        "--provider",
        choices=("heuristic_text", "transformers"),
        default="heuristic_text",
    )
    parser.add_argument(
        "--mode",
        choices=("candidate_ranker", "freeform"),
        default="candidate_ranker",
    )
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt-profile",
        choices=("minimal", "rules_zero_shot", "rules_few_shot", "candidate_ranker"),
        default="rules_zero_shot",
    )
    parser.add_argument("--few-shot-count", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=250)
    parser.add_argument(
        "--allow-missing-hole-cards",
        action="store_true",
        help="Keep rows where OCR did not capture two hole cards.",
    )
    parser.add_argument(
        "--missing-hole-cards",
        choices=("drop", "flag", "keep"),
        default="drop",
    )
    parser.add_argument(
        "--keep-all-in-class",
        action="store_true",
        help="Keep all_in as a separate target label. Default merges it into raise.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_provider(args: argparse.Namespace) -> HeuristicTextProvider | TransformersTextProvider:
    if args.provider == "heuristic_text":
        return HeuristicTextProvider()
    return TransformersTextProvider(
        model_id=args.model_id,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    probabilities: list[dict[str, float]],
) -> dict[str, Any]:
    labels = sorted(set(DECISION_ACTIONS) | set(y_true) | set(y_pred))
    true_counts = Counter(y_true)
    predicted_counts = Counter(y_pred)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    eps = 1e-12

    per_class: dict[str, dict[str, float]] = {}
    weighted_f1 = 0.0
    balanced_accuracy = 0.0
    for label in labels:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = float(true_counts[label])
        weighted_f1 += f1 * support
        balanced_accuracy += recall
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    examples = len(y_true)
    cross_entropy = 0.0
    brier_loss = 0.0
    for true, row in zip(y_true, probabilities):
        cross_entropy += -math_log(max(row.get(true, 0.0), eps))
        for label in labels:
            target = 1.0 if label == true else 0.0
            brier_loss += (row.get(label, 0.0) - target) ** 2
    cross_entropy = cross_entropy / examples if examples else 0.0
    brier_loss = brier_loss / (examples * len(labels)) if examples and labels else 0.0

    confusion = {
        "labels": labels,
        "matrix": [
            [sum(1 for true, pred in zip(y_true, y_pred) if true == row_label and pred == col_label) for col_label in labels]
            for row_label in labels
        ],
    }

    majority = true_counts.most_common(1)[0][1] / examples if examples else 0.0
    return {
        "examples": float(examples),
        "accuracy": correct / examples if examples else 0.0,
        "balanced_accuracy": balanced_accuracy / len(labels) if labels else 0.0,
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(labels) if labels else 0.0,
        "weighted_f1": weighted_f1 / examples if examples else 0.0,
        "cross_entropy": cross_entropy,
        "brier_loss": brier_loss,
        "majority_baseline_accuracy": majority,
        "lift_vs_majority": (correct / examples - majority) if examples else 0.0,
        "class_counts": dict(sorted(true_counts.items())),
        "predicted_class_counts": dict(sorted(predicted_counts.items())),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def math_log(value: float) -> float:
    import math

    return math.log(value)


def write_report(payload: dict[str, Any], path: Path) -> None:
    metrics = payload["metrics"]
    latency = payload["latency_ms"]
    lines = [
        "# Text-Model Poker Decision Baseline",
        "",
        "## Objective",
        "",
        "Evaluate a constrained text-model decision pipeline on historical poker actions. "
        "The model receives structured game state and returns one discrete action.",
        "",
        "## Configuration",
        "",
        f"- Provider: `{payload['provider']}`",
        f"- Mode: `{payload['mode']}`",
        f"- Prompt profile: `{payload['prompt_profile']}`",
        f"- Few-shot examples: `{payload['prompt_contract']['few_shot_count']}`",
        f"- Model ID: `{payload['model_id']}`",
        f"- Examples: `{int(metrics['examples'])}`",
        f"- Seed: `{payload['seed']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Macro F1 | {metrics['macro_f1']:.4f} |",
        f"| Weighted F1 | {metrics['weighted_f1']:.4f} |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.4f} |",
        f"| Cross entropy | {metrics['cross_entropy']:.4f} |",
        f"| Brier loss | {metrics['brier_loss']:.4f} |",
        f"| Majority baseline accuracy | {metrics['majority_baseline_accuracy']:.4f} |",
        f"| Lift vs majority | {metrics['lift_vs_majority']:.4f} |",
        f"| Invalid output rate | {payload['invalid_output_rate']:.4f} |",
        "",
        "## Latency",
        "",
        f"- Mean: `{latency['mean']:.2f} ms`",
        f"- P50: `{latency['p50']:.2f} ms`",
        f"- P95: `{latency['p95']:.2f} ms`",
        "",
        "## Notes",
        "",
        "This baseline is intended for comparison against the supervised policy model. "
        "The recommended production path remains a schema-routed extractor for noisy logs "
        "and a separately validated policy model for poker decisions.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prompt_report(payload: dict[str, Any], path: Path) -> None:
    contract = payload["prompt_contract"]
    sample_prompt = payload.get("sample_prompt", "")
    lines = [
        "# LLM Decision Prompt Contract",
        "",
        "## Objective",
        "",
        "Define the in-context learning contract for out-of-the-box LLM decision baselines. "
        "The zero-shot profile is not context-free: it includes the task, formal poker rules, legal-action constraints, and strict JSON output requirements.",
        "",
        "## Configuration",
        "",
        f"- Prompt profile: `{contract['profile']}`",
        f"- Includes poker rules: `{str(contract['include_rules']).lower()}`",
        f"- Includes formal constraints: `{str(contract['include_guidelines']).lower()}`",
        f"- Few-shot examples: `{contract['few_shot_count']}`",
        f"- Rules count: `{contract['rules_count']}`",
        f"- Guidelines count: `{contract['guidelines_count']}`",
        "",
        "## Output Schema",
        "",
        "```json",
        json.dumps(contract["output_schema"], indent=2, sort_keys=True),
        "```",
        "",
        "## Sample Prompt",
        "",
        "```text",
        sample_prompt,
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    records = load_training_examples(
        args.dataset,
        max_examples=args.max_examples,
        require_hole_cards=not args.allow_missing_hole_cards,
        missing_hole_cards=args.missing_hole_cards,
        merge_all_in=not args.keep_all_in_class,
        include_hand_id=True,
        include_request=True,
    )
    if not records:
        raise SystemExit(f"No decision examples found in {args.dataset}")

    provider = build_provider(args)
    prompt_profile = args.prompt_profile
    if args.mode == "candidate_ranker" and prompt_profile == "rules_zero_shot":
        prompt_profile = "candidate_ranker"
    agent = LLMDecisionAgent(
        provider=provider,
        mode=args.mode,
        prompt_profile=prompt_profile,
        few_shot_count=args.few_shot_count,
    )
    y_true: list[str] = []
    y_pred: list[str] = []
    probabilities: list[dict[str, float]] = []
    latencies: list[float] = []
    invalid_count = 0
    prediction_rows: list[dict[str, Any]] = []
    sample_prompt = ""

    for index, record in enumerate(records):
        request, _features, label, hand_id = record
        if not isinstance(request, PredictionRequest):
            raise TypeError("Expected load_training_examples(..., include_request=True) to return PredictionRequest")
        response = agent.predict(request)
        if not sample_prompt:
            sample_prompt = str(getattr(response, "prompt", ""))
        invalid = any("valid action" in warning for warning in response.warnings)
        invalid_count += int(invalid)
        y_true.append(label)
        y_pred.append(response.action)
        probabilities.append(response.probabilities)
        latency_ms = float(getattr(response, "latency_ms", 0.0))
        latencies.append(latency_ms)
        prediction_rows.append(
            {
                "index": index,
                "hand_id": hand_id,
                "expected": label,
                "predicted": response.action,
                "probabilities": response.probabilities,
                "confidence": response.confidence,
                "latency_ms": latency_ms,
                "invalid_output": invalid,
                "request": request_to_jsonable(request),
                "raw_text": getattr(response, "raw_text", ""),
                "prompt_profile": prompt_profile,
            }
        )

    metrics = classification_metrics(y_true, y_pred, probabilities)
    latency = {
        "mean": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "max": max(latencies, default=0.0),
    }
    payload = {
        "dataset": str(args.dataset),
        "provider": args.provider,
        "mode": args.mode,
        "model_id": args.model_id if args.provider == "transformers" else "local_scoring_policy",
        "device": args.device,
        "seed": args.seed,
        "settings": {
            "max_examples": args.max_examples,
            "allow_missing_hole_cards": args.allow_missing_hole_cards,
            "missing_hole_cards": args.missing_hole_cards,
            "keep_all_in_class": args.keep_all_in_class,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "prompt_profile": prompt_profile,
            "few_shot_count": args.few_shot_count,
        },
        "prompt_profile": prompt_profile,
        "prompt_contract": describe_prompt_profile(prompt_profile, few_shot_count=args.few_shot_count),
        "sample_prompt": sample_prompt,
        "metrics": metrics,
        "latency_ms": latency,
        "invalid_output_rate": invalid_count / len(records) if records else 0.0,
        "estimated_cost_usd": 0.0,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions_out.open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_report(payload, args.report_out)
    write_prompt_report(payload, args.prompt_report_out)

    print(f"examples={int(metrics['examples'])}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"cross_entropy={metrics['cross_entropy']:.4f}")
    print(f"invalid_output_rate={payload['invalid_output_rate']:.4f}")
    print(f"mean_latency_ms={latency['mean']:.2f}")
    print(f"out={args.out}")
    print(f"report_out={args.report_out}")
    print(f"prompt_profile={prompt_profile}")
    print(f"prompt_report_out={args.prompt_report_out}")


if __name__ == "__main__":
    main()
