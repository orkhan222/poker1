from __future__ import annotations

import argparse
import csv
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


DEFAULT_PROFILES = ("minimal", "rules_zero_shot", "rules_few_shot", "candidate_ranker")
PROFILE_TIEBREAK_PRIORITY = {
    "minimal": 0,
    "rules_zero_shot": 1,
    "rules_few_shot": 2,
    "candidate_ranker": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM decision prompts with different in-context levels")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-json", default=Path("reports/llm_context_ablation.json"), type=Path)
    parser.add_argument("--out-csv", default=Path("reports/llm_context_ablation.csv"), type=Path)
    parser.add_argument("--report-out", default=Path("reports/llm_context_ablation.md"), type=Path)
    parser.add_argument("--predictions-out", default=Path("reports/llm_context_ablation_predictions.jsonl"), type=Path)
    parser.add_argument("--provider", choices=("heuristic_text", "transformers"), default="heuristic_text")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--few-shot-count", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=250)
    parser.add_argument("--allow-missing-hole-cards", action="store_true")
    parser.add_argument("--missing-hole-cards", choices=("drop", "flag", "keep"), default="drop")
    parser.add_argument("--keep-all-in-class", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_profiles(raw: str) -> list[str]:
    profiles = [profile.strip() for profile in raw.split(",") if profile.strip()]
    invalid = sorted(set(profiles) - set(DEFAULT_PROFILES))
    if invalid:
        raise ValueError(f"Unsupported prompt profiles: {', '.join(invalid)}")
    return profiles or list(DEFAULT_PROFILES)


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
    correct = sum(1 for true, predicted in zip(y_true, y_pred) if true == predicted)
    examples = len(y_true)

    per_class: dict[str, dict[str, float]] = {}
    weighted_f1 = 0.0
    balanced_accuracy = 0.0
    for label in labels:
        tp = sum(1 for true, predicted in zip(y_true, y_pred) if true == label and predicted == label)
        fp = sum(1 for true, predicted in zip(y_true, y_pred) if true != label and predicted == label)
        fn = sum(1 for true, predicted in zip(y_true, y_pred) if true == label and predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = float(true_counts[label])
        weighted_f1 += f1 * support
        balanced_accuracy += recall
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    cross_entropy = 0.0
    brier_loss = 0.0
    for true, row in zip(y_true, probabilities):
        cross_entropy += -_log(max(float(row.get(true, 0.0)), 1e-12))
        for label in labels:
            target = 1.0 if label == true else 0.0
            brier_loss += (float(row.get(label, 0.0)) - target) ** 2

    majority = true_counts.most_common(1)[0][1] / examples if examples else 0.0
    accuracy = correct / examples if examples else 0.0
    return {
        "examples": float(examples),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy / len(labels) if labels else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(labels) if labels else 0.0,
        "weighted_f1": weighted_f1 / examples if examples else 0.0,
        "cross_entropy": cross_entropy / examples if examples else 0.0,
        "brier_loss": brier_loss / (examples * len(labels)) if examples and labels else 0.0,
        "majority_baseline_accuracy": majority,
        "lift_vs_majority": accuracy - majority,
        "class_counts": dict(sorted(true_counts.items())),
        "predicted_class_counts": dict(sorted(predicted_counts.items())),
        "per_class": per_class,
    }


def _log(value: float) -> float:
    import math

    return math.log(value)


def profile_mode(profile: str) -> str:
    return "candidate_ranker" if profile == "candidate_ranker" else "freeform"


def profile_few_shot_count(profile: str, default_count: int) -> int:
    if profile in {"rules_few_shot", "candidate_ranker"}:
        return default_count
    return 0


def best_profile_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["accuracy"]),
            -float(row["invalid_output_rate"]),
            PROFILE_TIEBREAK_PRIORITY.get(str(row["profile"]), -1),
        ),
    )


def evaluate_profile(
    *,
    profile: str,
    provider: HeuristicTextProvider | TransformersTextProvider,
    records: list[Any],
    few_shot_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mode = profile_mode(profile)
    agent = LLMDecisionAgent(
        provider=provider,
        mode=mode,
        prompt_profile=profile,
        few_shot_count=few_shot_count,
    )
    y_true: list[str] = []
    y_pred: list[str] = []
    probabilities: list[dict[str, float]] = []
    latencies: list[float] = []
    prompt_lengths: list[int] = []
    invalid_count = 0
    rows: list[dict[str, Any]] = []
    sample_prompt = ""

    for index, record in enumerate(records):
        request, _features, label, hand_id = record
        if not isinstance(request, PredictionRequest):
            raise TypeError("Expected PredictionRequest from load_training_examples(include_request=True)")
        response = agent.predict(request)
        prompt = str(getattr(response, "prompt", ""))
        if not sample_prompt:
            sample_prompt = prompt
        prompt_lengths.append(len(prompt))
        invalid = any("valid action" in warning for warning in response.warnings)
        invalid_count += int(invalid)
        y_true.append(label)
        y_pred.append(response.action)
        probabilities.append(response.probabilities)
        latency_ms = float(getattr(response, "latency_ms", 0.0))
        latencies.append(latency_ms)
        rows.append(
            {
                "profile": profile,
                "mode": mode,
                "index": index,
                "hand_id": hand_id,
                "expected": label,
                "predicted": response.action,
                "probabilities": response.probabilities,
                "confidence": response.confidence,
                "latency_ms": latency_ms,
                "invalid_output": invalid,
                "prompt_chars": len(prompt),
                "request": request_to_jsonable(request),
                "raw_text": getattr(response, "raw_text", ""),
            }
        )

    metrics = classification_metrics(y_true, y_pred, probabilities)
    contract = describe_prompt_profile(profile, few_shot_count=few_shot_count)
    summary = {
        "profile": profile,
        "mode": mode,
        "prompt_contract": contract,
        "metrics": metrics,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
        "prompt_chars": {
            "mean": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0,
            "sample": len(sample_prompt),
            "max": max(prompt_lengths, default=0),
        },
        "invalid_output_rate": invalid_count / len(records) if records else 0.0,
        "sample_prompt": sample_prompt,
    }
    return summary, rows


def comparison_row(summary: dict[str, Any], provider: str, model_id: str) -> dict[str, Any]:
    metrics = summary["metrics"]
    latency = summary["latency_ms"]
    contract = summary["prompt_contract"]
    prompt_chars = summary["prompt_chars"]
    return {
        "profile": summary["profile"],
        "mode": summary["mode"],
        "provider": provider,
        "model_id": model_id,
        "examples": int(metrics["examples"]),
        "accuracy": round(float(metrics["accuracy"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "weighted_f1": round(float(metrics["weighted_f1"]), 6),
        "cross_entropy": round(float(metrics["cross_entropy"]), 6),
        "invalid_output_rate": round(float(summary["invalid_output_rate"]), 6),
        "mean_latency_ms": round(float(latency["mean"]), 4),
        "mean_prompt_chars": round(float(prompt_chars["mean"]), 2),
        "rules_count": int(contract["rules_count"]),
        "guidelines_count": int(contract["guidelines_count"]),
        "few_shot_count": int(contract["few_shot_count"]),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    best = payload["recommended_profile"]
    lines = [
        "# LLM Decision Context Ablation",
        "",
        "## Objective",
        "",
        "Compare how different in-context learning contracts affect out-of-the-box LLM decision-agent behavior.",
        "",
        "## Results",
        "",
        "| Profile | Mode | Accuracy | Macro F1 | Invalid Output | Prompt Chars | Rules | Few-Shot |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["comparison"]:
        lines.append(
            f"| `{row['profile']}` | `{row['mode']}` | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['invalid_output_rate']:.4f} | {row['mean_prompt_chars']:.0f} | {row['rules_count']} | {row['few_shot_count']} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Profile",
            "",
            f"`{best['profile']}` is selected for the current run. Selection is based on macro F1, accuracy, invalid-output rate, "
            "and a production-safety tie-breaker that prefers constrained candidate ranking when metrics are tied.",
            "",
            "## Interpretation",
            "",
            "The minimal profile measures underspecified prompting. The rule-guided zero-shot profile keeps the evaluation zero-shot but supplies formal poker rules and output constraints. "
            "The few-shot profile adds representative decisions. The candidate-ranking profile constrains the action space and is the recommended out-of-the-box LLM-agent format for controlled evaluation.",
            "",
            "The default local scoring provider is deterministic and does not change behavior from prompt wording alone. Use `model.provider=transformers` with a local instruction model to measure actual prompt sensitivity.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    profiles = parse_profiles(args.profiles)
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
    model_id = args.model_id if args.provider == "transformers" else "local_scoring_policy"
    summaries: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for profile in profiles:
        summary, rows = evaluate_profile(
            profile=profile,
            provider=provider,
            records=records,
            few_shot_count=profile_few_shot_count(profile, args.few_shot_count),
        )
        summaries.append(summary)
        prediction_rows.extend(rows)

    comparison = [comparison_row(summary, args.provider, model_id) for summary in summaries]
    recommended = best_profile_row(comparison)
    payload = {
        "dataset": str(args.dataset),
        "provider": args.provider,
        "model_id": model_id,
        "seed": args.seed,
        "profiles": profiles,
        "settings": {
            "max_examples": args.max_examples,
            "few_shot_count": args.few_shot_count,
            "allow_missing_hole_cards": args.allow_missing_hole_cards,
            "missing_hole_cards": args.missing_hole_cards,
            "keep_all_in_class": args.keep_all_in_class,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
        "comparison": comparison,
        "recommended_profile": recommended,
        "profile_results": summaries,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(comparison, args.out_csv)
    write_predictions(prediction_rows, args.predictions_out)
    write_markdown(payload, args.report_out)

    print(f"context_ablation_profiles={','.join(profiles)}")
    print(f"context_ablation_best_profile={recommended.get('profile', '')}")
    print(f"context_ablation_best_accuracy={float(recommended.get('accuracy', 0.0)):.4f}")
    print(f"context_ablation_best_macro_f1={float(recommended.get('macro_f1', 0.0)):.4f}")
    print(f"context_ablation_out_json={args.out_json}")
    print(f"context_ablation_out_csv={args.out_csv}")
    print(f"context_ablation_report={args.report_out}")


if __name__ == "__main__":
    main()
