from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation import export_simulation_events, load_prediction_records
from src.training_contract import validate_training_contract, write_contract_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the consolidated LLM training delivery report")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--train-file", default=None, type=Path)
    parser.add_argument("--val-file", default=None, type=Path)
    parser.add_argument("--test-file", default=None, type=Path)
    parser.add_argument("--metrics-file", default=Path("outputs/qlora_metrics.json"), type=Path)
    parser.add_argument("--predictions-file", default=Path("outputs/qlora_predictions.jsonl"), type=Path)
    parser.add_argument("--baseline-csv", default=Path("outputs/baseline_vs_qlora.csv"), type=Path)
    parser.add_argument("--adapter-dir", default=Path("outputs/qwen25_qlora"), type=Path)
    parser.add_argument("--report-out", default=Path("reports/llm_training_delivery_report.json"), type=Path)
    parser.add_argument("--markdown-out", default=Path("reports/llm_training_delivery_report.md"), type=Path)
    parser.add_argument("--simulation-out", default=Path("outputs/simulation_events.jsonl"), type=Path)
    parser.add_argument("--simulation-summary-out", default=Path("reports/simulation_readiness.json"), type=Path)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def resolve_data_paths(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Path]:
    data_cfg = cfg.get("data", {})
    return {
        "train": args.train_file or Path(data_cfg.get("train_file", "data/train.jsonl")),
        "val": args.val_file or Path(data_cfg.get("val_file", "data/val.jsonl")),
        "test": args.test_file or Path(data_cfg.get("test_file", "data/test.jsonl")),
    }


def validate_metric_gate(metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    validation_cfg = cfg.get("validation", {})
    thresholds = {
        "min_accuracy": float(validation_cfg.get("min_accuracy", 0.0)),
        "min_macro_f1": float(validation_cfg.get("min_macro_f1", 0.0)),
        "min_schema_validity_rate": float(validation_cfg.get("min_schema_validity_rate", 0.95)),
        "max_unmatched_rate": float(validation_cfg.get("max_unmatched_rate", 1.0)),
    }
    checks = {
        "accuracy": float(metrics.get("accuracy", 0.0)) >= thresholds["min_accuracy"],
        "macro_f1": float(metrics.get("macro_f1", 0.0)) >= thresholds["min_macro_f1"],
        "schema_validity_rate": float(metrics.get("schema_validity_rate", 0.0))
        >= thresholds["min_schema_validity_rate"],
        "unmatched_rate": float(metrics.get("unmatched_rate", 1.0)) <= thresholds["max_unmatched_rate"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "thresholds": thresholds,
        "checks": checks,
        "metrics": metrics,
    }


def adapter_artifact_report(adapter_dir: Path) -> dict[str, Any]:
    required = {
        "adapter_config": adapter_dir / "adapter_config.json",
        "adapter_weights": adapter_dir / "adapter_model.safetensors",
        "tokenizer": adapter_dir / "tokenizer.json",
        "training_plan": adapter_dir / "training_plan.json",
        "train_metrics": adapter_dir / "train_metrics.json",
    }
    files = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }
        for name, path in required.items()
    }
    return {
        "status": "PASS" if all(item["exists"] for item in files.values()) else "FAIL",
        "adapter_dir": str(adapter_dir),
        "files": files,
        "training_plan": read_json(required["training_plan"]),
        "train_metrics": read_json(required["train_metrics"]),
    }


def read_baseline_comparison(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "WARN", "path": str(path), "rows": [], "message": "baseline comparison file missing"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_keys = {
        "accuracy",
        "macro_f1",
        "schema_validity_rate",
        "event_type_exact_match",
        "action_exact_match",
        "amount_exact_match",
        "average_latency_ms",
        "unmatched_rate",
    }
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed = dict(row)
        for key in numeric_keys:
            if key in parsed and parsed[key] not in ("", None):
                parsed[key] = float(parsed[key])
        parsed_rows.append(parsed)
    best = max(parsed_rows, key=lambda item: (float(item.get("macro_f1", 0.0)), float(item.get("accuracy", 0.0)))) if parsed_rows else {}
    qlora_rows = [row for row in parsed_rows if "qlora" in str(row.get("method", "")).lower()]
    return {
        "status": "PASS" if parsed_rows else "WARN",
        "path": str(path),
        "rows": parsed_rows,
        "best_method": best,
        "qlora_methods": qlora_rows,
    }


def simulation_readiness_report(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    if not args.predictions_file.exists():
        return {
            "status": "WARN",
            "message": f"predictions file not found: {args.predictions_file}",
            "total_events": 0,
            "ready_events": 0,
            "readiness_rate": 0.0,
        }
    records = load_prediction_records(args.predictions_file)
    summary = export_simulation_events(
        records=records,
        out_jsonl=args.simulation_out,
        summary_out=args.simulation_summary_out,
        min_confidence=float(cfg.get("simulation", {}).get("min_confidence", 0.0)),
    )
    min_rate = float(cfg.get("simulation", {}).get("min_readiness_rate", 0.0))
    summary["status"] = "PASS" if float(summary.get("readiness_rate", 0.0)) >= min_rate else "FAIL"
    summary["min_readiness_rate"] = min_rate
    summary["summary_file"] = str(args.simulation_summary_out)
    return summary


def overall_status(sections: list[dict[str, Any]]) -> str:
    statuses = [str(section.get("status", "WARN")) for section in sections]
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def build_markdown(report: dict[str, Any]) -> str:
    metrics = report["metric_gate"].get("metrics", {})
    adapter = report["adapter_artifacts"]
    simulation = report["simulation_readiness"]
    baseline = report["baseline_comparison"]
    best = baseline.get("best_method") or {}
    lines = [
        "# LLM Training Delivery Report",
        "",
        "## Executive Status",
        "",
        f"- Overall status: `{report['status']}`",
        f"- Selected architecture: `{report['architecture']['selected_architecture']}`",
        f"- Base model: `{report['training']['base_model']}`",
        f"- Fine-tuning method: `{report['training']['method']}`",
        f"- Adapter artifacts: `{adapter['status']}`",
        f"- Dataset contract: `{report['dataset_contract']['status']}`",
        f"- Metric gate: `{report['metric_gate']['status']}`",
        f"- Simulation readiness: `{simulation['status']}`",
        "",
        "## Validation Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "accuracy",
        "macro_f1",
        "schema_validity_rate",
        "event_type_exact_match",
        "action_exact_match",
        "amount_exact_match",
        "unmatched_rate",
        "average_latency_ms",
    ):
        if key in metrics:
            lines.append(f"| {key} | {float(metrics[key]):.4f} |")
    lines.extend(
        [
            "",
            "## Baseline Comparison",
            "",
            f"- Best method by macro F1: `{best.get('method', 'n/a')}`",
            f"- Best macro F1: `{float(best.get('macro_f1', 0.0)):.4f}`",
            f"- Best accuracy: `{float(best.get('accuracy', 0.0)):.4f}`",
            "",
            "## Simulation Preparation",
            "",
            f"- Simulation event file: `{simulation.get('output_file', '')}`",
            f"- Ready events: `{simulation.get('ready_events', 0)}`",
            f"- Rejected events: `{simulation.get('rejected_events', 0)}`",
            f"- Readiness rate: `{float(simulation.get('readiness_rate', 0.0)):.4f}`",
            "",
            "## Decision",
            "",
            "The current LLM component is accepted as an event-normalization and simulation-preparation layer. "
            "It is not promoted as the final autonomous poker policy. The next policy milestone should use the normalized event stream to train and evaluate the strategic decision model.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    paths = resolve_data_paths(args, cfg)
    validation_cfg = cfg.get("validation", {})
    dataset_contract = validate_training_contract(
        train_file=paths["train"],
        val_file=paths["val"],
        test_file=paths["test"],
        min_rows={
            "train": int(validation_cfg.get("min_train_rows", 1)),
            "val": int(validation_cfg.get("min_val_rows", 1)),
            "test": int(validation_cfg.get("min_test_rows", 1)),
        },
        min_schema_validity=float(validation_cfg.get("min_dataset_schema_validity", 1.0)),
        max_dominant_ratio=float(validation_cfg.get("max_dominant_ratio", 0.85)),
    )
    metrics = read_json(args.metrics_file)
    adapter = adapter_artifact_report(args.adapter_dir)
    metric_gate = validate_metric_gate(metrics, cfg)
    baseline = read_baseline_comparison(args.baseline_csv)
    simulation = simulation_readiness_report(args, cfg)

    architecture = {
        "selected_architecture": cfg.get("agent", {}).get("selected_architecture", "hybrid_parser_qlora"),
        "role": "event_normalization_and_simulation_preparation",
        "not_scope": "autonomous_poker_policy",
    }
    training = {
        "base_model": cfg.get("base_model"),
        "method": cfg.get("method"),
        "quantization": cfg.get("quantization"),
        "config": str(args.config),
        "adapter_dir": str(args.adapter_dir),
    }
    report = {
        "status": overall_status([dataset_contract, adapter, metric_gate, baseline, simulation]),
        "architecture": architecture,
        "training": training,
        "dataset_paths": {key: str(value) for key, value in paths.items()},
        "dataset_contract": dataset_contract,
        "adapter_artifacts": adapter,
        "metric_gate": metric_gate,
        "baseline_comparison": baseline,
        "simulation_readiness": simulation,
        "next_milestone": "Feed simulation-ready normalized events into the strategic poker policy simulation.",
    }
    write_contract_report(args.report_out, report)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(build_markdown(report), encoding="utf-8")

    print(f"llm_training_delivery_status={report['status']}")
    print(f"llm_training_adapter_status={adapter['status']}")
    print(f"llm_training_metric_gate={metric_gate['status']}")
    print(f"llm_training_simulation_status={simulation['status']}")
    print(f"llm_training_simulation_readiness_rate={float(simulation.get('readiness_rate', 0.0)):.6f}")
    print(f"llm_training_report={args.report_out}")
    print(f"llm_training_markdown={args.markdown_out}")


if __name__ == "__main__":
    main()
