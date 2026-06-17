from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Validate LLM training, evaluation, and simulation-readiness outputs")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--train-file", default=None, type=Path)
    parser.add_argument("--val-file", default=None, type=Path)
    parser.add_argument("--test-file", default=None, type=Path)
    parser.add_argument("--metrics-file", default=Path("outputs/qlora_metrics.json"), type=Path)
    parser.add_argument("--predictions-file", default=Path("outputs/qlora_predictions.jsonl"), type=Path)
    parser.add_argument("--report-out", default=Path("reports/llm_training_validation_report.json"), type=Path)
    parser.add_argument("--markdown-out", default=Path("reports/llm_training_validation_report.md"), type=Path)
    parser.add_argument("--simulation-out", default=Path("outputs/simulation_events.jsonl"), type=Path)
    parser.add_argument("--simulation-summary-out", default=Path("reports/simulation_readiness.json"), type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def resolved_paths(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Path]:
    data_cfg = cfg.get("data", {})
    return {
        "train": args.train_file or Path(data_cfg.get("train_file", "data/train.jsonl")),
        "val": args.val_file or Path(data_cfg.get("val_file", "data/val.jsonl")),
        "test": args.test_file or Path(data_cfg.get("test_file", "data/test.jsonl")),
    }


def metric_gate(metrics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
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


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")
    return payload


def build_markdown_report(report: dict[str, Any]) -> str:
    dataset = report["dataset_contract"]
    metric_gate_report = report["metric_gate"]
    simulation = report.get("simulation_readiness") or {}
    lines = [
        "# LLM Training Validation Report",
        "",
        "## Status",
        "",
        f"- Overall status: `{report['status']}`",
        f"- Dataset contract: `{dataset['status']}`",
        f"- Metric gate: `{metric_gate_report['status']}`",
        f"- Simulation readiness: `{simulation.get('status', 'NOT_RUN')}`",
        "",
        "## Dataset Splits",
        "",
        "| Split | Rows | Valid Rows | Schema Validity | Dominant Event | Dominant Ratio |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for split, profile in dataset.get("splits", {}).items():
        balance = dataset.get("class_balance", {}).get(split, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(profile.get("rows", 0)),
                    str(profile.get("valid_rows", 0)),
                    f"{float(profile.get('schema_validity_rate', 0.0)):.4f}",
                    str(balance.get("dominant_event_type")),
                    f"{float(balance.get('dominant_ratio', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Evaluation Metrics",
            "",
        ]
    )
    metrics = metric_gate_report.get("metrics") or {}
    if metrics:
        for key in ("accuracy", "macro_f1", "schema_validity_rate", "event_type_exact_match", "action_exact_match", "amount_exact_match", "unmatched_rate"):
            if key in metrics:
                lines.append(f"- {key}: `{float(metrics[key]):.4f}`")
    else:
        lines.append("- Metrics were not available at validation time.")
    if simulation:
        lines.extend(
            [
                "",
                "## Simulation Readiness",
                "",
                f"- Ready events: `{simulation.get('ready_events', 0)}`",
                f"- Rejected events: `{simulation.get('rejected_events', 0)}`",
                f"- Readiness rate: `{float(simulation.get('readiness_rate', 0.0)):.4f}`",
            ]
        )
    messages = report.get("messages") or []
    if messages:
        lines.extend(["", "## Findings", ""])
        lines.extend(f"- {message}" for message in messages)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    paths = resolved_paths(args, cfg)
    validation_cfg = cfg.get("validation", {})
    contract = validate_training_contract(
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

    metrics = load_metrics(args.metrics_file)
    if metrics is None:
        metric_report = {
            "status": "WARN",
            "metrics": {},
            "checks": {},
            "thresholds": {},
            "message": f"metrics file not found: {args.metrics_file}",
        }
    else:
        metric_report = metric_gate(metrics, cfg)

    simulation_report: dict[str, Any] | None = None
    if args.predictions_file.exists():
        records = load_prediction_records(args.predictions_file)
        simulation_report = export_simulation_events(
            records=records,
            out_jsonl=args.simulation_out,
            summary_out=args.simulation_summary_out,
            min_confidence=float(cfg.get("simulation", {}).get("min_confidence", 0.0)),
        )
        min_readiness = float(cfg.get("simulation", {}).get("min_readiness_rate", 0.0))
        simulation_report["status"] = "PASS" if simulation_report["readiness_rate"] >= min_readiness else "FAIL"
        simulation_report["min_readiness_rate"] = min_readiness
    else:
        simulation_report = {
            "status": "WARN",
            "message": f"predictions file not found: {args.predictions_file}",
            "total_events": 0,
            "ready_events": 0,
            "rejected_events": 0,
            "readiness_rate": 0.0,
        }

    statuses = [contract["status"], metric_report["status"], simulation_report["status"]]
    if "FAIL" in statuses:
        status = "FAIL"
    elif "WARN" in statuses:
        status = "WARN"
    else:
        status = "PASS"

    messages = list(contract.get("messages") or [])
    if metric_report.get("message"):
        messages.append(metric_report["message"])
    if simulation_report.get("message"):
        messages.append(simulation_report["message"])

    report = {
        "status": status,
        "config": str(args.config),
        "dataset_paths": {key: str(value) for key, value in paths.items()},
        "dataset_contract": contract,
        "metric_gate": metric_report,
        "simulation_readiness": simulation_report,
        "messages": messages,
    }
    write_contract_report(args.report_out, report)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"llm_training_validation_status={status}")
    print(f"llm_dataset_contract_status={contract['status']}")
    print(f"llm_metric_gate_status={metric_report['status']}")
    print(f"llm_simulation_readiness_status={simulation_report['status']}")
    print(f"llm_simulation_readiness_rate={float(simulation_report.get('readiness_rate', 0.0)):.6f}")
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.strict and status == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
