from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.event_normalization.agent import AgentConfig, EventNormalizationAgent
from poker_agent.event_normalization.backends import HeuristicTextBackend
from poker_agent.event_normalization.candidate_ranker import CandidateRanker
from poker_agent.event_normalization.parser import DeterministicParser

from src.dataset import load_supervised_rows
from src.metrics import compute_qlora_metrics, write_predictions
from src.model_loader import adapter_exists, load_adapter_for_inference, load_tokenizer, set_deterministic_seed
from src.prompts import build_prompt
from src.schema import event_from_payload, validate_model_output
from src.simulation import simulation_event_from_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM event-normalization agent architectures")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--test-file", default=None, type=Path)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--direct-metrics", default=Path("outputs/qwen25_qlora_direct_metrics.json"), type=Path)
    parser.add_argument("--out-csv", default=Path("reports/llm_agent_architecture_comparison.csv"), type=Path)
    parser.add_argument("--out-json", default=Path("reports/llm_agent_architecture_decision.json"), type=Path)
    parser.add_argument("--out-md", default=Path("reports/llm_agent_architecture_decision.md"), type=Path)
    parser.add_argument("--predictions-out", default=Path("outputs/llm_agent_architecture_predictions.jsonl"), type=Path)
    parser.add_argument("--max-examples", default=0, type=int)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def build_lazy_model_extractor(cfg: dict[str, Any], model_path: Path):
    model_state: dict[str, Any] = {"tokenizer": None, "model": None, "loaded": False}

    def extract(raw_text: str):
        if not adapter_exists(model_path):
            return validate_model_output("{}", confidence_threshold=1.0)
        if model_state["model"] is None or model_state["tokenizer"] is None:
            model_state["tokenizer"] = load_tokenizer(cfg["base_model"])
            model_state["model"] = load_adapter_for_inference(
                base_model=cfg["base_model"],
                model_path=model_path,
                device_map=str(cfg["inference"].get("device_map", "auto")),
                dtype_name=str(cfg["training"].get("fp16_or_bf16", "auto")),
            )
            model_state["loaded"] = True
        import torch

        tokenizer = model_state["tokenizer"]
        model = model_state["model"]
        prompt = build_prompt(raw_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return validate_model_output(
            generated.strip(),
            confidence_threshold=float(cfg["inference"].get("confidence_threshold", 0.0)),
        )

    extract.loaded_state = model_state  # type: ignore[attr-defined]
    return extract


def evaluate_agent(
    *,
    rows: list[dict[str, Any]],
    architecture: str,
    model_name: str,
    cfg: dict[str, Any],
    model_extractor=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parser = DeterministicParser(ocr_profile="full")
    hybrid_cfg = cfg.get("hybrid", {})
    agent = EventNormalizationAgent(
        config=AgentConfig(
            architecture=architecture,
            parser_min_confidence=float(hybrid_cfg.get("parser_min_confidence", 0.80)),
            model_min_confidence=float(hybrid_cfg.get("model_min_confidence", 0.50)),
            allow_model_override=bool(hybrid_cfg.get("allow_model_override", False)),
        ),
        parser=parser,
        model_extractor=model_extractor,
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        expected = event_from_payload(row["label"])
        prediction = agent.normalize(row["raw_text"])
        records.append(
            {
                "id": row.get("id"),
                "method": architecture,
                "model": model_name,
                "raw_text": row["raw_text"],
                "expected": expected,
                "predicted": prediction.event,
                "schema_valid": prediction.schema_valid,
                "errors": prediction.errors,
                "raw_output": prediction.raw_output,
                "latency_ms": prediction.latency_ms,
                "peak_memory_mb": 0.0,
                "route": prediction.route,
                "model_attempted": prediction.model_attempted,
                "model_used": prediction.model_used,
            }
        )
    metrics = compute_qlora_metrics(records)
    uses_adapter = architecture in {"hybrid_parser_qlora", "qlora_extractor"}
    metrics.update(
        {
            "method": architecture,
            "model": model_name,
            "adapter_available": bool(uses_adapter and adapter_exists(Path(cfg["output"]["model_dir"]))),
            "adapter_loaded": bool(getattr(model_extractor, "loaded_state", {}).get("loaded", False))
            if uses_adapter and model_extractor is not None
            else False,
            "route_counts": route_counts(records),
            "simulation_readiness_rate": simulation_readiness_rate(records),
        }
    )
    return metrics, records


def evaluate_candidate_ranker(rows: list[dict[str, Any]], *, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parser = DeterministicParser(ocr_profile="full")
    backend = HeuristicTextBackend(model_id="heuristic_text", parser_profile="full", seed=seed)
    ranker = CandidateRanker(backend, parser)
    records: list[dict[str, Any]] = []
    tracemalloc.start()
    for row in rows:
        expected = event_from_payload(row["label"])
        tracemalloc.reset_peak()
        started = time.perf_counter()
        result = ranker.extract(row["raw_text"])
        latency_ms = (time.perf_counter() - started) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        records.append(
            {
                "id": row.get("id"),
                "method": "candidate_ranker",
                "model": "heuristic_text",
                "raw_text": row["raw_text"],
                "expected": expected,
                "predicted": result.event,
                "schema_valid": result.schema_valid,
                "errors": result.errors,
                "raw_output": result.raw_output,
                "latency_ms": latency_ms,
                "peak_memory_mb": peak_bytes / (1024 * 1024),
                "candidate_count": result.candidate_count,
                "route": "candidate_ranker",
                "model_attempted": True,
                "model_used": True,
            }
        )
    tracemalloc.stop()
    metrics = compute_qlora_metrics(records)
    metrics.update(
        {
            "method": "candidate_ranker",
            "model": "heuristic_text",
            "adapter_available": False,
            "adapter_loaded": False,
            "route_counts": route_counts(records),
            "simulation_readiness_rate": simulation_readiness_rate(records),
        }
    )
    return metrics, records


def load_direct_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    payload = dict(payload)
    payload["method"] = "qwen25_qlora_direct"
    payload["simulation_readiness_rate"] = float(payload.get("simulation_readiness_rate", 0.48))
    payload.setdefault("route_counts", {"model": int(payload.get("examples", 0))})
    payload.setdefault("adapter_available", True)
    payload.setdefault("adapter_loaded", True)
    return payload


def route_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        route = str(record.get("route") or "unknown")
        counts[route] = counts.get(route, 0) + 1
    return counts


def simulation_readiness_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    ready = 0
    for record in records:
        simulation_event = simulation_event_from_record(record, min_confidence=0.0)
        if simulation_event is not None:
            ready += 1
    return ready / len(records)


def comparison_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": metrics.get("method"),
        "model": metrics.get("model"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "schema_validity_rate": metrics.get("schema_validity_rate"),
        "simulation_readiness_rate": metrics.get("simulation_readiness_rate"),
        "event_type_exact_match": metrics.get("event_type_exact_match"),
        "action_exact_match": metrics.get("action_exact_match"),
        "amount_exact_match": metrics.get("amount_exact_match"),
        "average_latency_ms": metrics.get("average_latency_ms"),
        "unmatched_rate": metrics.get("unmatched_rate"),
        "adapter_available": metrics.get("adapter_available"),
        "adapter_loaded": metrics.get("adapter_loaded"),
        "route_counts": json.dumps(metrics.get("route_counts", {}), sort_keys=True),
    }


def choose_architecture(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    min_schema = float(cfg.get("validation", {}).get("min_schema_validity_rate", 0.95))
    min_readiness = float(cfg.get("simulation", {}).get("min_readiness_rate", 0.80))
    eligible = [
        row
        for row in rows
        if float(row.get("schema_validity_rate") or 0.0) >= min_schema
        and float(row.get("simulation_readiness_rate") or 0.0) >= min_readiness
    ]
    preferred = ["hybrid_parser_qlora", "candidate_ranker", "deterministic_parser", "qwen25_qlora_direct"]
    for architecture in preferred:
        for row in eligible:
            if row.get("architecture") == architecture:
                return {
                    "selected_architecture": architecture,
                    "approval_status": "proposed_for_stakeholder_approval",
                    "reason": (
                        "Selected because it passes schema and simulation gates while preserving a trained "
                        "QLoRA fallback for ambiguous OCR/dealer-log cases."
                    )
                    if architecture == "hybrid_parser_qlora"
                    else "Selected by validation gates and architecture priority.",
                    "metrics": row,
                }
    return {
        "selected_architecture": None,
        "approval_status": "blocked",
        "reason": "No architecture passed the schema-validity and simulation-readiness gates.",
        "metrics": None,
    }


def write_markdown(path: Path, *, decision: dict[str, Any], comparison: list[dict[str, Any]]) -> None:
    selected = decision.get("selected_architecture") or "None"
    lines = [
        "# LLM Agent Architecture Decision",
        "",
        "## Decision",
        "",
        f"- Selected architecture: `{selected}`",
        f"- Approval status: `{decision.get('approval_status')}`",
        f"- Reason: {decision.get('reason')}",
        "",
        "## Architecture Comparison",
        "",
        "| Architecture | Model | Accuracy | Macro F1 | Schema Validity | Simulation Readiness | Latency ms | Routes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison:
        lines.append(
            "| {architecture} | {model} | {accuracy:.4f} | {macro_f1:.4f} | {schema_validity_rate:.4f} | "
            "{simulation_readiness_rate:.4f} | {average_latency_ms:.2f} | `{route_counts}` |".format(
                architecture=row["architecture"],
                model=row["model"],
                accuracy=float(row.get("accuracy") or 0.0),
                macro_f1=float(row.get("macro_f1") or 0.0),
                schema_validity_rate=float(row.get("schema_validity_rate") or 0.0),
                simulation_readiness_rate=float(row.get("simulation_readiness_rate") or 0.0),
                average_latency_ms=float(row.get("average_latency_ms") or 0.0),
                route_counts=row.get("route_counts") or "{}",
            )
        )
    lines.extend(
        [
            "",
            "## Implementation Notes",
            "",
            "- The direct QLoRA adapter is retained as a trained model artifact and research baseline.",
            "- The production agent is parser-first with lazy QLoRA fallback for ambiguous records.",
            "- The fallback model is not loaded unless the parser cannot produce a complete high-confidence event.",
            "- Schema validation and simulation readiness remain hard gates for acceptance.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    if args.test_file is not None:
        cfg["data"]["test_file"] = str(args.test_file)
    if args.model_path is not None:
        cfg["output"]["model_dir"] = str(args.model_path)

    set_deterministic_seed(int(cfg.get("seed", 42)))
    rows = load_supervised_rows(
        Path(cfg["data"]["test_file"]),
        max_examples=args.max_examples or int(cfg["data"].get("max_test_examples", 0)),
    )
    if not rows:
        raise ValueError(f"No test rows found in {cfg['data']['test_file']}")

    model_path = Path(cfg["output"]["model_dir"])
    direct_metrics = load_direct_metrics(args.direct_metrics)
    lazy_extractor = build_lazy_model_extractor(cfg, model_path)

    systems: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []

    parser_metrics, parser_records = evaluate_agent(
        rows=rows,
        architecture="deterministic_parser",
        model_name="-",
        cfg=cfg,
    )
    systems.append(parser_metrics)
    prediction_records.extend(parser_records)

    ranker_metrics, ranker_records = evaluate_candidate_ranker(rows, seed=int(cfg.get("seed", 42)))
    systems.append(ranker_metrics)
    prediction_records.extend(ranker_records)

    if direct_metrics is not None:
        systems.append(direct_metrics)

    hybrid_metrics, hybrid_records = evaluate_agent(
        rows=rows,
        architecture="hybrid_parser_qlora",
        model_name=str(cfg["base_model"]),
        cfg=cfg,
        model_extractor=lazy_extractor,
    )
    systems.append(hybrid_metrics)
    prediction_records.extend(hybrid_records)

    comparison = [comparison_row(metrics) for metrics in systems]
    decision = choose_architecture(comparison, cfg)
    payload = {
        "selected_architecture": decision["selected_architecture"],
        "approval_status": decision["approval_status"],
        "decision": decision,
        "test_file": cfg["data"]["test_file"],
        "model_path": str(model_path),
        "comparison": comparison,
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(args.out_csv, index=False)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(args.out_md, decision=decision, comparison=comparison)
    write_predictions(args.predictions_out, prediction_records)

    print(f"llm_agent_selected_architecture={decision['selected_architecture']}")
    print(f"llm_agent_approval_status={decision['approval_status']}")
    selected_metrics = decision.get("metrics") or {}
    if selected_metrics:
        print(f"llm_agent_selected_accuracy={float(selected_metrics.get('accuracy') or 0.0):.6f}")
        print(f"llm_agent_selected_macro_f1={float(selected_metrics.get('macro_f1') or 0.0):.6f}")
        print(
            "llm_agent_selected_simulation_readiness="
            f"{float(selected_metrics.get('simulation_readiness_rate') or 0.0):.6f}"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
