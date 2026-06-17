from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.event_normalization.parser import DeterministicParser

from src.dataset import load_supervised_rows
from src.hybrid_router import HybridEventRouter
from src.metrics import compute_qlora_metrics, write_baseline_comparison, write_metrics, write_predictions
from src.model_loader import adapter_exists, load_adapter_for_inference, load_tokenizer, set_deterministic_seed
from src.prompts import build_prompt
from src.schema import event_from_payload, validate_model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a QLoRA event-normalization adapter")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--test-file", default=None, type=Path)
    parser.add_argument("--outputs-dir", default=None, type=Path)
    parser.add_argument("--backend", default="auto", choices=("auto", "transformers", "parser", "hybrid"))
    parser.add_argument("--baseline-csv", default=Path("reports/phase2_event_benchmark_results.csv"), type=Path)
    parser.add_argument("--max-examples", default=0, type=int)
    parser.add_argument("--confidence-threshold", default=None, type=float)
    parser.add_argument("--max-new-tokens", default=None, type=int)
    parser.add_argument("--parser-min-confidence", default=None, type=float)
    parser.add_argument("--model-min-confidence", default=None, type=float)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def resolve_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = read_config(args.config)
    if args.model_path is not None:
        cfg["output"]["model_dir"] = str(args.model_path)
    if args.test_file is not None:
        cfg["data"]["test_file"] = str(args.test_file)
    if args.outputs_dir is not None:
        cfg["output"]["outputs_dir"] = str(args.outputs_dir)
    if args.max_examples:
        cfg["data"]["max_test_examples"] = int(args.max_examples)
    if args.confidence_threshold is not None:
        cfg["inference"]["confidence_threshold"] = float(args.confidence_threshold)
    if args.max_new_tokens is not None:
        cfg["inference"]["max_new_tokens"] = int(args.max_new_tokens)
    if args.parser_min_confidence is not None:
        cfg.setdefault("hybrid", {})["parser_min_confidence"] = float(args.parser_min_confidence)
    if args.model_min_confidence is not None:
        cfg.setdefault("hybrid", {})["model_min_confidence"] = float(args.model_min_confidence)
    cfg["evaluation_backend"] = args.backend
    cfg["baseline_csv"] = str(args.baseline_csv)
    return cfg


def generate_with_model(model, tokenizer, raw_text: str, cfg: dict[str, Any]) -> str:
    import torch

    prompt = build_prompt(raw_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
            temperature=float(cfg["inference"].get("temperature", 0.0)),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return generated.strip()


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    set_deterministic_seed(int(cfg["seed"]))
    test_file = Path(cfg["data"]["test_file"])
    rows = load_supervised_rows(test_file, max_examples=int(cfg["data"].get("max_test_examples", 0)))
    if not rows:
        raise ValueError(f"No test rows found in {test_file}")

    backend = str(cfg["evaluation_backend"])
    model_path = Path(cfg["output"]["model_dir"])
    has_adapter = adapter_exists(model_path)
    use_hybrid = backend == "hybrid"
    use_model = backend == "transformers" or use_hybrid or (backend == "auto" and has_adapter)
    parser = DeterministicParser(ocr_profile="full")
    tokenizer = None
    model = None
    model_loaded_at_runtime = False
    if use_hybrid:
        method_name = "hybrid_parser_qlora" if has_adapter else "hybrid_parser_only"
    elif use_model and has_adapter:
        tokenizer = load_tokenizer(cfg["base_model"])
        model = load_adapter_for_inference(
            base_model=cfg["base_model"],
            model_path=model_path,
            device_map=str(cfg["inference"].get("device_map", "auto")),
            dtype_name=str(cfg["training"].get("fp16_or_bf16", "auto")),
        )
        method_name = "qwen25_qlora"
    else:
        method_name = "qwen25_qlora_contract_check"

    def model_extract(raw_text: str) -> Any:
        nonlocal model, tokenizer, model_loaded_at_runtime
        if use_hybrid and has_adapter and (model is None or tokenizer is None):
            tokenizer = load_tokenizer(cfg["base_model"])
            model = load_adapter_for_inference(
                base_model=cfg["base_model"],
                model_path=model_path,
                device_map=str(cfg["inference"].get("device_map", "auto")),
                dtype_name=str(cfg["training"].get("fp16_or_bf16", "auto")),
            )
            model_loaded_at_runtime = True
        if model is None or tokenizer is None:
            event = parser.parse(raw_text)
            raw = json.dumps(
                {
                    "event_type": event.event_type,
                    "player": event.player,
                    "action": event.action,
                    "amount": event.amount,
                    "cards": event.cards or None,
                    "confidence": event.confidence,
                },
                sort_keys=True,
            )
            return validate_model_output(raw, confidence_threshold=float(cfg["inference"]["confidence_threshold"]))
        raw_output = generate_with_model(model, tokenizer, raw_text, cfg)
        return validate_model_output(
            raw_output,
            confidence_threshold=float(cfg["inference"]["confidence_threshold"]),
        )

    hybrid_cfg = cfg.get("hybrid", {})
    router = HybridEventRouter(
        parser=parser,
        model_extractor=model_extract if use_hybrid else None,
        parser_min_confidence=float(hybrid_cfg.get("parser_min_confidence", 0.80)),
        model_min_confidence=float(hybrid_cfg.get("model_min_confidence", 0.50)),
        allow_model_override=bool(hybrid_cfg.get("allow_model_override", False)),
    )
    route_counts: dict[str, int] = {}

    records: list[dict[str, Any]] = []
    tracemalloc.start()
    for row in rows:
        expected = event_from_payload(row["label"])
        tracemalloc.reset_peak()
        started = time.perf_counter()
        route = "model" if use_model else "parser"
        parser_event = None
        model_attempted = False
        if use_hybrid:
            decision = router.extract(row["raw_text"])
            result = decision.result
            raw_output = result.raw_output
            route = decision.route
            parser_event = decision.parser_event
            model_attempted = decision.model_attempted
        else:
            if use_model and model is not None and tokenizer is not None:
                result = model_extract(row["raw_text"])
                raw_output = result.raw_output
            else:
                event = parser.parse(row["raw_text"])
                raw_output = json.dumps(
                    {
                        "event_type": event.event_type,
                        "player": event.player,
                        "action": event.action,
                        "amount": event.amount,
                        "cards": event.cards or None,
                        "confidence": event.confidence,
                    },
                    sort_keys=True,
                )
                result = validate_model_output(
                    raw_output,
                    confidence_threshold=float(cfg["inference"]["confidence_threshold"]),
                )
        route_counts[route] = route_counts.get(route, 0) + 1
        latency_ms = (time.perf_counter() - started) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        records.append(
            {
                "id": row.get("id"),
                "method": method_name,
                "model": cfg["base_model"],
                "raw_text": row["raw_text"],
                "expected": expected,
                "predicted": result.event,
                "schema_valid": result.schema_valid,
                "errors": result.errors,
                "raw_output": result.raw_output or raw_output,
                "latency_ms": latency_ms,
                "peak_memory_mb": peak_bytes / (1024 * 1024),
                "route": route,
                "model_attempted": model_attempted,
                "parser_prediction": None
                if parser_event is None
                else {
                    "event_type": parser_event.event_type,
                    "player": parser_event.player,
                    "action": parser_event.action,
                    "amount": parser_event.amount,
                    "cards": parser_event.cards or None,
                    "confidence": parser_event.confidence,
                },
            }
        )
    tracemalloc.stop()

    metrics = compute_qlora_metrics(records)
    metrics.update(
        {
            "method": method_name,
            "model": cfg["base_model"],
            "test_file": str(test_file),
            "examples": len(records),
            "adapter_path": str(model_path),
            "adapter_available": bool(has_adapter),
            "adapter_loaded": bool((use_model and has_adapter and not use_hybrid) or model_loaded_at_runtime),
            "production_ready": method_name == "hybrid_parser_qlora",
            "route_counts": route_counts,
        }
    )
    outputs_dir = Path(cfg["output"]["outputs_dir"])
    write_predictions(outputs_dir / "qlora_predictions.jsonl", records)
    write_metrics(outputs_dir / "qlora_metrics.json", metrics)
    write_baseline_comparison(
        baseline_csv=Path(cfg["baseline_csv"]),
        metrics=metrics,
        out_csv=outputs_dir / "baseline_vs_qlora.csv",
        method_name=method_name,
    )
    return metrics


def main() -> None:
    args = parse_args()
    cfg = resolve_cfg(args)
    metrics = evaluate(cfg)
    print(f"qlora_eval_examples={metrics['examples']}")
    print(f"qlora_eval_accuracy={metrics['accuracy']:.6f}")
    print(f"qlora_eval_macro_f1={metrics['macro_f1']:.6f}")
    print(f"qlora_eval_schema_validity={metrics['schema_validity_rate']:.6f}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

