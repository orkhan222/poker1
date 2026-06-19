from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.event_normalization.agent import AgentConfig, EventNormalizationAgent

from src.model_loader import adapter_exists, load_adapter_for_inference, load_tokenizer, set_deterministic_seed
from src.prompts import build_prompt
from src.schema import validate_model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the selected event-normalization LLM agent")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--text", default=None)
    parser.add_argument("--input-jsonl", default=None, type=Path)
    parser.add_argument("--output-jsonl", default=Path("outputs/event_normalization_agent_predictions.jsonl"), type=Path)
    parser.add_argument("--summary-out", default=Path("reports/event_normalization_agent_summary.json"), type=Path)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--require-approved", action="store_true")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def iter_inputs(*, text: str | None, input_jsonl: Path | None) -> Iterable[dict[str, Any]]:
    if text is not None:
        yield {"id": "inline", "raw_text": text}
    if input_jsonl is None:
        return
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = {"raw_text": line.strip()}
            if not isinstance(payload, dict):
                payload = {"raw_text": str(payload)}
            raw_text = str(payload.get("raw_text") or payload.get("text") or "").strip()
            if not raw_text:
                continue
            yield {"id": payload.get("id", f"line-{line_number}"), "raw_text": raw_text, "source": payload}


class LazyQLoraExtractor:
    def __init__(self, cfg: dict[str, Any], model_path: Path) -> None:
        self.cfg = cfg
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.loaded = False

    def __call__(self, raw_text: str):
        if not adapter_exists(self.model_path):
            return validate_model_output("{}", confidence_threshold=1.0)
        if self.model is None or self.tokenizer is None:
            self.tokenizer = load_tokenizer(self.cfg["base_model"])
            self.model = load_adapter_for_inference(
                base_model=self.cfg["base_model"],
                model_path=self.model_path,
                device_map=str(self.cfg["inference"].get("device_map", "auto")),
                dtype_name=str(self.cfg["training"].get("fp16_or_bf16", "auto")),
            )
            self.loaded = True

        import torch

        prompt = build_prompt(raw_text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg["inference"]["max_new_tokens"]),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return validate_model_output(
            generated.strip(),
            confidence_threshold=float(self.cfg["inference"].get("confidence_threshold", 0.0)),
        )


def build_agent(cfg: dict[str, Any], *, architecture: str, model_path: Path) -> tuple[EventNormalizationAgent, LazyQLoraExtractor]:
    hybrid_cfg = cfg.get("hybrid", {})
    extractor = LazyQLoraExtractor(cfg, model_path)
    agent = EventNormalizationAgent(
        config=AgentConfig(
            architecture=architecture,
            parser_min_confidence=float(hybrid_cfg.get("parser_min_confidence", 0.80)),
            model_min_confidence=float(hybrid_cfg.get("model_min_confidence", 0.50)),
            allow_model_override=bool(hybrid_cfg.get("allow_model_override", False)),
        ),
        model_extractor=extractor,
    )
    return agent, extractor


def record_prediction(item: dict[str, Any], prediction, *, cfg: dict[str, Any], extractor: LazyQLoraExtractor) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "raw_text": item["raw_text"],
        "prediction": prediction.to_record(),
        "selected_architecture": cfg.get("agent", {}).get("selected_architecture"),
        "approval_status": cfg.get("agent", {}).get("approval_status"),
        "base_model": cfg.get("base_model"),
        "adapter_available": adapter_exists(Path(cfg["output"]["model_dir"])),
        "adapter_loaded": extractor.loaded,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    route_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    for record in records:
        prediction = record["prediction"]
        route = str(prediction.get("route") or "unknown")
        event_type = str(prediction.get("event", {}).get("event_type") or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1
        event_counts[event_type] = event_counts.get(event_type, 0) + 1
    return {
        "records": len(records),
        "route_counts": route_counts,
        "event_type_counts": event_counts,
        "adapter_loaded": any(bool(record.get("adapter_loaded")) for record in records),
    }


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    if args.model_path is not None:
        cfg["output"]["model_dir"] = str(args.model_path)

    selected_architecture = args.architecture or str(
        cfg.get("agent", {}).get("selected_architecture", "hybrid_parser_qlora")
    )
    approval_status = str(cfg.get("agent", {}).get("approval_status", "unknown"))
    if args.require_approved and approval_status != "approved":
        raise RuntimeError(f"Agent architecture is not approved yet: {approval_status}")

    set_deterministic_seed(int(cfg.get("seed", 42)))
    model_path = Path(cfg["output"]["model_dir"])
    agent, extractor = build_agent(cfg, architecture=selected_architecture, model_path=model_path)

    inputs = list(iter_inputs(text=args.text, input_jsonl=args.input_jsonl))
    if not inputs:
        raise ValueError("Provide --text or --input-jsonl")

    records = [
        record_prediction(item, agent.normalize(item["raw_text"]), cfg=cfg, extractor=extractor)
        for item in inputs
    ]
    summary = summarize(records)
    summary.update(
        {
            "selected_architecture": selected_architecture,
            "approval_status": approval_status,
            "base_model": cfg.get("base_model"),
            "model_path": str(model_path),
            "adapter_available": adapter_exists(model_path),
        }
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.text is not None and args.input_jsonl is None:
        prediction = records[0]["prediction"]
        event = prediction["event"]
        print(f"event_agent_selected_architecture={selected_architecture}")
        print(f"event_agent_approval_status={approval_status}")
        print(f"event_agent_route={prediction['route']}")
        print(f"event_agent_event_type={event['event_type']}")
        print(f"event_agent_action={event.get('action')}")
        print(f"event_agent_schema_valid={str(prediction['schema_valid']).lower()}")
        print(f"event_agent_adapter_available={str(adapter_exists(model_path)).lower()}")
        print(f"event_agent_adapter_loaded={str(extractor.loaded).lower()}")
        print(json.dumps(records[0], indent=2, sort_keys=True))
    else:
        print(f"event_agent_selected_architecture={selected_architecture}")
        print(f"event_agent_approval_status={approval_status}")
        print(f"event_agent_records={summary['records']}")
        print(f"event_agent_adapter_loaded={str(summary['adapter_loaded']).lower()}")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
