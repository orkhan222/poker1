from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poker_agent.event_normalization.parser import DeterministicParser

from src.model_loader import adapter_exists, load_adapter_for_inference, load_tokenizer, set_deterministic_seed
from src.prompts import build_prompt
from src.schema import event_to_jsonable, validate_model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize one noisy poker OCR/dealer-log text into JSON")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--backend", default="auto", choices=("auto", "transformers", "parser"))
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def main() -> None:
    args = parse_args()
    cfg = read_config(args.config)
    set_deterministic_seed(int(cfg["seed"]))
    model_path = args.model_path or Path(cfg["output"]["model_dir"])
    use_model = args.backend == "transformers" or (args.backend == "auto" and adapter_exists(model_path))
    if use_model:
        import torch

        tokenizer = load_tokenizer(cfg["base_model"])
        model = load_adapter_for_inference(
            base_model=cfg["base_model"],
            model_path=model_path,
            device_map=str(cfg["inference"].get("device_map", "auto")),
            dtype_name=str(cfg["training"].get("fp16_or_bf16", "auto")),
        )
        prompt = build_prompt(args.text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
                temperature=float(cfg["inference"].get("temperature", 0.0)),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_output = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        result = validate_model_output(
            raw_output,
            confidence_threshold=float(cfg["inference"]["confidence_threshold"]),
        )
    else:
        event = DeterministicParser(ocr_profile="full").parse(args.text)
        raw_output = json.dumps(event_to_jsonable(event), sort_keys=True)
        result = validate_model_output(
            raw_output,
            confidence_threshold=float(cfg["inference"]["confidence_threshold"]),
        )
    print(json.dumps(event_to_jsonable(result.event), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

