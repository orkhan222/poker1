from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import load_hf_json_dataset, load_supervised_rows
from src.formatting import format_training_row
from src.model_loader import load_qlora_training_model, load_tokenizer, set_deterministic_seed
from src.training_contract import validate_training_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 for OCR event normalization with QLoRA")
    parser.add_argument("--config", default=Path("config.yaml"), type=Path)
    parser.add_argument("--train-file", default=None, type=Path)
    parser.add_argument("--val-file", default=None, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-examples", default=0, type=int)
    parser.add_argument("--max-val-examples", default=0, type=int)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def resolved_training_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = read_config(args.config)
    data_cfg = cfg.setdefault("data", {})
    output_cfg = cfg.setdefault("output", {})
    if args.train_file is not None:
        data_cfg["train_file"] = str(args.train_file)
    if args.val_file is not None:
        data_cfg["val_file"] = str(args.val_file)
    if args.output_dir is not None:
        output_cfg["model_dir"] = str(args.output_dir)
    if args.max_train_examples:
        data_cfg["max_train_examples"] = int(args.max_train_examples)
    if args.max_val_examples:
        data_cfg["max_val_examples"] = int(args.max_val_examples)
    if args.dry_run:
        cfg["dry_run"] = True
    return cfg


def validate_files(cfg: dict[str, Any]) -> dict[str, Any]:
    train_file = Path(cfg["data"]["train_file"])
    val_file = Path(cfg["data"]["val_file"])
    test_file = Path(cfg["data"].get("test_file", cfg["data"]["val_file"]))
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    train_rows = load_supervised_rows(train_file, max_examples=int(cfg["data"].get("max_train_examples", 0)))
    val_rows = load_supervised_rows(val_file, max_examples=int(cfg["data"].get("max_val_examples", 0)))
    if not train_rows:
        raise ValueError("Training split is empty")
    if not val_rows:
        raise ValueError("Validation split is empty")
    validation_cfg = cfg.get("validation", {})
    contract = validate_training_contract(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        min_rows={
            "train": int(validation_cfg.get("min_train_rows", 1)),
            "val": int(validation_cfg.get("min_val_rows", 1)),
            "test": int(validation_cfg.get("min_test_rows", 1)),
        },
        min_schema_validity=float(validation_cfg.get("min_dataset_schema_validity", 1.0)),
        max_dominant_ratio=float(validation_cfg.get("max_dominant_ratio", 0.85)),
    )
    if contract["status"] == "FAIL":
        raise ValueError(f"Training dataset contract failed: {contract.get('messages', [])}")
    return {"train_rows": len(train_rows), "val_rows": len(val_rows), "dataset_contract": contract}


def write_training_plan(cfg: dict[str, Any], counts: dict[str, Any]) -> Path:
    output_dir = Path(cfg["output"]["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "base_model": cfg["base_model"],
        "method": cfg["method"],
        "quantization": cfg["quantization"],
        "dataset": {
            "train_file": cfg["data"]["train_file"],
            "val_file": cfg["data"]["val_file"],
            **counts,
        },
        "validation": cfg.get("validation", {}),
        "simulation": cfg.get("simulation", {}),
        "training": cfg["training"],
        "status": "validated",
    }
    plan_path = output_dir / "training_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan_path


def prepare_dataset(cfg: dict[str, Any], tokenizer):
    data = load_hf_json_dataset(Path(cfg["data"]["train_file"]), Path(cfg["data"]["val_file"]))
    eos_token = tokenizer.eos_token or ""
    max_train = int(cfg["data"].get("max_train_examples", 0))
    max_val = int(cfg["data"].get("max_val_examples", 0))
    if max_train > 0:
        data["train"] = data["train"].select(range(min(max_train, len(data["train"]))))
    if max_val > 0:
        data["validation"] = data["validation"].select(range(min(max_val, len(data["validation"]))))
    return data.map(lambda row: format_training_row(row, eos_token=eos_token), remove_columns=data["train"].column_names)


def trainer_precision_flags(precision_mode: str) -> dict[str, bool]:
    import torch

    requested = str(precision_mode or "auto").lower()
    if requested == "bf16":
        return {"fp16": False, "bf16": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8}
    if requested == "fp16":
        return {"fp16": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7, "bf16": False}
    return {"fp16": torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7, "bf16": False}


def cast_trainable_parameters_to_fp32(model: Any) -> None:
    import torch

    low_precision = {torch.float16, torch.bfloat16}
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype in low_precision:
            parameter.data = parameter.data.to(torch.float32)


def run_training(cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cfg["training"]
    tokenizer = load_tokenizer(cfg["base_model"])
    model, lora_config = load_qlora_training_model(cfg["base_model"], train_cfg)
    dataset = prepare_dataset(cfg, tokenizer)

    from trl import SFTConfig
    from trl import SFTTrainer

    sft_params = inspect.signature(SFTConfig.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in sft_params else "evaluation_strategy"
    precision_mode = str(train_cfg.get("fp16_or_bf16", "auto")).lower()
    precision_flags = trainer_precision_flags(precision_mode)
    sft_kwargs: dict[str, Any] = {
        "output_dir": cfg["output"]["model_dir"],
        "num_train_epochs": float(train_cfg["epochs"]),
        "per_device_train_batch_size": int(train_cfg["batch_size"]),
        "per_device_eval_batch_size": int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
        "gradient_accumulation_steps": int(train_cfg["gradient_accumulation_steps"]),
        "learning_rate": float(train_cfg["learning_rate"]),
        "warmup_ratio": float(train_cfg["warmup_ratio"]),
        "logging_steps": int(train_cfg["logging_steps"]),
        "eval_steps": int(train_cfg["eval_steps"]),
        "save_steps": int(train_cfg["save_steps"]),
        strategy_key: "steps",
        "save_strategy": "steps",
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "optim": str(train_cfg["optimizer"]),
        "fp16": precision_flags["fp16"],
        "bf16": precision_flags["bf16"],
        "report_to": [],
        "seed": int(cfg["seed"]),
        "data_seed": int(cfg["seed"]),
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "torch_empty_cache_steps": int(train_cfg.get("torch_empty_cache_steps", 1)),
        "dataloader_pin_memory": False,
        "auto_find_batch_size": bool(train_cfg.get("auto_find_batch_size", True)),
        "dataset_text_field": "text",
        "max_length": int(train_cfg["max_seq_len"]),
        "packing": False,
    }
    args = SFTConfig(**{key: value for key, value in sft_kwargs.items() if key in sft_params})

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "peft_config": lora_config,
        "args": args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = int(train_cfg["max_seq_len"])
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)
    cast_trainable_parameters_to_fp32(trainer.model)
    train_result = trainer.train()
    trainer.save_model(cfg["output"]["model_dir"])
    tokenizer.save_pretrained(cfg["output"]["model_dir"])
    metrics = dict(train_result.metrics)
    metrics_path = Path(cfg["output"]["model_dir"]) / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    cfg = resolved_training_config(args)
    set_deterministic_seed(int(cfg["seed"]))
    counts = validate_files(cfg)
    if bool(cfg.get("dry_run", False)):
        plan_path = write_training_plan(cfg, counts)
        print(f"qlora_train_status=validated")
        print(f"qlora_training_plan={plan_path.as_posix()}")
        print(json.dumps({"counts": counts, "plan": str(plan_path)}, indent=2, sort_keys=True))
        return
    metrics = run_training(cfg)
    print("qlora_train_status=trained")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
