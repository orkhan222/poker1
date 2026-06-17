from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any


QWEN_CAUSAL_LM_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def set_deterministic_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def compute_dtype(dtype_name: str):
    import torch

    requested = str(dtype_name or "auto").lower()
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_qlora_training_model(model_id: str, cfg: dict[str, Any]):
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = compute_dtype(cfg.get("fp16_or_bf16", "auto"))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.get("target_modules") or QWEN_CAUSAL_LM_TARGET_MODULES),
    )
    return model, lora_config


def load_adapter_for_inference(
    *,
    base_model: str,
    model_path: Path,
    device_map: str = "auto",
    dtype_name: str = "auto",
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    dtype = compute_dtype(dtype_name)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(model_path))
    model.eval()
    return model


def adapter_exists(model_path: Path) -> bool:
    return (model_path / "adapter_config.json").exists() or (model_path / "adapter_model.safetensors").exists()

