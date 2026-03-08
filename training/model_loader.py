"""
Unified model loading for KernelForge training pipeline.

Primary: unsloth/Qwen3-Coder-30B-A3B-Instruct (30.5B MoE, 3.3B active)
         via Unsloth FastLanguageModel + PatchFastRL for GRPO.
         bf16 on H200 141GB (~61GB model, ~80GB free for vLLM + GRPO).
         Unsloth's 2026 Faster MOE update handles MoE LoRA natively.

Dev:     Qwen2.5-Coder-0.5B-Instruct for macOS control-plane validation.
"""
from __future__ import annotations

import os
import sys

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")

# Primary model (MoE via Unsloth)
PRIMARY_MODEL = os.getenv("KERNELFORGE_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
DEV_MODEL = os.getenv("KERNELFORGE_DEV_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

# LoRA constants
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MAX_SEQ_LENGTH = 8192

# Singleton cache
_model = None
_tokenizer = None
_model_type = None  # "moe" or "portable"
_model_key = None


def load_model_and_tokenizer(
    checkpoint_path: str | None = None,
):
    """Load model and tokenizer.

    Args:
        checkpoint_path: Load from a fine-tuned checkpoint instead of base model.

    Returns:
        (model, tokenizer) tuple ready for training.
    """
    global _model, _tokenizer, _model_type, _model_key

    resolved_checkpoint = checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else None
    cache_key = resolved_checkpoint or "primary"
    if _model is not None and _model_key == cache_key:
        return _model, _tokenizer

    if resolved_checkpoint:
        _model, _tokenizer = _load_from_checkpoint(resolved_checkpoint)
        _model_key = cache_key
        return _model, _tokenizer

    if sys.platform != "linux":
        _model, _tokenizer = _load_portable_dev_model()
        _model_type = "portable"
    else:
        _model, _tokenizer = _load_primary()
        _model_type = "moe"
    _model_key = cache_key
    return _model, _tokenizer


def get_model_type() -> str | None:
    """Return 'moe' or 'portable' depending on which model loaded."""
    return _model_type


def _load_primary():
    """Load MoE model via Unsloth FastLanguageModel (supports MoE since 2026)."""
    from unsloth import FastLanguageModel, PatchFastRL

    candidates: list[str] = []
    unsloth_alias = (
        f"unsloth/{PRIMARY_MODEL.split('/', 1)[1]}"
        if not PRIMARY_MODEL.startswith("unsloth/") and "/" in PRIMARY_MODEL
        else (f"unsloth/{PRIMARY_MODEL}" if not PRIMARY_MODEL.startswith("unsloth/") else PRIMARY_MODEL.split("/", 1)[1])
    )
    for candidate in (PRIMARY_MODEL, unsloth_alias):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_error: Exception | None = None
    for candidate in candidates:
        print(f"Loading primary model: {candidate}")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=candidate,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=False,
                fast_inference=False,
            )
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                target_modules=LORA_TARGETS,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                use_gradient_checkpointing=True,
                random_state=3407,
                bias="none",
            )

            PatchFastRL("GRPO", FastLanguageModel)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(
                f"Primary model loaded from {candidate}: {trainable:,} trainable / {total:,} total params "
                f"({trainable / total * 100:.2f}%)"
            )
            return model, tokenizer
        except Exception as exc:
            last_error = exc
            print(f"Primary model load failed for {candidate}: {str(exc)[:500]}")

    print("Falling back to Transformers + PEFT primary model loader")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model_name = PRIMARY_MODEL.split("/", 1)[1] if PRIMARY_MODEL.startswith("unsloth/") else PRIMARY_MODEL
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
        ),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Primary model loaded via Transformers + PEFT from {hf_model_name}: "
        f"{trainable:,} trainable / {total:,} total params ({trainable / total * 100:.2f}%)"
    )
    return model, tokenizer


def _load_portable_dev_model():
    """Portable non-Linux fallback used for local control-plane validation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading portable dev model: {DEV_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        DEV_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Portable dev model loaded: {trainable:,} trainable / {total:,} total params")
    return model, tokenizer


def _load_from_checkpoint(checkpoint_path: str):
    """Load a fine-tuned checkpoint."""
    import torch

    print(f"Loading checkpoint: {checkpoint_path}")

    # Try Unsloth first (primary path)
    try:
        from unsloth import FastLanguageModel, PatchFastRL
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=False,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        PatchFastRL("GRPO", FastLanguageModel)
        print(f"Loaded checkpoint via Unsloth: {checkpoint_path}")
        return model, tokenizer
    except Exception:
        pass

    # Fall back to HF + PEFT
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    try:
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print(f"Loaded PEFT adapter from {checkpoint_path}")
    except Exception:
        print(f"No PEFT adapter found, using base model from {checkpoint_path}")

    return model, tokenizer
