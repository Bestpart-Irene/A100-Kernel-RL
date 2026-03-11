# 003 — Stabilization (Last Working State)

**Date**: 2026-03-10
**Version**: v1.2
**Commit**: `85f7c8c`

---

## Problem

Needed to upgrade from transformers 4.x to 5.2.0 for Qwen 3.5 architecture support. This broke Unsloth patching (RoPE shape mismatch) and torch.compile (Unsloth bug #4025).

## Solution

- Upgraded to transformers 5.2.0
- Disabled Unsloth patching (`KERNELFORGE_SKIP_UNSLOTH=1`)
- Disabled `torch.compile` (3 env vars)
- Added VLM detection logging (auto-extracts text LM from multimodal model)
- Fixed TRLOO advantage correction (was dead code before this)

## What Changed

| Action | File/Path | Why |
|--------|-----------|-----|
| UPDATED | `modal_train.py` | transformers 5.2.0, disable Unsloth + torch.compile |
| UPDATED | `training/custom_grpo_trainer.py` | Fixed TRLOO dead override |
| UPDATED | `training/model_loader.py` | VLM detection + text LM extraction |

## Working Config at This Point

| Setting | Value |
|---------|-------|
| Model | `opus_9b` (9B) |
| Eval backend | local (changed to modal for production) |
| `num_generations` (G) | 4 |
| `max_completion_length` | 2048 |
| LoRA rank | 64 |
| Quantization | bf16 (none) |
| Batch | 1×4 = 4 |
| Prompts | Raw strings (no chat format) |
| Architecture | Multi-turn rollout with `reward_from_env` |

## Reality

This was the last fully working state. Training completed with positive rewards and non-zero gradients. All subsequent changes until v1.8 broke the pipeline.
