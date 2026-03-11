# 005 — Breaking Optimization

**Date**: 2026-03-10
**Version**: v1.4
**Commit**: `8de4bc0`
**Status**: BREAKING COMMIT

---

## Problem

9B training on A100 was slow. Wanted to speed it up before a real run.

## Solution (WRONG)

Cut multiple corners simultaneously, breaking the training signal:

| Change | Before (v1.2) | After (v1.4) | Impact |
|--------|---------------|--------------|--------|
| `num_generations` | 4 | **2** | **CRITICAL** — Zero reward variance → zero gradients |
| LoRA rank | 64 | **16** | Reduced model capacity for code generation |
| Quantization | bf16 | **4-bit QLoRA** | Reduced model quality |
| Prompts | Raw strings | **Chat format + system prompt** | Type errors downstream |
| Batch | 1×4 | **2×1** | Changed effective batch math |
| `max_completion_length` | 2048 | **1024** | 100% truncation |

## Thought Process

### Why G=2 kills GRPO

GRPO computes per-group advantages by comparing rewards within a group of G completions for the same prompt. With G=2:
- Both samples almost always fail the same way
- Identical rewards → `reward_std = 0` → zero advantages → zero gradients → no learning

Literature recommends G>=4 minimum, G=8-16 preferred. Kevin-32B used G=16.

### Why chat-format prompts broke extraction

TRL passes completions back in the same format as prompts. Chat-format prompts produce chat-format completions (lists of dicts), but every downstream function (`extract_cuda_code`, `prompt_lookup`, reward function) expected plain strings.

## Reality

Training "runs" but produces zero learning. All 5 runs had `loss~=0`, `grad_norm~=0`, mean rewards -0.3 to -1.0. Required full revert (see 007).
