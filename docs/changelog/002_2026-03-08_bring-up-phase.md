# 002 — Bring-up Phase

**Date**: 2026-03-08
**Version**: v1.1
**Commits**: `3aa9950` through `78dbcd5`

---

## Problem

v1.0 code didn't run on Modal. Multiple dependency conflicts, training hangs, and model loading failures needed fixing before a single GRPO step could complete.

## Solution

Iterative debugging of the full stack — pinning deps, fixing hangs, adding safety checks:

- Pinned `trl==0.24.0`, `transformers==4.57.6`, `vllm==0.12.0`
- Added spawn guard + canary generation test
- Removed broken flash-attention
- Added `rollout_completions` compat shim for trl<=0.24.0
- Added quantized model loading (4-bit/8-bit via BitsAndBytes)
- Added model scaling ladder (2B, 9B, 27B, 35B MoE) in `configs/scaling_ladder.json`
- Fixed registry-driven model selection and GRPO rewards

## What Changed

| Action | File/Path | Why |
|--------|-----------|-----|
| UPDATED | `modal_train.py` | Pinned deps, removed flash-attn, added spawn guard |
| UPDATED | `training/stage1_warmup.py` | Added canary generation test |
| CREATED | `training/model_loader.py` | Quantized model loading |
| CREATED | `training/model_registry.py` | Registry-driven model selection |
| UPDATED | `configs/scaling_ladder.json` | Full model ladder |

## Reality

Training runs with positive rewards achieved. Modal eval backend working. First successful end-to-end GRPO step completed.
