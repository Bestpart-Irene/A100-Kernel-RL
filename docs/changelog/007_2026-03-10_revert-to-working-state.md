# 007 — Revert to Working State

**Date**: 2026-03-10
**Version**: v1.8
**Base commit**: `85f7c8c` (v1.2)

---

## Problem

v1.4 through v1.7 broke the training signal. Incremental fixes were not converging — the root cause was too many simultaneous changes in v1.4.

## Solution

Full revert of all training files to v1.2 (`85f7c8c`) working state with two config changes:

- Model: `opus_2b` → `opus_9b` (user preference for 9B model)
- Eval backend: `local` → `modal` (production eval service)

## What Changed

| Action | File/Path | Why |
|--------|-----------|-----|
| REVERTED | `modal_train.py` | Back to v1.2 state |
| REVERTED | `training/stage1_warmup.py` | Back to v1.2 state |
| REVERTED | `training/grpo_config.py` | Back to v1.2 state |
| REVERTED | `training/multi_turn_rollout.py` | Back to v1.2 state |

## Thought Process

### Why revert instead of incremental fix?

The breaking commit (v1.4) changed 6 things at once. Fixing them one-by-one would take longer than reverting and re-applying only the valid optimizations. See `docs/SPEED_OPTIMIZATION.md` for the proper approach.

### What broke and why (summary)

1. **G=2** eliminated reward variance (GRPO needs diversity to learn)
2. **Chat-format prompts** introduced type mismatches throughout the pipeline
3. **LoRA rank 16** was too small for code generation (rank 64 recommended)
4. **4-bit QLoRA** reduced generation quality below the threshold for valid CUDA code
5. **max_completion_length=1024** caused 100% truncation

## Reality

Training run launched: `ap-4CM6NCqkrIaUuuvrtpkJg2`. 5-step smoke test completed with improving metrics (reward -0.625 → -0.475, eval_ok 1/4 → 3/4). No positive rewards due to max_completion_length=512 (later fixed back to 1024).
