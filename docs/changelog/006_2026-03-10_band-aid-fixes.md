# 006 — Band-aid Fixes

**Date**: 2026-03-10
**Versions**: v1.5, v1.6, v1.7
**Commits**: `1788e50`, `8788b19`, `7a78b3a`

---

## Problem

v1.4's chat-format prompts caused type errors throughout the pipeline. Three rapid patches attempted to fix symptoms without addressing root cause.

## Solution (WRONG — treated symptoms, not cause)

### v1.5 — `1788e50` — Band-aid fix

Added `str(list)` handling in `extract_cuda_code` for chat-format inputs. This converted `[{"role": "assistant", "content": "..."}]` to its Python repr string — garbage output.

### v1.6 — `8788b19` — Proper type fix

Added `_to_text()` helper to extract text from chat-format completions/prompts. Fixed the immediate `expected string or bytes-like object, got 'list'` error.

### v1.7 — `7a78b3a` — Prompt lookup fix

Chat-format prompts didn't match lookup dict keys (which used raw strings). Added fallback strategies for prompt matching.

## What Changed

| Action | File/Path | Why |
|--------|-----------|-----|
| UPDATED | `openenv_env/reward.py` | v1.5: str(list) hack, v1.6: proper _to_text() |
| UPDATED | `training/multi_turn_rollout.py` | v1.7: prompt lookup fallback |

## Reality

Type errors fixed, but training still produced zero learning. The root cause (G=2, LoRA 16, 4-bit quant from v1.4) was untouched. All 5 runs: `loss~=0`, `grad_norm~=0`, mean rewards -0.3 to -1.0.
