# KernelForge Training — Changelog

Chronological record of changes to the KernelForge GRPO training pipeline, including what broke and why.

---

## v1.0 — `ef0ca70` — Mar 8, 10:50 AM

**Initial commit.** OpenEnv RL environment for CUDA kernel generation.

- KernelForge-OpenEnv 3-layer architecture (OpenEnv wrapper → Task/Rollout → Eval Backend)
- Stage 1 GRPO warm-up training with TRL GRPOTrainer
- Modal deployment for eval service and training
- Multi-turn rollout engine with CUDA code extraction

---

## v1.1 — `3aa9950` through `78dbcd5` — Mar 8, 11:56 AM – 5:44 PM

**Bring-up phase.** Fixed Modal deps, training hangs, model loading, reward signal.

- Pinned `trl==0.24.0`, `transformers==4.57.6`, `vllm==0.12.0`
- Added spawn guard + canary generation test
- Removed broken flash-attention
- Added `rollout_completions` compat shim for trl<=0.24.0
- Added quantized model loading (4-bit/8-bit via BitsAndBytes)
- Added model scaling ladder (2B, 9B, 27B, 35B MoE) in `configs/scaling_ladder.json`
- Fixed registry-driven model selection and GRPO rewards

**Status:** Training runs with positive rewards achieved. Modal eval backend working.

---

## v1.2 — `85f7c8c` — Mar 10, 2:17 PM *(LAST WORKING STATE)*

**Stabilization.** Fixed TRLOO dead override, upgraded to transformers 5.2.0 for Qwen 3.5.

- Added VLM detection logging (auto-extracts text LM from multimodal model)
- Disabled Unsloth patching (`KERNELFORGE_SKIP_UNSLOTH=1`) due to Qwen 3.5 RoPE shape mismatch bug
- Disabled `torch.compile` (Unsloth bug #4025)

**Working config at this point:**

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

---

## v1.3 — `d618c07` — Mar 10, 6:10 PM

**Docs alignment.** Mostly README/docs updates to reflect A100/Modal as default path. Minor config changes in eval_backend defaults. Low risk.

---

## v1.4 — `8de4bc0` — Mar 10, 6:28 PM *(BREAKING COMMIT)*

**Attempted "optimization" of 9B training.** Cut multiple corners simultaneously, breaking the training signal.

| Change | Before (v1.2) | After (v1.4) | Impact |
|--------|---------------|--------------|--------|
| `num_generations` | 4 | **2** | **CRITICAL** — Zero reward variance → zero gradients |
| LoRA rank | 64 | **16** | Reduced model capacity for code generation |
| Quantization | bf16 | **4-bit QLoRA** | Reduced model quality |
| Prompts | Raw strings | **Chat format + system prompt** | Type errors downstream |
| Batch | 1×4 | **2×1** | Changed effective batch math |
| `max_completion_length` | 2048 | **1024** | 100% truncation (model uses all tokens for reasoning) |

### Why G=2 kills GRPO

GRPO computes per-group advantages by comparing rewards within a group of G completions for the same prompt. With G=2:
- Both samples almost always fail the same way
- Identical rewards → `reward_std = 0` → zero advantages → zero gradients → no learning

Literature recommends G≥4 minimum, G=8–16 preferred. Kevin-32B used G=16.

### Why chat-format prompts broke extraction

TRL passes completions back in the same format as prompts. Chat-format prompts produce chat-format completions (lists of dicts), but every downstream function (`extract_cuda_code`, `prompt_lookup`, reward function) expected plain strings.

---

## v1.5 — `1788e50` — Mar 10, 8:47 PM

**Band-aid fix (wrong).** Added `str(list)` handling in `extract_cuda_code` for chat-format inputs.

This converted `[{"role": "assistant", "content": "..."}]` to its Python repr string, producing garbage like `"[{'role': 'assistant', 'content': '...'}]"` instead of extracting the actual text content.

---

## v1.6 — `8788b19` — Mar 10, 8:48 PM

**Proper type fix.** Added `_to_text()` helper to extract text from chat-format completions/prompts. Fixed the immediate `expected string or bytes-like object, got 'list'` error.

---

## v1.7 — `7a78b3a` — Mar 10, 9:13 PM

**Prompt lookup fix.** Chat-format prompts didn't match lookup dict keys (which used raw strings). Added fallback strategies for prompt matching.

**Result of v1.4–v1.7:** Training "runs" but produces zero learning. All 5 runs tonight had `loss≈0`, `grad_norm≈0`, mean rewards -0.3 to -1.0. The type fixes (v1.5–v1.7) addressed symptoms but the root cause (G=2, LoRA 16, 4-bit quant) remained.

---

## v1.8 — Revert to `85f7c8c` — Mar 10, ~10:30 PM

**Full revert.** All training files restored to v1.2 (`85f7c8c`) working state with two config changes:

- Model: `opus_2b` → `opus_9b` (user preference for 9B model)
- Eval backend: `local` → `modal` (production eval service)

Files reverted:
- `modal_train.py`
- `training/stage1_warmup.py`
- `training/grpo_config.py`
- `training/multi_turn_rollout.py`

Training run launched: `ap-4CM6NCqkrIaUuuvrtpkJg2`

---

## Summary: What Broke and Why

The breaking commit (`8de4bc0`) tried to speed up training by reducing computational cost, but cut the training signal itself:

1. **G=2** eliminated reward variance (GRPO needs diversity to learn)
2. **Chat-format prompts** introduced type mismatches throughout the pipeline
3. **LoRA rank 16** was too small for code generation (rank 64 recommended)
4. **4-bit QLoRA** reduced generation quality below the threshold for valid CUDA code
5. **max_completion_length=1024** caused 100% truncation

The fix was a full revert to the last known-good state, not incremental patching. See `docs/SPEED_OPTIMIZATION.md` for how to properly speed up training without cutting corners.
