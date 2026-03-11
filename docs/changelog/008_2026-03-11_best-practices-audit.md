# 008 — Best Practices Audit

**Date**: 2026-03-11
**Version**: v2.0
**Author**: Claude Code (automated)

---

## Problem

The 9B model (Qwen 3.5) passed a 5-step smoke test on Modal A100-80GB (reward -0.625 → -0.475, eval_ok 1/4 → 3/4), but several config values were suboptimal:

- TRL 0.24.0 in Modal image despite pyproject.toml requiring 0.29.0
- G=4 instead of consensus-recommended G=8
- flash-attn disabled (removed due to old incompatibility, now fixed upstream)
- torch.compile disabled (Unsloth bug, but we bypass Unsloth)
- max_steps=5 (smoke-test default, not a real training run)

Before committing real GPU hours, needed to audit config against Unsloth, TRL, and GRPO literature.

## Solution

Applied 6 config changes to align with best practices:

| # | Change | Before | After | Why |
|---|--------|--------|-------|-----|
| 1 | TRL version | 0.24.0 | **0.29.0** | 5 months of bug fixes, matches pyproject.toml |
| 2 | G (num_generations) | 4 | **8** | Consensus recommendation (Unsloth, DeepSeek-R1). 2x better reward variance |
| 3 | grad_accum_steps | 4 | **8** | Match G=8 for batch divisibility (1x8=8, divisible by G=8) |
| 4 | flash-attn | removed | **>=2.5.0** | 10-20% attention speedup. 2.5+ works with torch 2.9 + CUDA 12.4 + Qwen 3.5 |
| 5 | torch.compile | disabled (3 env vars) | **re-enabled** | Bug #4025 only affects Unsloth's compiled trainer; we bypass via SKIP_UNSLOTH=1 |
| 6 | max_steps default | 5 | **100** | 5 was smoke-test only. Unsloth recommends 300+; 100 is a reasonable first real run |

## What Changed

| Action | File/Path | Why |
|--------|-----------|-----|
| UPDATED | `modal_train.py:43` | TRL 0.24.0 → 0.29.0 |
| UPDATED | `modal_train.py:36` | Added flash-attn>=2.5.0 pip install |
| UPDATED | `modal_train.py:60-66` | Removed torch.compile disable env vars |
| UPDATED | `modal_train.py:166` | G=4 → G=8 |
| UPDATED | `modal_train.py:168` | grad_accum 4 → 8 |
| UPDATED | `modal_train.py:170` | max_steps 5 → 100 |
| UPDATED | `training/custom_grpo_trainer.py` | Updated stale TRL 0.24.0 docstring comments |

## Thought Process

### TRL 0.29.0 compatibility

Verified `_generate_and_score_completions` exists in TRL 0.29.0 with the same signature. The custom `TRLOOGRPOTrainer` override is fully compatible — no code changes needed.

### Why G=8 not G=16?

G=8 doubles variance over G=4 at 2x generation cost (~15 min/step). G=16 would be 4x cost for diminishing returns. G=8 is the sweet spot per Unsloth and DeepSeek-R1 literature.

### Why re-enable torch.compile?

Unsloth bug #4025 only affects Unsloth's `chunked_hidden_states_selective_log_softmax`. We set `KERNELFORGE_SKIP_UNSLOTH=1`, so Unsloth's compiled code is never used. Vanilla torch.compile on the model should give 5-15% forward pass speedup.

## Constraints & Trade-offs

| Constraint | Trade-off Made | Rationale |
|------------|---------------|-----------|
| flash-attn might fail to build | xformers still installed as fallback | Low risk, high reward |
| torch.compile might cause shape errors | Test with 1-step smoke test first | Re-disable if broken |
| G=8 doubles step time | ~15 min/step instead of ~8 | Better reward variance worth the cost |
| 100 default steps = ~25 hours | Override with `--max-steps 50` for first run | Smoke tests use `--max-steps 1` or `--max-steps 5` |

## What was NOT changed (and why)

| Setting | Value | Rationale |
|---------|-------|-----------|
| beta | 0.0 | Matches DAPO findings, saves VRAM (no ref model) |
| temperature | 1.0 (Stage 1) | High exploration correct for warmup phase |
| LoRA rank | 64 | Code gen needs high rank per literature |
| bf16 | Full precision | Stable; FP16 only if reward collapse observed |
| vLLM | disabled | Blocked by transformers 5.x incompatibility |
| SKIP_UNSLOTH | 1 | Qwen 3.5 RoPE bug in Unsloth patching |
| OVF | disabled | Enable after confirming G=8 works correctly |

## What was deferred

- **causal-conv1d**: Still removed (build fails on Modal). Not critical for correctness.
- **WandB logging**: Add `report_to="wandb"` separately when ready to track runs.

## Expected Output

1. `--max-steps 1`: Image builds (flash-attn + TRL 0.29.0), model loads, step completes
2. `--max-steps 5`: G=8 generation works, reward_std > 0, torch.compile stable
3. `--max-steps 50`: ~12 hours. Target reward_mean > -0.2, eval_ok > 50% by step 50

## Reality

_To be filled in after verification runs._

- [ ] flash-attn builds successfully on Modal CUDA 12.4 image
- [ ] TRL 0.29.0 resolves without dependency conflicts (Unsloth caps trl)
- [ ] torch.compile doesn't cause shape tracing errors with VLM-extracted CausalLM
- [ ] G=8 generation completes without OOM on A100-80GB
- [ ] Step time ~15 min (8 generations x 1024 tokens)
- [ ] reward_std > 0 on every step
