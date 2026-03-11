# 001 — Initial Commit

**Date**: 2026-03-08
**Version**: v1.0
**Commit**: `ef0ca70`

---

## Problem

No codebase existed for training CUDA kernel generation models via RL. The KernelForge concept (OpenEnv RL environment + GRPO training + Modal GPU deployment) was documented but had zero executable code.

## Solution

Initial implementation of the full KernelForge pipeline:

1. **OpenEnv wrapper** — 3-layer architecture (OpenEnv → Task/Rollout → Eval Backend)
2. **Stage 1 GRPO warm-up** — TRL GRPOTrainer with LoRA fine-tuning
3. **Modal deployment** — eval service and training on cloud GPUs
4. **Multi-turn rollout engine** — CUDA code extraction from model outputs

## What Changed

| Action | File/Path | Why |
|--------|-----------|-----|
| CREATED | `modal_train.py` | Modal training entrypoint |
| CREATED | `training/stage1_warmup.py` | Stage 1 GRPO warm-up |
| CREATED | `training/custom_grpo_trainer.py` | TRLOO-augmented GRPOTrainer |
| CREATED | `training/multi_turn_rollout.py` | Multi-turn rollout engine |
| CREATED | `openenv_env/reward.py` | Reward function for CUDA kernel eval |
| CREATED | `eval_service/` | Local + Modal eval backends |
| CREATED | `configs/scaling_ladder.json` | Model scaling ladder (2B–35B) |

## Reality

First commit — no training runs yet. Pipeline assembled but untested on GPU.
