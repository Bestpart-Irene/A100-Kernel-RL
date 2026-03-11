# KernelForge-OpenEnv

KernelForge is a CUDA-kernel RL repo built around a **custom lightweight OpenEnv wrapper** over a **KernelGYM-style execution backend** for reward-bearing kernel evaluation.

The current hackathon posture is:

- **Training** on A100 80GB
- **Reward-bearing eval** on A100 80GB
- **Active backend**: Modal for cloud bring-up, `local` for same-process smoke checks
- **Optional backend**: CoreWeave/Northflank HTTP service when explicitly configured

References:
- [OpenEnv overview](https://meta-pytorch.github.io/OpenEnv/)
- [OpenEnv core/runtime docs](https://meta-pytorch.github.io/OpenEnv/core/)
- [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv)
- [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM)
- [Dr. Kernel paper](https://arxiv.org/abs/2602.05885)
- [Northflank GPU workloads](https://northflank.com/docs/v1/application/gpu-workloads/gpus-on-northflank)
- [CoreWeave on Northflank](https://northflank.com/docs/v1/application/bring-your-own-cloud/coreweave-on-northflank)

## Architecture

KernelForge should be understood as three layers:

1. **Custom lightweight OpenEnv wrapper**
   - `openenv_env/`
   - standardized `reset()` / `step()` / `state()` surface for judges, demos, and OpenEnv-compatible integrations
   - backed by `openenv.yaml` and `openenv_env/server/app.py`

2. **KernelForge task / reward / rollout logic**
   - `training/task_support.py`
   - `training/multi_turn_rollout.py`
   - `openenv_env/reward.py`
   - `training/custom_grpo_trainer.py`

3. **KernelGYM-style execution backend**
   - `openenv_env/eval_backend.py` — backend switch
   - `eval_service/eval_core.py` — shared pure evaluator core
   - `modal_app.py` — Modal A100 evaluation path
   - `eval_service/app.py` — optional Northflank/CoreWeave FastAPI eval service

## Current Backend Selection

The active evaluator is controlled by environment variables:

```bash
KERNELFORGE_EVAL_BACKEND=modal
KERNELFORGE_MODAL_APP=kernelforge-a100
```

Smoke-test / same-process fallback:

```bash
KERNELFORGE_EVAL_BACKEND=local
```

Optional HTTP backend:

```bash
KERNELFORGE_EVAL_BACKEND=coreweave
KERNELFORGE_EVAL_URL=https://your-northflank-service
```

## What Is Implemented in Code

- OpenEnv server and client packaging
- provider-neutral eval dispatch
- shared pure compile / verify / benchmark core
- FastAPI eval service for CoreWeave/Northflank
- Modal fallback wrapper
- multi-turn rollout path with local compile fast-fail and remote reward dispatch
- Stage 1 / Stage 2 / Stage 3 training entrypoints
- GRPO preflight checks for dataset/assets and backend health

Internal source-of-truth docs:
- [`docs/KERNELFORGE_FINAL_PRD.md`](docs/KERNELFORGE_FINAL_PRD.md)
- [`docs/GRPO_DEEP_DIVE.md`](docs/GRPO_DEEP_DIVE.md)
- [`docs/OPENENV_AUDIT_PLAN_3.md`](docs/OPENENV_AUDIT_PLAN_3.md)

## What Still Needs Live Validation

- end-to-end Stage 1 run on Modal A100
- end-to-end Stage 2 RFT/SFT continuation from a real Stage 1 checkpoint
- end-to-end Stage 3 GRPO pilot on Modal A100
- throughput / stability measurements for the optional HTTP eval service

## Quick Start

### 1. Install the repo

```bash
uv sync --extra train --extra openenv
```

If you need the Modal fallback tooling too:

```bash
uv sync --extra train --extra openenv --extra modal
```

### 2. Run the OpenEnv server locally

```bash
uv run python -m uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000
```

This follows the standard OpenEnv local server pattern documented in the OpenEnv quickstart:
- [OpenEnv quickstart](https://meta-pytorch.org/OpenEnv/quickstart/)

### 3. Preflight the training stack

Primary hackathon path:

```bash
export KERNELFORGE_EVAL_BACKEND=modal
export KERNELFORGE_MODAL_APP=kernelforge-a100
uv run python -m training.grpo_train --preflight-only
```

### 4. Run training stages

```bash
uv run python -m training.grpo_train --stage stage1
uv run python -m training.grpo_train --stage stage2
uv run python -m training.grpo_train --stage stage3
```

### 5. Modal fallback path

```bash
modal deploy modal_app.py
modal run modal_train.py --stage 0
modal run modal_train.py --stage 1
modal run modal_train.py --stage 2
modal run modal_train.py --stage 3
```

Legacy experimental files remain in the repo for reference, but they are not part of the default path:

- `training/stage0_sft.py`
- `training/stage1_enhanced_grpo.py`
- `training/test_sft_model.py`
- `scripts/run_full_pipeline.py`
- `docs/OPTIMIZATION_PLAN.md`

## OpenEnv + GRPO Integration

KernelForge uses the OpenEnv-compatible environment surface for standards compatibility, but the GRPO inner loop may still call the shared evaluator contract directly through the custom rollout path. That is consistent with TRL's documented `rollout_func` model for OpenEnv integrations:
- [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv)

## Docs Map

- [`docs/KERNELFORGE_FINAL_PRD.md`](docs/KERNELFORGE_FINAL_PRD.md) — architecture, rollout order, launch criteria
- [`docs/GRPO_DEEP_DIVE.md`](docs/GRPO_DEEP_DIVE.md) — GRPO/TRLOO strategy and rollout details aligned to the PRD runtime path
- [`docs/OPENENV_AUDIT_PLAN_3.md`](docs/OPENENV_AUDIT_PLAN_3.md) — OpenEnv contract review and architecture audit
- [`docs/skills/CUDA_AGENT.md`](docs/skills/CUDA_AGENT.md) — CUDA-Agent prior
- [`docs/skills/DOUBLEGRAPH_A100.md`](docs/skills/DOUBLEGRAPH_A100.md) — DoubleGraph A100 prior
- [`docs/skills/SKYDISCOVER_ADAEVOLVE_EVOX.md`](docs/skills/SKYDISCOVER_ADAEVOLVE_EVOX.md) — search-system prior
- [`docs/skills/KERNELGYM_DR_KERNEL.md`](docs/skills/KERNELGYM_DR_KERNEL.md) — backend architecture and TRLOO framing

## Remaining Architectural Caveat

One cleanup still remains in code:

- `openenv_env/kernel_forge_env.py` still imports helper logic from `training/task_support.py`

So the wrapper/backend architecture is implemented and documented, but the environment package boundary is not yet perfectly neutral.
