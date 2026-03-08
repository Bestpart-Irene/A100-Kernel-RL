# KernelGYM / Dr. Kernel — KernelForge Architecture Skill Reference

## Purpose

This file is the authoritative skill doc for how KernelForge should use **KernelGYM** and **Dr. Kernel** ideas while building a **custom lightweight OpenEnv wrapper** over that execution stack rather than replacing **OpenEnv**.

## Authoritative Stance

- **KernelForge is building a custom lightweight OpenEnv wrapper**
  - typed `Action` / `Observation` / `State`
  - `step()` / `reset()` / `state()`
  - lightweight enough to wrap the shared evaluator backend without reproducing all of KernelGYM inside the environment layer
  - usable by judges/demos over HTTP and integrable with RL/GRPO via OpenEnv-compatible interfaces
  - Sources: [OpenEnv docs](https://meta-pytorch.github.io/OpenEnv/), [OpenEnv env authoring guide](https://github.com/meta-pytorch/OpenEnv/blob/main/envs/README.md), [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv)

- **KernelForge = task / reward / rollout logic**
  - task routing
  - prompt construction
  - reward normalization
  - multi-turn rollout behavior
  - TRLOO-corrected GRPO training

- **KernelGYM = execution backend architecture reference**
  - separate learning clients from execution workers
  - isolate failures
  - serialize timing-sensitive evaluation
  - scale from single-node bring-up toward worker-style execution
  - Sources: [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM), [Dr. Kernel paper](https://arxiv.org/abs/2602.05885)

## What KernelForge Should Copy from KernelGYM

### 1. Shared execution plane

Training code should not directly own remote GPU execution policy. It should submit work into a shared backend contract, while the custom lightweight OpenEnv wrapper exposes that contract in environment form when we need judge/demo/standardized-env compatibility.

For KernelForge, that means:

- `openenv_env/eval_backend.py` selects the backend
- `eval_service/eval_core.py` owns pure compile / verify / benchmark logic
- `eval_service/app.py` exposes the remote service boundary
- `modal_app.py` remains a fallback wrapper

### 2. Failure isolation

KernelGYM emphasizes isolation because GPU kernel evaluation is fragile. A bad candidate can crash or poison the process. Source: [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM)

KernelForge implication:

- keep local compile fast-fail
- keep remote execution behind a service boundary
- prefer subprocess-per-eval or worker-pool isolation as the next backend hardening step
- do not let one bad kernel take down the training control plane

### 3. Stable evaluator contract

The interface between learning code and execution code should stay stable while infrastructure changes underneath.

KernelForge contract today:

- `compiles`
- `correct`
- `runtime_ms`
- `runtime_stats`
- `speedup_vs_orig`
- `speedup_vs_dg`
- `verifier_msg`
- `error`

This contract is the reward-bearing seam.

### 4. Worker-style evolution, not full distributed complexity on day one

KernelGYM supports API server + task manager + queue + workers. That is the long-term reference, not a mandatory day-one hackathon dependency. Source: [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM)

KernelForge hackathon interpretation:

- one remote eval service on A100 is enough to launch
- queueing and multi-worker orchestration are later improvements
- keep architecture compatible with future workerization

## What KernelForge Should Copy from Dr. Kernel

### 1. TRLOO / leave-one-out correction

Dr. Kernel highlights the bias in group-relative advantages and the importance of the leave-one-out correction at small `G`, especially `G=2`. Source: [Dr. Kernel paper](https://arxiv.org/abs/2602.05885)

KernelForge implication:

- keep `TRLOOGRPOTrainer`
- keep `N / (N - 1)` scaling active
- do not revert to vanilla GRPO while using small groups

### 2. Multi-turn kernel optimization is evaluation-bound

The lesson is not just RL math. It is that kernel RL succeeds only when evaluation is reliable, isolated, and structurally shared across training and environment-facing flows. Source: [Dr. Kernel paper](https://arxiv.org/abs/2602.05885)

KernelForge implication:

- protect the reward-bearing seam first
- keep local compile filtering
- keep remote A100 timing authoritative

## How the wrapper should be used with RL / GRPO

There are two valid paths, both sharing the same backend contract:

1. **OpenEnv wrapper path**
   - `KernelForgeEnv` exposes the standardized environment surface
   - external users, judges, demos, and any OpenEnv-native tooling talk to this layer

2. **Direct RL / GRPO path**
   - `training/multi_turn_rollout.py` can keep calling the shared evaluator seam directly
   - this avoids paying an extra HTTP hop inside the inner loop
   - this is consistent with TRL's custom `rollout_func` integration model for OpenEnv-backed training. Source: [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv)

The important invariant is that both paths hit the same reward-bearing evaluator contract.

## KernelForge Mapping (Current Repo)

| Layer | Current files | Meaning |
|-------|---------------|---------|
| OpenEnv surface | `openenv_env/kernel_forge_env.py`, `openenv_env/server/app.py`, `openenv_env/client.py`, `openenv.yaml` | Custom lightweight OpenEnv wrapper over the shared backend for judge-facing use and standards compatibility |
| Task / reward layer | `training/task_support.py`, `training/multi_turn_rollout.py`, `openenv_env/reward.py`, `training/custom_grpo_trainer.py` | Project-specific logic |
| Backend switch | `openenv_env/eval_backend.py` | CoreWeave HTTP default, Modal fallback |
| Shared evaluator core | `eval_service/eval_core.py` | Pure compile / verify / benchmark implementation |
| Remote service | `eval_service/app.py`, `eval_service/Dockerfile` | Northflank/CoreWeave deployable A100 service |
| Fallback wrapper | `modal_app.py` | Legacy/fallback transport layer |

## Current Architectural Caveat

KernelForge is now close to the right layering, but one dependency inversion remains:

- `KernelForgeEnv` still imports payload/result helpers from `training/task_support.py`

That means the backend seam exists and is functional, but the environment package still depends on a training namespace.

## What To Do Next

### Immediate next cleanup

Move payload-building and result-normalization helpers out of `training/task_support.py` into a neutral runtime module so:

- OpenEnv no longer imports from `training/`
- training and env both depend on the same neutral runtime package
- the backend contract remains stable regardless of provider

### Backend hardening after that

Add KernelGYM-style worker hardening incrementally:

- subprocess-per-eval isolation
- timeout / retry policy
- persistent baseline cache
- worker health checks
- optional queue-based dispatch if eval concurrency becomes necessary

## What Not To Do

- Do **not** replace OpenEnv with KernelGYM
- Do **not** force training to go through the OpenEnv HTTP server in the inner loop
- Do **not** couple reward logic to provider-specific SDK calls
- Do **not** present Modal as the primary hackathon story now that the backend switch and eval service exist

## Hackathon-Ready Mental Model

Use this sentence consistently:

> **KernelForge is building a custom lightweight OpenEnv wrapper over a KernelGYM-style execution backend. That wrapper gives us a standards-compatible environment surface for judges and demos, while RL/GRPO can integrate directly with the same shared backend contract. Dr. Kernel provides the GRPO/TRLOO learning lesson that keeps the small-group rollout path statistically correct.**

## Related Skill Docs

- `docs/skills/CUDA_AGENT.md`
- `docs/skills/DOUBLEGRAPH_A100.md`
- `docs/skills/SKYDISCOVER_ADAEVOLVE_EVOX.md`
