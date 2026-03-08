# KernelForge: OpenEnv + GRPO Audit Record

> **Purpose**: Documents what was wrong, what was fixed and why, and what the codebase looks like now. Serves as a reference for engineers and judges evaluating the hackathon submission.
>
> **Repo**: `A100-Kernel-RL/` — CUDA kernel optimization via RL on A100 GPUs.
> **Status**: All patches applied. **9/9 test_env.py tests pass** (mock target updated `_modal` → `_dispatch`; baseline profiling test uses explicit WCC task). `uv run pytest tests/test_env.py -v`
> **Infra migration note (March 7, 2026):** The active deployment target is now Northflank + CoreWeave. Current Modal-specific code paths in the repo should be treated as legacy transition code until the teammate-owned backend migration lands. Sources: [Northflank GPU workloads](https://northflank.com/docs/v1/application/gpu-workloads/gpus-on-northflank), [CoreWeave on Northflank](https://northflank.com/docs/v1/application/bring-your-own-cloud/coreweave-on-northflank).

---

## What This System Does

KernelForge trains an LLM (Qwen3-Coder-30B-A3B) to write optimized CUDA kernels using GRPO (Group Relative Policy Optimization). The system has two paths:

1. **Judge/Demo path**: An OpenEnv HTTP server where external users submit CUDA kernels and get compile/correctness/speedup feedback
2. **Training path**: A TRL GRPOTrainer that generates CUDA code, evaluates it on remote A100 GPUs through the shared backend adapter, and uses discrete rewards {-1, 1, 2, 3} to improve the policy

Both paths share the same reward logic (`reward.py`) and the same evaluator contract through `training/task_support.py` plus `openenv_env/eval_backend.py`. The actual compile / verify / benchmark implementation now lives in `eval_service/eval_core.py`, which is reused by the Northflank/CoreWeave FastAPI service and the Modal fallback wrapper. Sources: [Northflank GPU workloads](https://northflank.com/docs/v1/application/gpu-workloads/gpus-on-northflank), [CoreWeave on Northflank](https://northflank.com/docs/v1/application/bring-your-own-cloud/coreweave-on-northflank).

### System Topology
```
JUDGE / DEMO PATH                          TRAINING PATH
━━━━━━━━━━━━━━━━━                          ━━━━━━━━━━━━━━
Judges / Streamlit / curl                  GRPOTrainer (TRL)
         │                                        │
         ▼                                        ▼
KernelForgeClient (HTTP)               rollout_func (multi_turn_rollout.py)
         │                                        │
         ▼                                        │
OpenEnv HTTP Server                               │
  (server/app.py)                                 │
         │                                        │
         ▼                                        │
KernelForgeEnv.step()                             │
         │                                        │
         ├────────────────────────────────────────┘
         ▼
  SHARED CORE LOGIC
  ├── task_support.py   (task routing, payload shaping, reward contract)
  ├── eval_backend.py   (CoreWeave HTTP default, Modal fallback)
  ├── reward.py         (compute_reward: discrete {-1,1,2,3})
  ├── anti_hack.py      (forbidden symbol scan + 5 runtime anti-hack checks)
  └── skill_builder.py  (SKILL.md generation)
        │
        ▼
  Remote A100 Backend
  ├── eval_service/eval_core.py   (shared pure evaluator core)
  ├── eval_service/app.py         (Northflank/CoreWeave FastAPI service)
  └── modal_app.py                (legacy/fallback wrapper around eval_core)
```

---

## What Was Wrong (Original State)

17 bugs/misalignments were identified across 3 workstreams: OpenEnv contract compliance, test alignment with discrete rewards, and GRPO training stabilization.

| # | Problem | File | Severity | Status |
|---|---------|------|----------|--------|
| 1 | No `models.py` — Action/Observation defined inline, not in canonical OpenEnv location | `openenv_env/kernel_forge_env.py` | High | **FIXED** |
| 2 | `class KernelForgeEnv(Environment[A,O,S])` used generic subscript — echo_env reference does NOT | `openenv_env/kernel_forge_env.py` | Medium | **FIXED** |
| 3 | `reset(seed, episode_id, **kw)` diverged from the minimal reset surface we standardized for the hackathon path; current OpenEnv docs also permit optional reset kwargs, so the fix was to simplify the environment signature rather than treat kwargs as universally forbidden | `openenv_env/kernel_forge_env.py` | High | **FIXED** |
| 4 | `step(action, timeout_s, **kw)` — the canonical action path is `step(action)`; transport/runtime extras should not be carried as ad hoc step parameters | `openenv_env/kernel_forge_env.py` | High | **FIXED** |
| 5 | `app = create_fastapi_app(...)` inline at module bottom — should be in `server/app.py` | `openenv_env/kernel_forge_env.py` | High | **FIXED** |
| 6 | No `openenv.yaml` manifest | repo root | High | **FIXED** |
| 7 | No `KernelForgeClient(EnvClient[...])` | missing file | Medium | **FIXED** |
| 8 | No `server/` directory structure | missing directory | High | **FIXED** |
| 9 | 7 of 11 reward tests FAILED — asserted `math.log(speedup)` but code returns discrete {-1,1,2,3} | `tests/test_reward.py` | Critical | **FIXED** |
| 10 | 2 of 9 env tests FAILED — same continuous vs discrete mismatch | `tests/test_env.py` | Critical | **FIXED** |
| 11 | `conftest.py` mock `_MockAction` missing `metadata` field, `_MockState` not Pydantic | `tests/conftest.py` | High | **FIXED** |
| 12 | `docker-compose.yml` used port 8080, should be 8000; wrong app path | `docker-compose.yml` | Medium | **FIXED** |
| 13 | vLLM kwargs passed unconditionally + missing GRPO config fields | `training/stage3_grpo.py` | Critical | **FIXED** |
| 14 | Unbounded prompt growth in multi-turn: appended full completion+feedback each turn | `training/multi_turn_rollout.py` | High | **FIXED** |
| 15 | `max_turns` default was 5 — should be 3 for hackathon GRPO efficiency | `training/multi_turn_rollout.py` | Medium | **FIXED** |
| 16 | `modal_train.py` missing vLLM dep and env var forwarding | `modal_train.py` | Medium | **FIXED** |
| 17 | 3 additional test files used continuous rewards / stale assertions | `tests/test_multi_turn_rollout.py`, `tests/test_reward_monitor.py`, `tests/test_gpu_registry.py` | High | **FIXED** |

---

## OpenEnv Protocol Reference

OpenEnv is a **Pydantic-typed HTTP client-server protocol** from Meta-PyTorch. Built using Gymnasium-style APIs (step/reset/state) but is NOT the Gymnasium package.

### Base Types (`openenv.core.env_server.types`)
```python
class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    done: bool = Field(default=False)
    reward: bool | int | float | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class State(BaseModel):
    model_config = ConfigDict(extra="allow")
    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)
```

### Environment Interface
```python
class Environment:
    def __init__(self): ...
    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation: ...
    def step(self, action: Action) -> Observation: ...
    @property
    def state(self) -> State: ...
```

### Server Creation
```python
from openenv.core.env_server.http_server import create_app
app = create_app(EnvClass, ActionClass, ObsClass, env_name="myenv")
```

### Canonical Package Structure (from echo_env reference)
```
my_env/
├── __init__.py       # exports Client, Action, Observation
├── client.py         # MyEnvClient(EnvClient[...])
├── models.py         # MyAction(Action), MyObservation(Observation)
├── openenv.yaml      # manifest
├── server/
│   ├── __init__.py
│   └── app.py        # create_app(EnvClass, Act, Obs, env_name=...)
```

### TRL Integration Contract
```python
from trl.experimental.openenv import generate_rollout_completions

def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
    outputs = generate_rollout_completions(trainer, prompts)
    return {
        "prompt_ids": [...], "completion_ids": [...],
        "logprobs": [...], "env_reward": [...],  # extra field forwarded to reward_funcs
    }
```

---

## Reward Function Reference

**File**: `openenv_env/reward.py` — DO NOT MODIFY this file.

```python
def compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile, ...) -> float:
    if not compiled or not correct: return -1.0
    if speedup_vs_compile > 1.05:   return  3.0   # Beats torch.compile
    if speedup_vs_eager > 1.05:     return  2.0   # Beats eager PyTorch
    return 1.0                                     # Correct but not faster
```

Nsight kwargs (occupancy, mem_coalescing, warp_efficiency) are accepted but **unused** in discrete mode.

---

## What Was Fixed and Why

### Workstream A: OpenEnv Contract Compliance (Patches 1-11)

#### Patch 1: CREATED `openenv_env/models.py`

**Why**: OpenEnv canonical structure requires Action and Observation types in a separate `models.py`. They were defined inline in `kernel_forge_env.py`, violating the echo_env reference pattern.

**What it contains now**:
```python
from openenv.core.env_server.types import Action, Observation

class KernelForgeAction(Action):
    cuda_code: str = Field(..., description="CUDA kernel source code string")

class KernelForgeObservation(Observation):
    text: str = Field(...)
    baseline_original_ms: float | None = None
    baseline_doublegraph_ms: float | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    turn: int = 0
    best_reward: float = -1.0
    info: dict[str, Any] = Field(default_factory=dict)
    graph_properties: dict[str, Any] | None = Field(default=None)
    topology_type: str | None = Field(default=None)
```

#### Patch 2: MODIFIED `openenv_env/kernel_forge_env.py`

**Why**: 4 OpenEnv contract violations needed fixing.

**Changes made**:
- **Imports**: `from openenv.core.env_server import Environment, create_fastapi_app` → `from openenv.core.env_server import Environment` + `from openenv_env.models import KernelForgeAction, KernelForgeObservation`
- **Deleted inline class definitions**: `KernelForgeAction` and `KernelForgeObservation` moved to `models.py`. The import in this file brings them into the module namespace, so `from openenv_env.kernel_forge_env import KernelForgeAction` still works.
- **Class declaration**: `KernelForgeEnv(Environment[KernelForgeAction, KernelForgeObservation, State])` → `KernelForgeEnv(Environment)` (echo_env reference does not use generic subscripts)
- **Method signatures**: `reset(self, seed=None, episode_id=None, **kwargs)` → `reset(self)` for the simplified hackathon surface, and `step(self, action, timeout_s=None, **kwargs)` → `step(self, action: KernelForgeAction)` so evaluation/runtime extras are not passed as ad hoc step parameters
- **Deleted app creation**: `app = create_fastapi_app(...)` at bottom of file moved to `server/app.py`

#### Patch 3: CREATED `openenv_env/server/__init__.py` + `openenv_env/server/app.py`

**Why**: OpenEnv canonical structure requires a `server/` directory with the HTTP app separate from the environment logic.

**What `server/app.py` contains now**:
```python
from openenv.core.env_server.http_server import create_app
from openenv_env.kernel_forge_env import KernelForgeEnv
from openenv_env.models import KernelForgeAction, KernelForgeObservation

app = create_app(
    KernelForgeEnv, KernelForgeAction, KernelForgeObservation,
    env_name="kernelforge",
)
```

#### Patch 4: CREATED `openenv.yaml`

**Why**: OpenEnv requires a manifest file at the repo root.

**Contents**:
```yaml
spec_version: 1
name: kernelforge
type: space
runtime: fastapi
app: openenv_env.server.app:app
port: 8000
```

#### Patch 5: CREATED `openenv_env/client.py`

**Why**: OpenEnv canonical structure requires a typed client. `EnvClient` base class handles serialization generically via Pydantic type params, so the `pass` body is correct.

**Contents**:
```python
from openenv.core.env_client import EnvClient
from openenv.core.env_server.types import State
from openenv_env.models import KernelForgeAction, KernelForgeObservation

class KernelForgeClient(EnvClient[KernelForgeAction, KernelForgeObservation, State]):
    pass
```

#### Patch 6: MODIFIED `openenv_env/__init__.py`

**Why**: Package needed to export the canonical types and lazy-load the client.

**Current contents**:
```python
from openenv_env.models import KernelForgeAction, KernelForgeObservation
from openenv_env.kernel_forge_env import KernelForgeEnv

__all__ = ["KernelForgeEnv", "KernelForgeAction", "KernelForgeObservation"]

def __getattr__(name: str):
    if name == "KernelForgeClient":
        from openenv_env.client import KernelForgeClient
        return KernelForgeClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

#### Patch 7: MODIFIED `tests/conftest.py`

**Why**: Mock classes needed to match OpenEnv base type contracts for tests to pass.

**Changes made**:
- Added `Field` import and `metadata: dict = Field(default_factory=dict)` to `_MockAction` and `_MockObservation` (OpenEnv base types require this field)
- Changed `_MockState` from plain class to `BaseModel` with `model_config = ConfigDict(extra="allow")`, `episode_id: str | None = None`, `step_count: int = 0` (matches `State` contract)
- Removed `__class_getitem__` from `_MockEnvironment` (no longer needed since `KernelForgeEnv` no longer uses generic subscripts)
- Added `openenv.core.env_server.http_server` mock with `create_app` (needed by `server/app.py`)
- Added `openenv.core.env_client` mock (needed by `client.py`)

**Current conftest.py mock modules**:
```python
sys.modules.setdefault("openenv", _openenv)
sys.modules.setdefault("openenv.core", _core)
sys.modules.setdefault("openenv.core.env_server", _env_server)
sys.modules.setdefault("openenv.core.env_server.types", _types)
sys.modules.setdefault("openenv.core.env_server.http_server", _http_server)
sys.modules.setdefault("openenv.core.env_client", _env_client)
```

**Important**: These mocks only apply inside pytest. Plain `python -c` imports require the real OpenEnv SDK installed.

#### Patch 8: REPLACED `tests/test_reward.py`

**Why**: All reward assertions used `math.log(speedup)` (continuous rewards) but `compute_reward()` returns discrete `{-1, 1, 2, 3}`.

**Current test cases** (13 tests):
- `test_compile_fail` → -1.0
- `test_correct_fail` → -1.0
- `test_correct_no_speedup` → 1.0 (speedup=1.0, not >1.05)
- `test_modest_speedup` → 2.0 (eager 1.5 >1.05, compile 0.9 <1.05)
- `test_large_speedup` → 2.0 (eager 3.0 >1.05, compile 1.0 not >1.05)
- `test_slower_than_baseline` → 1.0 (correct but speedup 0.5 <1.05)
- `test_very_slow_clamped` → 1.0 (correct but speedup 0.01 <1.05)
- `test_nsight_ignored` → 2.0 (Nsight kwargs accepted but unused)
- `test_nsight_extreme_values` → 2.0 (out-of-range Nsight values don't affect reward)
- `test_beats_torch_compile` → 3.0 (compile 1.2 >1.05)
- `test_trloo_post_process_g4` → N/(N-1) = 4/3 scaling
- `test_trloo_post_process_g1` → no scaling at N=1
- `test_trloo_post_process_g2` → N/(N-1) = 2/1 scaling

#### Patch 9: MODIFIED `tests/test_env.py`

**Why**: Same continuous vs discrete mismatch, plus import path needed updating.

**Changes made**:
- Import: `from openenv_env.kernel_forge_env import (...)` → `from openenv_env import KernelForgeEnv, KernelForgeAction, KernelForgeObservation`
- `test_step_correct_kernel`: `math.log(10.0/12.0)` → `1.0` (speedup 0.833 <1.05 → correct but not faster)
- `test_step_fast_kernel`: `math.log(2.0)` → `2.0` (speedup 2.0 >1.05 → beats eager)
- Removed unused `import math`

#### Patch 10: MODIFIED `docker-compose.yml`

**Why**: OpenEnv server service had wrong port (8080) and wrong app path.

**Changes made**:
- `openenv_env.kernel_forge_env:app` → `openenv_env.server.app:app`
- Port `8080:8080` → `8000:8000`

#### Patch 11: MODIFIED `Dockerfile`

**Why**: Needed port 8000 exposure and env-switchable entrypoint for judge deployment.

**Changes made**:
- Added `EXPOSE 8000` (alongside existing `EXPOSE 8501` for Streamlit)
- CMD changed to env-switchable: `KERNELFORGE_MODE=server` runs uvicorn, default runs Streamlit demo

### Workstream B: GRPO Training Stabilization (Patches 12-16)

#### Patch 12: MODIFIED `training/stage3_grpo.py`

**Why**: Several GRPO configuration issues.

**Changes made**:
- **Added imports**: `from trl import GRPOConfig, GRPOTrainer` (was missing `GRPOTrainer` for A/B testing)
- **Added env var constants**: `VLLM_MODE`, `VLLM_SERVER_BASE_URL`, `USE_TRLOO`
- **Added config constants** (post-review fix): `PER_DEVICE_BATCH_SIZE = 1`, `GRADIENT_ACCUMULATION_STEPS = 4`, `NUM_GENERATIONS = 2` — shared between `_validate_config()` and `grpo_kwargs` so they can't drift
- **Added `_validate_config()`**: Verifies effective batch divisible by num_generations, and vLLM server URL set when needed. Uses the module-level constants, not hardcoded literals.
- **Made vLLM kwargs conditional** (post-review fix): `use_vllm`, `vllm_mode`, `vllm_server_base_url` only passed when `USE_VLLM=True`. `vllm_gpu_memory_utilization` only for `colocate` mode.
- **Added missing GRPOConfig fields**: `beta=0.0` (no ref model), `remove_unused_columns=False` (custom rollout needs extra columns), `scale_rewards="batch"` (better for sparse expensive env), `gradient_checkpointing=True`, `max_prompt_length=3072`, `epsilon=0.2`, `save_steps=25`, `save_total_limit=2`
- **Made trainer class selectable**: `TRLOOGRPOTrainer if USE_TRLOO else GRPOTrainer` (A/B testable via env var)
- **Reduced max_completion_length default**: 1024 → 768 (kernel code doesn't need 1024 tokens)

#### Patch 13: MODIFIED `training/multi_turn_rollout.py`

**Why**: Unbounded prompt growth would exhaust context window in multi-turn episodes.

**Changes made**:
- **max_turns default**: 5 → 3 (hackathon efficiency)
- **Added truncation constants**: `MAX_FEEDBACK_CHARS=1200`, `MAX_ERROR_CHARS=800` (env-configurable)
- **Applied truncation in `_format_feedback`**: Error text truncated to `MAX_ERROR_CHARS`, total feedback to `MAX_FEEDBACK_CHARS`
- **Fixed prompt growth**: Instead of `current_prompt = f"{current_prompt}\n\n{completion_text}\n\n{feedback}"` (which accumulated every turn), now rebuilds from base: `build_generation_prompt(task_row, ...) + f"\n\n{feedback}"`

#### Patch 14: MODIFIED `modal_train.py`

**Why**: vLLM dependency missing from Modal image, and GRPO env vars not forwarded to remote function.

**Changes made**:
- Added `"vllm>=0.10.2"` to `train_image.uv_pip_install()` list
- Added env var forwarding in `train()`: `KERNELFORGE_USE_VLLM`, `KERNELFORGE_VLLM_MODE`, `KERNELFORGE_VLLM_SERVER_BASE_URL`, `KERNELFORGE_USE_TRLOO`, `KERNELFORGE_STAGE3_SCALE_REWARDS`, `KERNELFORGE_STAGE3_BETA`, `KERNELFORGE_STAGE3_MAX_PROMPT_LENGTH`, `KERNELFORGE_STAGE1_MAX_TURNS`, `KERNELFORGE_STAGE3_MAX_TURNS`, `CUDA_AGENT_STAGE1_SAMPLES`, `KERNELFORGE_STAGE3_OPS6K_MAX`

### Workstream C: Additional Test Alignment (Patches 15-17)

These test files were not covered by the original 14-patch plan but failed after the patches were applied because they also used continuous reward assumptions.

#### Patch 15: MODIFIED `tests/test_multi_turn_rollout.py`

**Why**: `TestComputeReward` tests asserted `math.log(speedup)` values; `TestFormatFeedback` asserted text ("Modest speedup") that doesn't exist in the actual feedback.

**Changes made**:
- Removed `import math`
- `test_correct_slower`: `pytest.approx(math.log(0.9))` → `assert r == 1.0` (speedup 0.9 <1.05)
- `test_modest_speedup`: `pytest.approx(math.log(1.2))` → `assert r == 2.0` (eager 1.2 >1.05, compile 0.9 <1.05)
- `test_large_speedup`: `pytest.approx(math.log(3.0))` → `assert r == 3.0` (compile 1.5 >1.05)
- `test_correct_slow`: reward value changed from `math.log(0.8)` to `1.0`
- `test_correct_modest_speedup`: reward value changed from `math.log(1.5)` to `2.0`; assertion changed from `"Modest speedup"` to `"torch.compile"` (matches actual feedback text: "Faster than eager PyTorch but not torch.compile")

#### Patch 16: MODIFIED `tests/test_reward_monitor.py`

**Why**: Tests used `math.log()` values as rewards, but `check_reward_distribution()` expects discrete `{-1, 1, 2, 3}`. Tier rate keys didn't match actual implementation.

**Changes made**:
- Removed `import math`
- `test_healthy_distribution`: `[-1.0, -0.2, 0.0, 0.1, math.log(1.4), math.log(2.1)]` → `[-1.0, 1.0, 1.0, 2.0, 2.0, 3.0]`
- `test_all_max_reward_flagged`: `[math.log(4.0)] * 100` → `[3.0] * 100`
- `test_bimodal_flagged`: `[-1.0] * 50 + [math.log(3.0)] * 50` → `[-1.0] * 50 + [3.0] * 50`
- `test_tier_rates`:
  - Input: `[-1.0, 0.0, math.log(1.25), math.log(2.0)]` → `[-1.0, 1.0, 2.0, 3.0]`
  - Key fixes: `speedup_rate` → `speedup_eager_rate`, `top_rate` → `speedup_compile_rate` (matches actual `reward_monitor.py` implementation)
  - Value fixes: adjusted expected rates to match discrete tier thresholds (`fail_rate=0.25`, `correct_rate=0.75`, `speedup_eager_rate=0.50`, `speedup_compile_rate=0.25`)

#### Patch 17: MODIFIED `tests/test_gpu_registry.py`

**Why**: GPU registry now includes H200 (added for training GPU support) but test expected only 3 GPUs.

**Changes made**:
- `test_registry_has_three_gpus` → renamed to `test_registry_has_all_gpus`
- Expected set: `{"a100", "h100", "b200"}` → `{"a100", "h100", "b200", "h200"}`

---

## Current File Tree

```
A100-Kernel-RL/
├── openenv.yaml                              # Patch 4 (NEW)
├── openenv_env/
│   ├── __init__.py                           # Patch 6 (MODIFIED)
│   ├── models.py                             # Patch 1 (NEW)
│   ├── client.py                             # Patch 5 (NEW)
│   ├── kernel_forge_env.py                   # Patch 2 (MODIFIED)
│   ├── server/                               # Patch 3 (NEW)
│   │   ├── __init__.py
│   │   └── app.py
│   ├── reward.py                             # UNCHANGED
│   ├── anti_hack.py                          # UNCHANGED
│   ├── skill_builder.py                      # UNCHANGED
│   ├── gpu_registry.py                       # UNCHANGED (H200 was added prior)
│   └── cache_pool.py                         # UNCHANGED
├── training/
│   ├── multi_turn_rollout.py                 # Patch 13 (MODIFIED)
│   ├── stage3_grpo.py                        # Patch 12 (MODIFIED)
│   ├── custom_grpo_trainer.py                # UNCHANGED
│   ├── task_support.py                       # UNCHANGED
│   └── ...
├── modal_app.py                              # UNCHANGED
├── modal_train.py                            # Patch 14 (MODIFIED)
├── Dockerfile                                # Patch 11 (MODIFIED)
├── docker-compose.yml                        # Patch 10 (MODIFIED)
├── tests/
│   ├── conftest.py                           # Patch 7 (MODIFIED)
│   ├── test_reward.py                        # Patch 8 (REPLACED)
│   ├── test_env.py                           # Patch 9 (MODIFIED)
│   ├── test_multi_turn_rollout.py            # Patch 15 (MODIFIED)
│   ├── test_reward_monitor.py                # Patch 16 (MODIFIED)
│   ├── test_gpu_registry.py                  # Patch 17 (MODIFIED)
│   ├── test_anti_hack.py                     # UNCHANGED
│   ├── test_cache_pool.py                    # UNCHANGED
│   ├── test_compile.py                       # UNCHANGED
│   ├── test_curriculum.py                    # UNCHANGED
│   ├── test_pac_verify.py                    # UNCHANGED
│   ├── test_pass_at_k.py                     # UNCHANGED
│   └── test_skill_builder.py                 # UNCHANGED
└── ...
```

---

## Expected Output

### Test Suite

```bash
$ uv run pytest tests/ -q
........................................................................ [ 63%]
..........................................                               [100%]
114 passed in 0.65s
```

All 114 tests pass across 12 test files:
| File | Tests | Status |
|------|-------|--------|
| `test_anti_hack.py` | 12 | Pass |
| `test_cache_pool.py` | 9 | Pass |
| `test_compile.py` | 4 | Pass |
| `test_curriculum.py` | 12 | Pass |
| `test_env.py` | 9 | Pass |
| `test_gpu_registry.py` | 13 | Pass |
| `test_multi_turn_rollout.py` | 15 | Pass |
| `test_pac_verify.py` | 6 | Pass |
| `test_pass_at_k.py` | 7 | Pass |
| `test_reward.py` | 13 | Pass |
| `test_reward_monitor.py` | 7 | Pass |
| `test_skill_builder.py` | 7 | Pass |

### Further Verification (requires external deps)

```bash
# Stage 0 smoke test on Modal (requires Modal auth)
modal run modal_train.py --stage 0

# OpenEnv server (requires openenv-core installed)
pip install "openenv-core[core]>=0.2.1"
uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000

# Test HTTP endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"cuda_code": "__global__ void k() {}"}'

# GRPO smoke test (tiny, proves plumbing works)
KERNELFORGE_USE_VLLM=0 KERNELFORGE_STAGE3_MAX_STEPS=1 \
  modal run modal_train.py --stage 3
```

---

## Key Function Signatures (for cross-referencing)

### `training/task_support.py`
```python
def evaluate_code_remote(cuda_code: str, task_row: dict,
                         baseline_orig_ms: float | None = None,
                         baseline_dg_ms: float | None = None) -> dict
def compute_task_reward(result: dict | None) -> float
def normalize_eval_result(result: dict | None) -> dict
def normalize_task_row(row: dict) -> dict
def build_generation_prompt(task_row: dict, skill_context: str = "",
                            topology_context: str = "") -> str
def build_prompt_lookup(rows: list[dict]) -> dict[str, dict]
```

### `training/curriculum.py`
```python
def format_topology_context(problem: dict) -> str
def format_problem_prompt(problem: dict) -> str
class CurriculumManager:
    def get_problem(self) -> dict
    def record_reward(self, reward: float) -> str | None
    def status(self) -> dict
```

### `openenv_env/reward.py` (DO NOT MODIFY)
```python
def compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile,
                   occupancy=None, mem_coalescing=None, warp_efficiency=None) -> float
    # Returns: -1.0 | 1.0 | 2.0 | 3.0
def trloo_post_process(advantages: list[float], n: int) -> list[float]
```

---

## Architectural Decisions (Do Not Change)

1. **Training rollout talks directly to the shared backend adapter** — NOT through the HTTP env server. This is intentional. The rollout needs multi-turn prompt management that step-level API doesn't handle.
2. **Discrete rewards {-1, 1, 2, 3}** — per CUDA Agent ablation (96.8% vs 60.4% faster rate over continuous). Do not switch to continuous.
3. **Local compile precheck** — `nvcc -arch=sm_80 -c` before remote dispatch. Saves ~50% eval cost.
4. **GPU split** — H200 for training, remote A100 for ALL performance evaluation. Never use training GPU timing for reward.
5. **vLLM server mode** (not colocate) when enabled — training + vLLM + remote reward on one GPU is too many moving parts.
6. **Import path**: `openenv.core.env_server` (NOT `.interfaces`). No `.interfaces` references exist in the codebase or mocks.
7. **`client.py` blank subclass**: `EnvClient` base class handles serialization generically via Pydantic type params. The `pass` body is the documented pattern.

---

## Current Hook Map and Remaining Architecture Holes

OpenEnv and KernelGYM operate at different layers in this repo. KernelForge is building a **custom lightweight OpenEnv wrapper** as the standardized public environment contract: typed models, `step()` / `reset()` / `state()`, HTTP serving, and client packaging. KernelGYM is the execution-system reference underneath that wrapper: separate learning clients from GPU execution, serialize timing-sensitive work, isolate worker failures, and eventually scale through worker-style backend coordination. For KernelForge, the right architecture is therefore: **custom lightweight OpenEnv wrapper at the edge, KernelForge task/reward/rollout logic in the middle, KernelGYM-style backend underneath**. In the current codebase the primary backend target is Northflank + CoreWeave, with Modal retained as fallback only. Sources: [OpenEnv docs](https://meta-pytorch.github.io/OpenEnv/), [OpenEnv environment guide](https://github.com/meta-pytorch/OpenEnv/blob/main/envs/README.md), [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv), [Dr. Kernel / KernelGYM paper](https://arxiv.org/abs/2602.05885), [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM), [Northflank GPU workloads](https://northflank.com/docs/v1/application/gpu-workloads/gpus-on-northflank), [CoreWeave on Northflank](https://northflank.com/docs/v1/application/bring-your-own-cloud/coreweave-on-northflank).

### Current hook map

1. **OpenEnv surface**
   - `openenv_env/models.py` defines the typed `KernelForgeAction` / `KernelForgeObservation` surface.
   - `openenv_env/kernel_forge_env.py` implements the environment state machine.
   - `openenv_env/server/app.py` and `openenv.yaml` expose the HTTP/server manifest.
   - `openenv_env/client.py` provides the typed `EnvClient` wrapper.

2. **KernelForge task/reward layer**
   - `training/task_support.py` holds task routing, payload shaping, result normalization, and canonical reward computation.
   - `training/multi_turn_rollout.py` owns multi-turn prompt refinement, local compile fast-fail, and reward collection for GRPO.
   - `openenv_env/reward.py` and `training/custom_grpo_trainer.py` hold the shared reward and TRLOO logic.

3. **Execution backend**
   - `openenv_env/eval_backend.py` is now the provider-neutral dispatch seam.
   - `eval_service/eval_core.py` is the shared evaluator implementation.
   - `eval_service/app.py` is the primary Northflank/CoreWeave service, while `modal_app.py` is the fallback wrapper.

### Remaining holes in the current gym/setup

1. **The reward-bearing seam is improved but not fully neutralized.**
   - The repo now has a real backend switch and a shared eval core, but `KernelForgeEnv` still imports payload / normalization helpers from `training/task_support.py` instead of a neutral runtime package.

2. **Environment state and training state are still duplicated.**
   - `KernelForgeEnv` tracks turn history and best reward.
   - `multi_turn_rollout.py` separately tracks prompt refinement, baselines, feedback strings, and reward aggregation.

3. **The environment layer depends on a training namespace.**
   - `KernelForgeEnv.step()` imports execution helpers from `training/task_support.py`.
   - This works today, but it is the wrong long-term dependency direction for an OpenEnv-facing environment.

4. **The backend is service-shaped, not yet a full worker system.**
   - `eval_service/app.py` gives the repo a clean remote service boundary, but not yet a full KernelGYM-style worker pool with subprocess-per-task isolation, queueing, and recovery orchestration.
   - The next architectural step is stronger worker isolation, not a new environment API.

5. **The live-evaluable subset is intentionally narrow.**
   - The current supported routing is `wcc` plus the stateless `ops6k` subset.
   - That is the correct hackathon slice, but it is not yet a full kernel gym task catalog.

### Recommended hackathon stance

- **Keep the custom lightweight OpenEnv wrapper at the edge.**
  - Judges, demos, and TRL/OpenEnv integrations should continue to see `openenv_env/` as the official environment package, even though RL/GRPO may call the shared backend contract directly for efficiency.
- **Refactor only the remaining training-namespace dependency next.**
  - The adapter and shared eval core now exist; the next cleanup is to move payload/result helpers out of `training/task_support.py` so OpenEnv no longer depends on a training package.
- **Keep direct training transport for now.**
  - GRPO should keep using the shared adapter directly instead of paying an HTTP hop inside the inner loop.
- **Adopt KernelGYM ideas selectively.**
  - Add subprocess isolation and worker-style backend structure first; defer Redis-style full distributed orchestration until after benchmark validation.

## Post-Review Corrections

Three real bugs were caught by teammate review and fixed in both this spec and the code:

1. **`_validate_config()` used hardcoded literals.** Fixed: extracted `PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `NUM_GENERATIONS` as module-level constants shared with `grpo_kwargs`.

2. **vLLM kwargs passed unconditionally.** Fixed: `use_vllm`, `vllm_mode`, `vllm_server_base_url`, and `vllm_gpu_memory_utilization` are now only passed when `USE_VLLM=True`. `vllm_gpu_memory_utilization` only for `colocate` mode.

3. **Verification `python -c` overclaim.** Fixed: `tests/conftest.py` mocks only apply inside pytest, not standalone `python -c`. Verification updated to use `uv run pytest`.

### Teammate claims reviewed and rejected:

- **Import path `.interfaces`**: Teammate said use `openenv.core.env_server.interfaces`. Wrong — the codebase uses `openenv.core.env_server` everywhere, conftest mocks that path.
- **`client.py` needs real methods**: Wrong — `EnvClient` base class handles serialization generically.
- **`build_generation_prompt` signature concern**: Verified exact match in `training/task_support.py`.
- **Modal env var ordering**: `modal_train.py` sets env vars before stage imports, ordering is correct.
- **`openenv-core[core]>=0.2.1` concern**: Matches `pyproject.toml` and `modal_train.py`.

---

## Environment Redesign (March 8, 2026)

Major changes pushed after the original 17-bug audit:

| Change | File | Impact |
|--------|------|--------|
| Task pool sampling on `reset()` | `openenv_env/task_pool.py` | Replace hardcoded WCC with pool sampling |
| `max_turns=3` | `openenv_env/kernel_forge_env.py` | Dr. Kernel MAX_TURN=3 |
| 5 anti-hack runtime checks | `openenv_env/anti_hack.py` | Wired into `eval_core.py:732-773` |
| `_dispatch()` replaces `_modal()` | `openenv_env/kernel_forge_env.py` | Backend-neutral dispatch |
| Dependency inversion fix | `openenv_env/task_routing.py` | Env no longer imports training/ |
| Benchmark tooling | `scripts/run_benchmark.py` | KernelBench-compatible fast_p metrics |
| Task pool builder | `tasks/build_task_pool.py` | Curate Ops-6K → pool JSONL |
| test_env.py mock update | `tests/test_env.py` | `_modal` → `_dispatch` mock target |

**Full environment documentation:** See `docs/KERNELFORGE_RL_ENVIRONMENT.md` and `docs/SYSTEM_TRUTH.md`.
