# KernelForge RL Environment — Complete System Reference

**Last verified:** March 8, 2026
**Hackathon:** OpenEnv Hackathon SF (SHACK15, March 7-8, 2026)
**Repo:** `A100-Kernel-RL`

---

## Reading Guide

| Audience | Start here | Focus sections |
|----------|-----------|----------------|
| **CTO / Judge** | Section 1 (executive summary) + Section 14 (risk assessment) | "Can this actually run?" |
| **Senior Engineer** | Sections 4-6 (environment + reward + eval pipeline) | Code path tracing with file:line citations |
| **New Contributor** | Section 2 (research foundations) + Section 15 (runbook) | End-to-end understanding |

---

## 1. Executive Summary

KernelForge is an **OpenEnv-compatible RL environment** that trains an LLM to generate optimized CUDA kernels for NVIDIA A100 GPUs (`sm_80`). The model receives a PyTorch operator task, reference code, and A100 optimization patterns, then generates a CUDA kernel that gets compiled, verified, and benchmarked on real A100 hardware. Reward is a discrete milestone `{-1, 1, 2, 3}` based on correctness and speedup tiers.

### Nine Core Questions Answered

| # | Question | Answer |
|---|----------|--------|
| 1 | **What A100 kernel are we optimizing?** | PyTorch operator tasks from Ops-6K dataset (6000 operators from CUDA Agent) + WCC graph kernels from doubleGraph. TaskPool samples on `reset()`. |
| 2 | **What benchmark?** | KernelBench-compatible `fast_p` metrics via `scripts/run_benchmark.py` — `fast_0` (correctness), `fast_1.05` (beats eager), `fast_compile` (beats torch.compile), geomean speedups. |
| 3 | **Are we using Dr. Kernel?** | Yes, three elements: TRLOO `N/(N-1)` advantage correction (`custom_grpo_trainer.py`), 5 anti-hack checks (`anti_hack.py` wired into `eval_core.py:732-773`), and `max_turns=3`. |
| 4 | **Are we using KernelBench?** | Yes: `fast_p` metric format, 5-seed correctness protocol, 3-level difficulty structure. We use our own tasks, not their 250. |
| 5 | **CUDA optimization agents?** | Qwen3-Coder-30B-A3B-Instruct (30.5B MoE, 3.3B active) with SKILL.md injection (7 A100 patterns from doubleGraph). NOT a ReAct agent — single/multi-turn code generation. |
| 6 | **How does doubleGraph help?** | 4 ways: (1) 192 expert kernels for SFT data, (2) 7 A100 patterns in SKILL.md, (3) WCC PAC verification with 3 invariants, (4) curriculum phases 2-3 with graph topology. |
| 7 | **What is the evaluation harness?** | `eval_service/eval_core.py` (873 lines): two paths — Ops-6K (torch.utils.cpp_extension compile → 5-seed correctness → anti-hack → benchmark) and WCC (nvcc compile → PAC verify → benchmark). |
| 8 | **Is SkyDiscover integrated?** | Yes but limited. ~950 lines reimplementation of AdaEvolve/EvoX. Mutation is **regex-only**, not LLM-based. Role: hedge path if GRPO doesn't converge. |
| 9 | **Can GRPO actually run?** | Architecture is ready. **Blocked on eval backend deployment.** Without `KERNELFORGE_EVAL_URL` set or Modal configured, training crashes during the first rollout. Nobody has run end-to-end GRPO yet. |

### System Readiness: ~88% Production-Ready

The code is real, not stubs. The **#1 blocker is eval backend deployment** — deploying `eval_service/` to CoreWeave/Northflank with an A100 GPU, or configuring Modal credentials.

---

## 2. Research Foundations

### 2.1 CUDA Agent (ByteDance/Tsinghua, arXiv [2602.24286](https://arxiv.org/abs/2602.24286))

RL-trained Seed1.6 (230B MoE) writes CUDA kernels, achieving 2.11x over torch.compile with 98.8% pass rate.

**What we take:**

| Element | Our code | Verified? |
|---------|----------|-----------|
| Ops-6K dataset (6000 operator tasks) | `training/cuda_agent_integration.py:load_cuda_agent_prompt_dataset()` → [HuggingFace](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K) | YES |
| Discrete milestone reward `{-1,1,2,3}` | `openenv_env/reward.py:compute_reward()` lines 29-69 | YES — matches paper ablation (96.8% vs 60.4%) |
| SKILLS.md concept | `openenv_env/skill_builder.py:build_skill_md()` (247 lines, 7 A100 patterns) | YES |
| 3-file kernel contract (kernel + binding + model) | `eval_service/eval_core.py:evaluate_ops6k_kernel_impl()` lines 584-873 | YES |
| 5-seed correctness check | `eval_service/eval_core.py` lines 704-726 | YES |

**What we DON'T take:** Seed1.6 model (not public), 128-GPU scale, ReAct agent loop, data synthesis pipeline.

### 2.2 Dr. Kernel / KernelGYM (HKUST, arXiv [2602.05885](https://arxiv.org/abs/2602.05885))

Dr. Kernel-14B trained on KernelGYM with TRLOO correction, competitive with Claude-4.5-Sonnet on KernelBench.

**What we take:**

| Element | Our code | Verified? |
|---------|----------|-----------|
| TRLOO `N/(N-1)` advantage correction | `training/custom_grpo_trainer.py:TRLOOGRPOTrainer._compute_advantages()` lines 32-59 | YES — fixes 50% gradient shrinkage at G=2 |
| 5 anti-hack runtime checks | `openenv_env/anti_hack.py` lines 87-274, wired into `eval_core.py` lines 732-773 | YES |
| `max_turns=3` | `openenv_env/kernel_forge_env.py` constructor, line 45 | YES |

**What we DON'T take:** KernelGYM distributed env, Triton-first approach, PR/PRS profiling rewards.

### 2.3 KernelBench (Stanford, arXiv [2502.10517](https://arxiv.org/abs/2502.10517))

250-task GPU kernel benchmark with `fast_p` metric.

**What we take:**

| Element | Our code | Verified? |
|---------|----------|-----------|
| `fast_p` metric format | `scripts/run_benchmark.py:compute_fast_p()` | YES |
| 3-level difficulty structure | `training/curriculum.py:_default_phases()` — 4 phases, 23 tasks | YES |
| 5-seed correctness protocol | `eval_service/eval_core.py` lines 704-726 | YES |

**What we DON'T take:** Their 250 tasks, L40S target (we use A100).

### 2.4 doubleGraph (doubleAI)

192 hyper-optimized A100 graph kernels, 3.6x avg speedup over cuGraph. Source: [github.com/double-ai/doubleGraph](https://github.com/double-ai/doubleGraph).

**4 ways we use it:**

| Use | Code | Verified? |
|-----|------|-----------|
| 192 expert A100 kernels → SFT training data | `datasets/doublegraph_sft.jsonl` (192 rows) | YES |
| 7 engineering patterns → SKILL.md prompt | `openenv_env/skill_builder.py:_append_a100_patterns()` lines 133-246 | YES |
| WCC verification → 3 PAC invariants | `verification/pac_verify.py:verify_wcc()` (225 lines) | YES |
| Graph topology → curriculum phases 2-3 | `training/curriculum.py` lines 108-305 | YES |

We do NOT run doubleGraph at inference time. We teach the model to write kernels LIKE doubleGraph.

---

## 3. Architecture Overview

### GPU Split

| GPU | Role | Cost |
|-----|------|------|
| **H200** (141GB HBM3e) | Model weights, generation, gradient updates, local nvcc compile checks | $4.54/hr |
| **A100 80GB** (CoreWeave/Northflank) | All performance reward: compile, correctness, benchmark timing | Remote |

**H200 timing ≠ A100 timing.** Performance reward must come from A100 execution only.

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING GPU (H200)                   │
│                                                         │
│  Model (Qwen3-Coder-30B-A3B) ← GRPO with TRLOO        │
│       ↓ generates CUDA code                             │
│  multi_turn_rollout.py                                  │
│       ↓ local nvcc compile check (fast-fail)            │
│       ↓                                                 │
│  ┌─── KernelForgeEnv (OpenEnv) ────┐                   │
│  │  reset(task_id) → observation    │                   │
│  │    • SKILL.md (7 A100 patterns)  │                   │
│  │    • task prompt + reference code│                   │
│  │    • interface contract          │                   │
│  │                                  │                   │
│  │  step(cuda_code) → observation   │                   │
│  │    • dispatch to eval backend    │──────────────┐    │
│  │    • compute_reward() → {-1,1,2,3}              │    │
│  │    • history for multi-turn      │              │    │
│  │    • max_turns = 3               │              │    │
│  └──────────────────────────────────┘              │    │
│                                                    │    │
└────────────────────────────────────────────────────│────┘
                                                     │
                    HTTP POST / Modal RPC             │
                                                     ↓
┌─────────────────────────────────────────────────────────┐
│                    EVAL GPU (A100 80GB)                  │
│                                                         │
│  eval_service/app.py (FastAPI on Northflank/CoreWeave)  │
│       ↓                                                 │
│  eval_service/eval_core.py                              │
│       │                                                 │
│       ├── Ops-6K path (evaluate_ops6k_kernel_impl):     │
│       │   1. torch.utils.cpp_extension compile          │
│       │   2. 5-seed correctness vs PyTorch reference    │
│       │   3. Anti-hack: shapes, constant, passthrough   │
│       │   4. Benchmark: median of N runs (cudaEvent)    │
│       │   5. Speedup vs eager + vs torch.compile        │
│       │   6. Anti-hack: no-op check (< 1μs)            │
│       │                                                 │
│       └── WCC path (evaluate_kernel_impl):              │
│           1. nvcc -arch=sm_80 compile                   │
│           2. scan_forbidden_symbols (nm -D)             │
│           3. PAC verify: 5 graphs, 3 invariants         │
│           4. CudaEvent benchmark                        │
│           5. Optional Nsight profiling                  │
│                                                         │
│  modal_app.py (FALLBACK — same eval_core underneath)    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Environment Contract Deep Dive

### 4.1 Types

```python
# openenv_env/models.py
class KernelForgeAction(Action):
    cuda_code: str          # CUDA kernel source code

class KernelForgeObservation(Observation):
    text: str               # Feedback text (SKILL.md / errors / benchmark)
    baseline_original_ms: float | None   # Eager PyTorch baseline
    baseline_doublegraph_ms: float | None # torch.compile baseline
    hardware: dict[str, Any]             # GPU spec from registry
    turn: int = 0
    best_reward: float = -1.0
    info: dict[str, Any]                 # Metadata (task_id, backend, etc.)
    graph_properties: dict | None        # Topology context
    topology_type: str | None            # "power-law", "sparse-islands", etc.
```

**State** (from `openenv.core.env_server.types`): `episode_id`, `step_count`, `history`, `best_reward`.

### 4.2 `reset()` Call Path

File: `openenv_env/kernel_forge_env.py:53-129`

| Step | Line | What Happens |
|------|------|-------------|
| 1 | 67-73 | Clear history, turn, best_reward, baselines. Create new State with UUID. |
| 2 | 76 | `self.task_pool.sample(task_id, seed)` — random task from JSONL pool (`task_pool.py:68-92`). |
| 3 | 77 | `normalize_task_row(task_row)` — standardize fields via `training/task_support.py`. |
| 4 | 80-84 | Check `task_pool.get_cached_baselines(tid)` — skip remote profiling if cached. |
| 5 | 87-96 | For WCC tasks only: `self._dispatch("profile_baselines")` → remote A100 call to `eval_core.py:profile_baselines_impl()`. Sets `original_baseline_ms` and `doublegraph_baseline_ms`. |
| 6 | 98-129 | Build observation: `build_skill_md(gpu)` + task prompt + reference code + `task_interface_contract()`. Return `KernelForgeObservation` with `reward=0.0, done=False, turn=0`. |

### 4.3 `step()` Call Path

File: `openenv_env/kernel_forge_env.py:131-253`

| Step | Line | What Happens |
|------|------|-------------|
| 1 | 138-139 | Increment `self.turn`, update `_state.step_count`. |
| 2 | 142-147 | `build_modal_payload(cuda_code, task)` → constructs eval payload with fn_name (`"evaluate_kernel"` or `"evaluate_ops6k_kernel"`). Source: `training/task_support.py`. |
| 3 | 148 | `self._dispatch(fn_name, payload)` → `eval_backend.py:dispatch_eval()` → HTTP POST to CoreWeave or Modal RPC. |
| 4 | 148 | `normalize_eval_result(result)` — standardize keys, clamp NaN/inf. |
| 5 | 161-189 | Branch on `compiles` and `correct`: compute speedups, call `compute_reward()` → discrete `{-1, 1, 2, 3}`. |
| 6 | 191-201 | Cache baselines from Ops-6K result (if first eval for this task). |
| 7 | 221 | `done = (turn >= max_turns) or (reward >= 3.0)` — early exit on torch.compile beat. |
| 8 | 224-228 | Append to history: `{turn, reward, obs_summary}`. |
| 9 | 231-236 | If multi-turn (history > 1): prepend previous attempts to observation text. |
| 10 | 238-253 | Return `KernelForgeObservation` with reward, done, turn, info, topology context. |

### 4.4 `_dispatch()` — Backend Routing

File: `openenv_env/kernel_forge_env.py:269-272` → `openenv_env/eval_backend.py:18-55`

```python
def _dispatch(self, fn_name, payload=None):
    from openenv_env.eval_backend import dispatch_eval
    return dispatch_eval(fn_name, payload)
```

`dispatch_eval()` routes based on `KERNELFORGE_EVAL_BACKEND`:
- `"coreweave"` (default): HTTP POST to `KERNELFORGE_EVAL_URL/{fn_name}` via httpx (300s timeout).
- `"modal"`: `modal.Function.from_name(app_name, fn_name).remote(payload)`.

### 4.5 `state` Property

File: `openenv_env/kernel_forge_env.py:255-262`

Returns `State(episode_id, step_count=turn, history, best_reward)` — serializable snapshot for TRL/OpenEnv compatibility.

---

## 5. Reward Function

### Discrete Milestones (DO NOT MODIFY)

File: `openenv_env/reward.py:29-69`

```python
def compute_reward(compiled, correct, speedup_vs_eager, speedup_vs_compile, ...):
    if not compiled or not correct:  return -1.0
    if speedup_vs_compile > 1.05:    return  3.0   # beats torch.compile
    if speedup_vs_eager > 1.05:      return  2.0   # beats eager PyTorch
    return 1.0                                      # correct but no speedup
```

### Why Discrete, Not Continuous

CUDA Agent ablation (paper Table 3): discrete milestone reward achieves **96.8%** faster-than-reference rate vs **60.4%** with continuous `log(speedup)` reward. Discrete milestones normalize across problem difficulty — beating torch.compile on sparse graphs earns the same signal as beating it on dense elementwise ops.

### TRLOO Post-Process

File: `openenv_env/reward.py:72-82`

```python
def trloo_post_process(advantages, n):
    scale = n / (n - 1)  # At G=2: scale = 2.0 (fixes 50% shrinkage)
    return [a * scale for a in advantages]
```

Dr. Kernel (arXiv 2602.05885) proves GRPO's self-inclusion bias shrinks expected gradients by `(1 - 1/N)`. At `G=2`, that is 50% per step — TRLOO fixes this with `N/(N-1)` scaling.

### What We Dropped

| Technique | Why Dropped | Reference |
|-----------|------------|-----------|
| Continuous `log(speedup)` reward | CUDA Agent ablation: 96.8% vs 60.4% | reward.py |
| MARS return-to-go credit | Degenerates with outcome-only rewards | GRPO-4 |
| Full Nsight structured reward | API-compatible but unused in discrete mode | GRPO-9 |

Nsight kwargs (`occupancy`, `mem_coalescing`, `warp_efficiency`) are accepted by `compute_reward()` for API compatibility but are **unused** in discrete mode.

---

## 6. Evaluation Pipeline

### 6.1 Dispatch Layer

File: `openenv_env/eval_backend.py` (56 lines)

Routes evaluation calls to one of two backends:

| Backend | Env var | How it works |
|---------|---------|-------------|
| CoreWeave/Northflank (default) | `KERNELFORGE_EVAL_BACKEND=coreweave` | HTTP POST to `KERNELFORGE_EVAL_URL/{fn_name}` via httpx, 300s timeout |
| Modal (fallback) | `KERNELFORGE_EVAL_BACKEND=modal` | `modal.Function.from_name(KERNELFORGE_MODAL_APP, fn_name).remote(payload)` |

### 6.2 FastAPI Service

File: `eval_service/app.py` (76 lines)

5 endpoints, all calling `eval_core.py` functions:

| Endpoint | Function | Purpose |
|----------|----------|---------|
| `POST /evaluate_kernel` | `evaluate_kernel_impl()` | WCC kernel evaluation |
| `POST /evaluate_ops6k_kernel` | `evaluate_ops6k_kernel_impl()` | Ops-6K kernel evaluation |
| `POST /evaluate_kernels_batch` | `evaluate_kernels_batch_impl()` | Batch evaluation |
| `POST /profile_baselines` | `profile_baselines_impl()` | Baseline profiling |
| `POST /test_gpu_features` | `test_gpu_features_impl()` | GPU feature detection |
| `GET /health` | — | Liveness probe |

Global exception handler returns eval-compatible error dicts (HTTP 200 with `compiles=False`) instead of 500s.

### 6.3 Ops-6K Evaluation Path

File: `eval_service/eval_core.py:584-873` — `evaluate_ops6k_kernel_impl(payload)`

Input: `{cuda_code, task_code, warmup_iters=10, benchmark_runs=10}`

| Step | Lines | What Happens |
|------|-------|-------------|
| 1. Validate | 626-634 | Check `cuda_code` and `task_code` present, task is stateless (`_ops_task_supported()`). |
| 2. Compile | 636-678 | Write kernel to temp file. `torch.utils.cpp_extension.load()` with `-O3 -arch=sm_80` + whitelisted CU_FLAGS. Must expose `run_kernel` via PYBIND11_MODULE. |
| 3. Load reference | 680-696 | `importlib.util` loads reference Model class from `task_code`. Instantiate `Model(*get_init_inputs()).eval().cuda()`. |
| 4. 5-seed correctness | 698-730 | For seeds 42..46: generate inputs via `get_inputs()`, run both reference and candidate, `_assert_close()`. Stash first 2 outputs for anti-hack. |
| 5. Anti-hack (post-correctness) | 732-773 | `check_shapes_match()` → `check_output_not_constant()` → `check_not_passthrough()`. Imported from `openenv_env/anti_hack.py`. Failures set `correct=False`. |
| 6. Benchmark eager | 775-799 | Warmup + N timed runs with `torch.cuda.Event`. Compute `median(eager_times)`. |
| 7. Benchmark torch.compile | 801-823 | `torch.compile(ref_model)`, warmup + N timed runs. Compute `median(compile_times)`. |
| 8. Benchmark candidate | 825-850 | Warmup + N timed runs of `extension.run_kernel()`. Compute stats (mean, median, std, min, max). |
| 9. Speedups | 852-855 | `speedup_vs_orig = eager_ms / kernel_ms`, `speedup_vs_dg = compile_ms / kernel_ms`. |
| 10. Anti-hack (post-benchmark) | 858-869 | `check_not_noop(kernel_ms)` — flag if runtime < 1μs (likely no-op). |

Output: `{compiles, correct, runtime_ms, runtime_stats, baseline_eager_ms, baseline_compile_ms, speedup_vs_orig, speedup_vs_dg, verifier_msg, error}`

### 6.4 WCC Evaluation Path

File: `eval_service/eval_core.py:108-400` (approximate) — `evaluate_kernel_impl(payload)`

| Step | What Happens |
|------|-------------|
| 1. Compile | `nvcc -arch=sm_80 -O3 --shared -Xcompiler -fPIC` + whitelisted CU_FLAGS → `.so` |
| 2. Forbidden symbols | `scan_forbidden_symbols(so_path)` — `nm -D` scan for torch/triton/c10 symbols |
| 3. PAC verify | Load `.so` via ctypes, verify WCC on 5 adversarial graphs with 3 invariants (component count, edge consistency, cross-component distinctness) |
| 4. Benchmark | `cudaEvent` timing: warmup + N timed runs → median runtime |
| 5. Nsight (optional) | Lightweight `ptxas` profiling for occupancy/register stats |

---

## 7. Anti-Hack Stack

File: `openenv_env/anti_hack.py` (275 lines) — 5 checks inspired by Dr. Kernel's three failure modes.

### Static Checks

| Check | Function | Lines | What It Catches |
|-------|----------|-------|-----------------|
| Forbidden symbols | `scan_forbidden_symbols(so_path)` | 61-79 | Kernel that imports torch/triton/c10 (via `nm -D` on compiled .so) |
| CU_FLAGS whitelist | `extract_cu_flags(cuda_code)` | 36-58 | Only `--use_fast_math`, `--extra-device-vectorization`, `--rdc=true`, `--maxrregcount=16..128` |

### Runtime Checks

| Check | Function | Lines | What It Catches | Wired At |
|-------|----------|-------|-----------------|----------|
| Shape match | `check_shapes_match()` | 191-229 | Output tensor count/shape mismatch vs reference | eval_core.py:740-749 |
| Not constant | `check_output_not_constant()` | 87-133 | Decoy kernel returning hardcoded values regardless of input | eval_core.py:752-760 |
| Not passthrough | `check_not_passthrough()` | 150-188 | Kernel returning input tensor unchanged | eval_core.py:763-771 |
| Not no-op | `check_not_noop()` | 136-147 | Runtime < 1μs — kernel skips computation | eval_core.py:858-869 |

### Suite Runner

`run_anti_hack_suite(candidate_outputs, reference_outputs, inputs_list, runtime_ms)` at lines 232-274 orchestrates all 4 runtime checks in sequence. Returns `(passed, reason)`.

---

## 8. SKILL.md and Prompt Construction

### build_skill_md()

File: `openenv_env/skill_builder.py` (247 lines)

**Priority chain:**
1. Env var `KERNELFORGE_SKILL_FILE` → load from file path
2. Static file `skill_{gpu_name}.md` from project root
3. Dynamic generation from `gpu_registry.py` specs

### Generated Content

1. **GPU spec header** — A100: 108 SMs, 40MB L2, 2.0 TB/s HBM, 164KB SMEM/SM
2. **4 priority tiers:**
   - P1: Algorithmic reduction (>50% impact)
   - P2: Memory hierarchy (20-50%) — arch-specific
   - P3: Compute optimization (10-20%) — warp primitives, TF32
   - P4: Library integration — cuBLAS, cuDNN, cuSPARSE
3. **7 A100 expert patterns from doubleGraph** (`_append_a100_patterns()` lines 133-246):

| # | Pattern | Source File |
|---|---------|-------------|
| 1 | Degree-based kernel dispatch (warp-coop vs thread-per-vertex) | degree dispatch |
| 2 | Warp-level shared-memory hash tables (128-entry per-warp) | louvain_f32.cu |
| 3 | Zero-copy convergence flags (pinned memory, no cudaMemcpy per iter) | convergence |
| 4 | Bitmap frontier for BFS (direction-optimizing traversal) | bfs_direction_optimizing.cu |
| 5 | Path-halving Union-Find (non-atomic lock-free WCC) | wcc |
| 6 | Compilation flag tuning by algorithm class | build flags |
| 7 | `__launch_bounds__` for explicit occupancy control | launch bounds |

### GPU Registry

File: `openenv_env/gpu_registry.py`

| Key | A100 | H100 | B200 |
|-----|------|------|------|
| arch | sm_80 | sm_90a | sm_100a |
| SMs | 108 | 132 | 192 |
| L2 cache | 40 MB | 50 MB | 96 MB |
| SMEM/SM | 164 KB | 228 KB | 228 KB |
| HBM BW | 2.0 TB/s | 3.35 TB/s | 8.0 TB/s |

---

## 9. Dataset Architecture

### Sources

| Dataset | Source | Rows | What |
|---------|--------|------|------|
| Ops-6K | [HuggingFace](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K) | 6000 | PyTorch operator tasks from CUDA Agent |
| doubleGraph SFT | `datasets/doublegraph_sft.jsonl` | 192 | Expert A100 graph kernels |
| Combined | `datasets/combined_kernelforge.jsonl` | 224 | 192 doubleGraph + 32 ops |

### TaskPool

File: `openenv_env/task_pool.py` (174 lines)

**Fallback chain:**
1. `tasks/pool_v0.jsonl` (curated stateless subset) — **NOT YET GENERATED**
2. `datasets/combined_kernelforge.jsonl` (224 rows, filtered to supported backends)
3. Built-in tasks: 1 WCC + 1 ELU (minimal fallback)

```python
class TaskPool:
    @classmethod
    def load(cls, pool_path=None) -> TaskPool
    def sample(self, task_id=None, seed=None, backend=None) -> dict
    def get_cached_baselines(self, task_id) -> dict | None
    def cache_baselines(self, task_id, baselines) -> None
```

### Honest Status

**Only ~19 tasks are live-evaluable today** (15 ops6k + 4 WCC). The rest are either stateful (use `nn.Conv`, `nn.Linear`, etc.) or have complex `get_init_inputs()` that the current harness cannot handle. Running `python tasks/build_task_pool.py` would expand to ~50+ tasks by pulling from the full Ops-6K HuggingFace dataset.

---

## 10. Training Pipeline

### 3-Stage Architecture

| Stage | File | Type | Steps | LR | G | Max Turns |
|-------|------|------|-------|-----|---|-----------|
| 1: Warmup | `training/stage1_warmup.py` | GRPO warm-up on easy ops | 100 | 2e-6 | 2 | 3 |
| 2: SFT | `training/stage2_rft.py` | Reject-filter + SFT on doubleGraph | 3 epochs | 5e-6 | — | — |
| 3: GRPO | `training/stage3_grpo.py` | TRLOO-corrected GRPO + curriculum | 50 | 3e-6 | 2 | 3 |

**Model:** `Qwen3-Coder-30B-A3B-Instruct` (30.5B total, 3.3B active, 128 experts / 8 active)
**Training GPU:** H200 (bf16, LoRA r=16, gradient checkpointing)
**Eval GPU:** A100 80GB on CoreWeave/Northflank

### Multi-Turn Rollout

File: `training/multi_turn_rollout.py` (299 lines)

`make_multi_turn_rollout(max_turns=3)` returns a TRL-compatible rollout function.

**Flow per completion (lines 199-263):**

| Step | Line | What Happens |
|------|------|-------------|
| 1 | 200 | Generate response on H200 via `generate_rollout_completions()` |
| 2 | 208 | `extract_cuda_code(text)` — extract ```cuda blocks or raw `__global__` code |
| 3 | 220 | `_local_compile_check(code)` — nvcc syntax check on training GPU (saves ~50% eval cost) |
| 4 | 232-238 | If compiles locally: `evaluate_code_remote(code, task_row)` → A100 via dispatch |
| 5 | 239 | `_compute_reward_from_result(result)` → discrete `{-1, 1, 2, 3}` |
| 6 | 262 | Early exit at `reward >= 3.0` (beats torch.compile) |
| 7 | 265-270 | `_format_feedback(result, reward, turn)` → construct next-turn prompt with error/benchmark info |

Returns: `{prompt_ids, completion_ids, logprobs, env_reward}` — compatible with TRL GRPOTrainer.

### TRLOO Trainer

File: `training/custom_grpo_trainer.py` (94 lines)

`TRLOOGRPOTrainer(GRPOTrainer)` — drop-in replacement. Overrides `_compute_advantages()`:

```python
# Line 52: the single-line fix
unbiased = (r_tensor - mean) * (N / (N - 1.0))
```

### Curriculum

File: `training/curriculum.py` (457 lines) — 4 phases, 23 tasks

| Phase | Name | Target Reward | Example Problems |
|-------|------|---------------|-----------------|
| 0 | `single_ops` | 1.0 | vector_add, relu, softmax, matmul, gelu |
| 1 | `fusion_2op` | 2.0 | LayerNorm+ReLU, MatMul+BiasAdd, Softmax+Dropout |
| 2 | `arch_specific` | 2.0 | WCC, reduce_sum, GEMM, attention |
| 3 | `advanced` | 3.0 | LayerNorm+GELU+Linear, Flash-Attention, SpMV |

**Promotion:** >50% of last 10 rewards hit target → advance.
**Demotion:** <20% of last 10 rewards positive → regress.

### Critical Dependency Chain

```
GRPO training → multi_turn_rollout → evaluate_code_remote → dispatch_eval →
    CoreWeave HTTP POST (KERNELFORGE_EVAL_URL) OR Modal serverless
```

**If `KERNELFORGE_EVAL_URL` is not set and Modal is not configured, the entire training pipeline crashes during the first rollout.**

---

## 11. SkyDiscover Integration

### What It Is

UC Berkeley Sky Computing Lab's evolutionary search framework. Two algorithms: AdaEvolve (adaptive multi-island, [arXiv 2602.20133](https://arxiv.org/abs/2602.20133)) and EvoX (self-evolving strategies, [arXiv 2602.23413](https://arxiv.org/abs/2602.23413)).

### Our Implementation

| File | Lines | What | Status |
|------|-------|------|--------|
| `skydiscover_integration/evaluator.py` | 210 | `KernelForgeEvaluator` — cascade eval bridge | IMPLEMENTED |
| `skydiscover_integration/adaevolve.py` | 430 | Multi-island search + UCB1 scheduling | IMPLEMENTED |
| `skydiscover_integration/evox_strategies.py` | 312 | Self-evolving mutation strategies | IMPLEMENTED |
| `skydiscover_integration/initial_kernels/` | dir | 5 seed kernels from doubleGraph | EXISTS |
| `skydiscover_integration/run_evolution.sh` | script | Shell runner | EXISTS |

### Honest Assessment

**What works:**
- `KernelForgeEvaluator` chains Stage 1 (local compile) → Stage 2 (remote A100 via `dispatch_eval()`). Same backend as the RL env.
- AdaEvolve loop: UCB1 island selection, tournament selection, breakthrough broadcasting.
- EvoX strategy manager: LogWindowScorer, stagnation detection, hybrid strategy generation.

**What is LIMITED:**
- **Mutation is regex-based, not LLM-based.** `adaevolve.py:_mutate()` (lines 278-324) does simple string replacements: add `__launch_bounds__`, add `__ldg`, add comments. The docstring explicitly says: *"Override this method to wire in LLM-based mutation."*
- This means evolutionary search will NOT discover novel optimizations — only superficial code transformations.

**Missing vs real SkyDiscover:** Quality-diversity archive, NSGA-II Pareto ranking, LLM-based paradigm breakthrough, diff-based generation, checkpointing, live monitoring.

**Role:** This is a **HEDGE**, not the primary path. If GRPO doesn't converge, evolutionary search provides an alternative route. Requires LLM mutation upgrade to be useful.

---

## 12. Complete File Inventory

### Environment Layer (`openenv_env/`)

| File | Lines | What It Does |
|------|-------|-------------|
| `kernel_forge_env.py` | 273 | RL environment: `reset()` samples task, `step()` evaluates CUDA code |
| `reward.py` | 83 | Discrete reward `{-1,1,2,3}` + TRLOO post-process |
| `skill_builder.py` | 247 | Dynamic SKILL.md with 7 A100 patterns |
| `task_pool.py` | 174 | TaskPool: load, sample, cache baselines |
| `task_routing.py` | 27 | Re-exports from `training/task_support.py` (dependency inversion fix) |
| `anti_hack.py` | 275 | 5 anti-hack checks + suite runner |
| `eval_backend.py` | 56 | CoreWeave HTTP + Modal dispatch |
| `models.py` | 33 | Pydantic Action/Observation types |
| `gpu_registry.py` | — | A100/H100/B200 specs |
| `cache_pool.py` | — | LRU GPU cache |
| `server/app.py` | 15 | FastAPI OpenEnv server |
| `client.py` | 11 | HTTP client |

### Evaluation Layer (`eval_service/`)

| File | Lines | What It Does |
|------|-------|-------------|
| `eval_core.py` | 873 | 6 eval functions: kernel, ops6k, batch, baselines, GPU features |
| `app.py` | 76 | FastAPI endpoints for Northflank/CoreWeave |
| `Dockerfile` | — | GPU deployment container |

### Training Layer (`training/`)

| File | Lines | What It Does |
|------|-------|-------------|
| `stage1_warmup.py` | 166 | GRPO warm-up |
| `stage2_rft.py` | 130 | Reject-filter + SFT |
| `stage3_grpo.py` | 268 | TRLOO GRPO + curriculum |
| `grpo_train.py` | 141 | Preflight + stage launcher |
| `multi_turn_rollout.py` | 299 | TRL-compatible rollout with local compile check |
| `custom_grpo_trainer.py` | 94 | TRLOO `N/(N-1)` correction |
| `model_loader.py` | 246 | Unsloth + fallback chain |
| `dataset_loader.py` | 169 | Stage-specific dataset loading |
| `task_support.py` | 327 | Task routing, payload building, eval dispatch |
| `curriculum.py` | 457 | 4-phase curriculum, 23 tasks |
| `rft_filter.py` | 247 | Trajectory collection + filtering |

### SkyDiscover Integration (`skydiscover_integration/`)

| File | Lines | What It Does |
|------|-------|-------------|
| `evaluator.py` | 210 | Cascade eval bridge (local → remote A100) |
| `adaevolve.py` | 430 | Multi-island evolutionary search + UCB1 |
| `evox_strategies.py` | 312 | Self-evolving mutation strategies |
| `initial_kernels/` | dir | Seed kernels |
| `run_evolution.sh` | script | Runner |

### Scripts & Tasks

| File | Lines | What It Does |
|------|-------|-------------|
| `scripts/run_benchmark.py` | 302 | Benchmark with fast_p metrics |
| `scripts/compare_results.py` | 171 | Before/after comparison |
| `tasks/build_task_pool.py` | 235 | Curate Ops-6K → pool JSONL |

### Verification

| File | Lines | What It Does |
|------|-------|-------------|
| `verification/pac_verify.py` | 225 | WCC: 5 graphs, 3 invariants |
| `kernels/baseline_wcc.cu` | 145 | Baseline WCC kernel (Union-Find) |

---

## 13. Environment Variables Reference

| Variable | Default | What It Controls |
|----------|---------|-----------------|
| `KERNELFORGE_EVAL_BACKEND` | `coreweave` | Eval dispatch: `coreweave` or `modal` |
| `KERNELFORGE_EVAL_URL` | (none) | CoreWeave/Northflank eval service URL |
| `KERNELFORGE_TARGET_GPU` | `a100` | Target GPU for kernel generation |
| `KERNELFORGE_TARGET_ARCH` | `sm_80` | CUDA arch for nvcc |
| `KERNELFORGE_MAX_TURNS` | `3` | Max turns per episode |
| `KERNELFORGE_MODEL` | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Training model |
| `KERNELFORGE_USE_VLLM` | `0` | Disable vLLM for hackathon |
| `KERNELFORGE_USE_TRLOO` | `1` | Enable TRLOO correction |
| `KERNELFORGE_LOCAL_COMPILE` | `1` | Enable local nvcc fast-fail |
| `KERNELFORGE_STAGE` | `stage1` | Which stage to run |
| `KERNELFORGE_MODAL_APP` | `kernelforge-a100` | Modal app name (fallback) |
| `KERNELFORGE_SKILL_FILE` | (none) | Override SKILL.md path |
| `KERNELFORGE_DECISION_GATE` | `1` | Enable decision gate (set to 0 for hackathon) |
| `KERNELFORGE_ROLLOUT_LOG` | `outputs/rollout_metrics.jsonl` | Rollout log path |
| `KERNELFORGE_MAX_FEEDBACK_CHARS` | `1200` | Max feedback text length |

---

## 14. Honest Risk Assessment

### "Can I Run This Right Now?" Matrix

| Component | Works locally? | Works end-to-end? | Blocker |
|-----------|---------------|-------------------|---------|
| `reset()` → observation | YES (with mocks) | YES (needs TaskPool JSONL) | None |
| `step()` → reward | YES (with mocks) | **NO** | Eval backend not deployed |
| GRPO training loop | Code exists | **NO** | Eval backend + installed packages |
| Benchmark suite | Code exists | **NO** | Eval backend |
| SkyDiscover search | Code exists | **NO** | Eval backend + regex-only mutation |
| Test suite | 9/9 pass | — | None |

### Blockers by Severity

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Eval backend not deployed** | CRITICAL | Deploy `eval_service/` to CoreWeave/Northflank FIRST |
| **19 live tasks is thin** | HIGH | Run `build_task_pool.py` to expand to 50+ |
| **Nobody has run GRPO yet** | HIGH | Start with Stage 1 warmup, 10 steps, check reward distribution |
| **SkyDiscover mutation is regex-only** | MEDIUM | Connect `_mutate()` to model generation |
| **pool_v0.jsonl doesn't exist** | MEDIUM | Run `python tasks/build_task_pool.py` |
| **Decision gate module missing** | LOW | Set `KERNELFORGE_DECISION_GATE=0` |

---

## 15. Teammate Runbook

### Prerequisites

```bash
# Install dependencies
pip install trl transformers torch peft unsloth
pip install "openenv-core[core]>=0.2.1"

# Set eval backend
export KERNELFORGE_EVAL_BACKEND=coreweave
export KERNELFORGE_EVAL_URL=https://your-northflank-service.com
export KERNELFORGE_TARGET_GPU=a100
```

### Step 1: Deploy Eval Service

```bash
cd eval_service
docker build -t kernelforge-eval .
# Push to CoreWeave/Northflank with A100 GPU
# Verify: curl https://your-service/health
```

### Step 2: Build Task Pool

```bash
python tasks/build_task_pool.py
# Generates: tasks/pool_v0.jsonl (50+ stateless tasks from Ops-6K)
```

### Step 3: Run Baseline Benchmark (Before Training)

```bash
python scripts/run_benchmark.py --baselines-only --output results/before.json
```

### Step 4: Run Training

```bash
# Stage 1: Warm-up
python training/grpo_train.py --stage stage1

# Stage 2: SFT
python training/grpo_train.py --stage stage2

# Stage 3: GRPO (disable decision gate)
KERNELFORGE_DECISION_GATE=0 python training/grpo_train.py --stage stage3
```

### Step 5: Run Post-Training Benchmark

```bash
python scripts/run_benchmark.py --model outputs/kernelforge-stage3 --output results/after.json
```

### Step 6: Compare

```bash
python scripts/compare_results.py results/before.json results/after.json
```

---

## 16. Benchmark Metrics (KernelBench-Compatible)

| Metric | What It Measures |
|--------|-----------------|
| `fast_0` | Correctness rate (compiled + correct) |
| `fast_1.05` | Correct AND >5% faster than eager PyTorch |
| `fast_compile` | Correct AND >5% faster than torch.compile |
| `geomean_vs_eager` | Geometric mean speedup over eager baseline |
| `geomean_vs_compile` | Geometric mean speedup over torch.compile |

---

## 17. Sources

- [CUDA Agent (arXiv 2602.24286)](https://arxiv.org/abs/2602.24286) — Ops-6K, milestone reward, SKILLS.md
- [CUDA-Agent-Ops-6K (HuggingFace)](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K) — 6000 training tasks
- [Dr. Kernel (arXiv 2602.05885)](https://arxiv.org/abs/2602.05885) — TRLOO, KernelGYM, anti-hack
- [KernelBench (arXiv 2502.10517)](https://arxiv.org/abs/2502.10517) — fast_p metric, 250 tasks
- [doubleGraph (GitHub)](https://github.com/double-ai/doubleGraph) — A100 kernel patterns, 192 variants
- [WarpSpeed (doubleAI)](https://www.doubleai.com/research/doubleais-warpspeed-surpassing-expert-written-kernels-at-scale) — 3.6x over cuGraph
- [OpenEnv (GitHub)](https://github.com/meta-pytorch/OpenEnv) — RL environment framework
- [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv) — TRL + OpenEnv
- [AdaEvolve (arXiv 2602.20133)](https://arxiv.org/abs/2602.20133) — Adaptive multi-island search
- [EvoX (arXiv 2602.23413)](https://arxiv.org/abs/2602.23413) — Self-evolving strategies
- [SkyDiscover (GitHub)](https://github.com/skydiscover-ai/skydiscover) — Search framework
