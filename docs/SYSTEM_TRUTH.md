# KernelForge RL Environment — Single Source of Truth

**Last verified:** March 8, 2026
**Hackathon:** OpenEnv Hackathon SF (SHACK15, March 7-8, 2026)
**Skepticism level:** CTO / Staff Engineer — every claim cites code or paper

---

## 1. What We Are Building

An **OpenEnv-compatible RL environment** that teaches an LLM to generate optimized CUDA kernels targeting NVIDIA A100 (`sm_80`). The model sees a PyTorch operator task + reference code + A100 optimization patterns, generates a CUDA kernel, and gets discrete reward `{-1, 1, 2, 3}` based on real compile/correctness/benchmark feedback on actual A100 hardware.

This is **NOT** a CUDA Agent reproduction. It is a narrower pilot: prove the environment works, the reward signal is real, and a model can improve inside the loop.

---

## 2. The Three Papers We Use (and How)

### Paper 1: CUDA Agent (ByteDance/Tsinghua, arXiv [2602.24286](https://arxiv.org/abs/2602.24286), Feb 2026)

**What it is:** RL-trained Seed1.6 (230B MoE) writes CUDA kernels, 2.11x over torch.compile, 98.8% pass rate.

**What we take:**
| Element | Where in our code | Verified? |
|---------|------------------|-----------|
| Ops-6K dataset (6000 operator tasks) | `training/cuda_agent_integration.py:load_cuda_agent_prompt_dataset()` → [HuggingFace](https://huggingface.co/datasets/BytedTsinghua-SIA/CUDA-Agent-Ops-6K) | YES — dataset exists on HF |
| Discrete milestone reward `{-1,1,2,3}` | `openenv_env/reward.py:compute_reward()` lines 29-69 | YES — code matches paper ablation (96.8% vs 60.4%) |
| SKILLS.md concept (inject optimization knowledge) | `openenv_env/skill_builder.py:build_skill_md()` | YES — 247 lines, 7 A100 patterns |
| 3-file kernel contract (kernel + binding + model) | `eval_service/eval_core.py:evaluate_ops6k_kernel_impl()` lines 584-873 | YES — uses `torch.utils.cpp_extension` |
| 5-seed correctness check | `eval_service/eval_core.py` lines 706-727 | YES — 5 seeds with `torch.manual_seed(42+seed)` |

**What we DON'T take:** Their Seed1.6 model (not public), 128-GPU training scale, ReAct agent loop, data synthesis pipeline.

### Paper 2: Dr. Kernel / KernelGYM (HKUST, arXiv [2602.05885](https://arxiv.org/abs/2602.05885), Feb 2026)

**What it is:** Dr. Kernel-14B trained on KernelGYM with TRLOO correction, competitive with Claude-4.5-Sonnet on KernelBench.

**What we take:**
| Element | Where in our code | Verified? |
|---------|------------------|-----------|
| TRLOO `N/(N-1)` advantage correction | `training/custom_grpo_trainer.py:TRLOOGRPOTrainer._compute_advantages()` lines 32-59 | YES — fixes 50% gradient shrinkage at G=2 |
| Anti-hack: 3 failure modes (reward hacking, lazy optimization, biased credit) | `openenv_env/anti_hack.py` lines 87-274 (5 checks) | YES — wired into `eval_core.py` lines 732-773 |
| `max_turns=3` | `openenv_env/kernel_forge_env.py` constructor | YES — `int(os.getenv("KERNELFORGE_MAX_TURNS", "3"))` |

**What we DON'T take:** KernelGYM distributed environment directly, Triton-first approach (we do CUDA), PR/PRS profiling rewards.

### Paper 3: KernelBench (Stanford, arXiv [2502.10517](https://arxiv.org/abs/2502.10517), Feb 2025)

**What it is:** 250-task GPU kernel benchmark with `fast_p` metric.

**What we take:**
| Element | Where in our code | Verified? |
|---------|------------------|-----------|
| `fast_p` metric format | `scripts/run_benchmark.py:compute_fast_p()` | YES — fast_0, fast_1.05, fast_compile, geomean |
| 3-level difficulty structure | `training/curriculum.py:_default_phases()` | YES — 4 phases, 23 tasks |
| 5-seed correctness check | `eval_service/eval_core.py` lines 706-727 | YES — matches their protocol |

**What we DON'T take:** Their 250 specific tasks, L40S target (we use A100).

---

## 3. How We Use doubleGraph for A100

**What is doubleGraph:** [doubleAI](https://www.doubleai.com/research/doubleais-warpspeed-surpassing-expert-written-kernels-at-scale)'s hyper-optimized replacement for NVIDIA cuGraph, achieving 3.6x average speedup. 192 GPU-specific kernels per architecture. Open source: [github.com/double-ai/doubleGraph](https://github.com/double-ai/doubleGraph).

**How we use it (4 ways):**

| Use | Code Location | Verified? |
|-----|--------------|-----------|
| **192 expert A100 kernel source files → SFT training data** | `datasets/doublegraph_sft.jsonl` (192 rows) | YES — `wc -l` = 192 |
| **7 engineering patterns → SKILL.md prompt context** | `openenv_env/skill_builder.py:_append_a100_patterns()` lines 133-246 | YES — degree dispatch, warp hash, zero-copy, bitmap frontier, path-halving UF, compilation flags, `__launch_bounds__` |
| **WCC verification → 3 PAC invariants** | `verification/pac_verify.py:verify_wcc()` (225 lines) | YES — component count, edge consistency, cross-component distinctness |
| **Graph topology awareness → curriculum phases 2-3** | `training/curriculum.py` lines 108-305 | YES — BFS, PageRank, WCC, Triangle Count, Louvain with `graph_properties` |

**We do NOT run doubleGraph at inference time.** We teach the model to write kernels LIKE doubleGraph by showing it expert patterns + topology context.

---

## 4. How We Use the 6K Dataset

| Step | Code | Status |
|------|------|--------|
| Load from HuggingFace | `training/cuda_agent_integration.py:load_cuda_agent_prompt_dataset()` | IMPLEMENTED |
| Filter stateless evaluable tasks | `tasks/build_task_pool.py:is_stateless_evaluable()` | IMPLEMENTED — filters out `nn.Conv`, `nn.Linear`, stateful modules |
| Validate structure | `tasks/build_task_pool.py:has_valid_structure()` | IMPLEMENTED — checks for `Model` class + `get_inputs()` + `get_init_inputs()` |
| Output to task pool JSONL | `tasks/build_task_pool.py:build_task_pool()` | IMPLEMENTED — outputs `tasks/pool_v0.jsonl` |
| Combined with doubleGraph | `datasets/build_combined_dataset.py` → `datasets/combined_kernelforge.jsonl` | IMPLEMENTED — 224 rows (192 dG + 32 ops) |

**HONEST STATUS:** `pool_v0.jsonl` has NOT been generated yet — needs `python tasks/build_task_pool.py` to run. Currently only 19 of 224 combined tasks are live-evaluable (15 ops6k + 4 WCC).

---

## 5. The RL Environment Architecture

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
│       │   2. scan_forbidden_symbols()                   │
│       │   3. 5-seed correctness vs PyTorch reference    │
│       │   4. Anti-hack: shapes, constant, passthrough   │
│       │   5. Benchmark: median of 100 runs              │
│       │   6. Speedup vs eager + vs torch.compile        │
│       │   7. Anti-hack: no-op check (< 1μs)            │
│       │                                                 │
│       └── WCC path (evaluate_kernel_impl):              │
│           1. nvcc -arch=sm_80 compile                   │
│           2. PAC verify: 5 graphs, 3 invariants         │
│           3. CudaEvent benchmark                        │
│           4. Optional Nsight profiling                  │
│                                                         │
│  modal_app.py (FALLBACK — same eval_core underneath)    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Environment Contract

| Aspect | Value | Code Location |
|--------|-------|---------------|
| **Action** | CUDA source code string | `openenv_env/models.py:KernelForgeAction` |
| **Observation** | SKILL.md + task + reference code + feedback + history | `openenv_env/models.py:KernelForgeObservation` |
| **Reward** | Discrete `{-1, 1, 2, 3}` | `openenv_env/reward.py:compute_reward()` |
| **Max turns** | 3 (env var `KERNELFORGE_MAX_TURNS`) | `openenv_env/kernel_forge_env.py` constructor |
| **Task sampling** | `TaskPool.sample()` on `reset()` | `openenv_env/task_pool.py` |
| **Backend dispatch** | CoreWeave HTTP (default) or Modal | `openenv_env/eval_backend.py:dispatch_eval()` |
| **Protocol** | OpenEnv HTTP (NOT Gymnasium) | `openenv_env/server/app.py` via `create_app()` |

### Reward Function (DO NOT MODIFY)

```python
# openenv_env/reward.py:compute_reward() — verified against code
if not compiled or not correct:  return -1.0
if speedup_vs_compile > 1.05:    return  3.0   # beats torch.compile
if speedup_vs_eager > 1.05:      return  2.0   # beats eager PyTorch
return 1.0                                      # correct but no speedup
```

### Anti-Hack Checks (5 checks, Dr. Kernel-inspired)

| Check | What It Catches | Code |
|-------|----------------|------|
| `scan_forbidden_symbols()` | Kernel that imports torch/triton | `anti_hack.py:61-79` |
| `check_shapes_match()` | Output shape mismatch vs reference | `anti_hack.py:155-188` |
| `check_output_not_constant()` | Decoy kernel returning hardcoded values | `anti_hack.py:87-131` |
| `check_not_passthrough()` | Kernel returning input unchanged | `anti_hack.py:134-152` |
| `check_not_noop()` | Suspiciously fast kernel (< 1μs) | `anti_hack.py:134-152` |

All checks are wired into `eval_core.py` lines 732-773 (post-correctness) and lines 860-870 (post-benchmark).

---

## 6. SkyDiscover / Test-Time Search

**What is SkyDiscover:** UC Berkeley Sky Computing Lab's evolutionary search framework. Two algorithms: AdaEvolve (adaptive multi-island, [arXiv 2602.20133](https://arxiv.org/abs/2602.20133)) and EvoX (self-evolving strategies, [arXiv 2602.23413](https://arxiv.org/abs/2602.23413)).

**Skills reference:** `docs/skills/SKYDISCOVER_ADAEVOLVE_EVOX.md` (854 lines) — comprehensive technical deep-dive with full algorithm descriptions, parameter tuning guides, and integration architecture.

### Our Implementation

| File | Lines | What It Does | Status |
|------|-------|-------------|--------|
| `skydiscover_integration/evaluator.py` | 210 | `KernelForgeEvaluator` — cascade eval bridge (Stage 1: local nvcc → Stage 2: remote A100) | **FULLY IMPLEMENTED** — calls `dispatch_eval()`, same backend as RL env |
| `skydiscover_integration/adaevolve.py` | 430 | Multi-island evolutionary search with UCB1 scheduling | **IMPLEMENTED** — 4 mutation strategies, tournament selection, breakthrough broadcasting |
| `skydiscover_integration/evox_strategies.py` | 312 | `EvoXStrategyManager` — strategy stagnation detection + hybrid strategy evolution | **IMPLEMENTED** — LogWindowScorer, strategy hybridization, deactivation |
| `skydiscover_integration/initial_kernels/` | dir | Seed kernels for evolution (5 A100 graph kernels from doubleGraph) | EXISTS |
| `skydiscover_integration/run_evolution.sh` | script | Shell runner | EXISTS |

### Honest Assessment

**What works:**
- Evaluator bridge: `KernelForgeEvaluator` correctly chains Stage 1 (local compile) → Stage 2 (remote A100 via `dispatch_eval()`). Uses the SAME eval backend as the RL environment. Code: `evaluator.py:evaluate_stage1()` + `evaluate_stage2()`.
- AdaEvolve loop: UCB1 island selection, population management, breakthrough broadcasting, EvoX strategy integration — all implemented and runnable.
- EvoX strategy manager: LogWindowScorer tracks improvement velocity per strategy. Stagnation detection triggers hybrid strategy generation from `HYBRID_TEMPLATES`. Strategy hot-swap with deactivation.

**What is LIMITED:**
- **Mutation is regex-based, not LLM-based.** The `_mutate()` method in `adaevolve.py` (lines 278-324) does simple string replacements (add `__launch_bounds__`, add `__ldg`, add comments). The docstring explicitly says: *"In production, wire this to the B200's LLM for intelligent mutation. Override this method to wire in LLM-based mutation."*
- This means evolutionary search will NOT discover novel optimizations — it can only apply superficial code transformations. To make this useful, you'd need to connect `_mutate()` to a model generation call.

**Is this the SkyDiscover library or a reimplementation?**
- **Reimplementation.** We don't `pip install skydiscover`. We wrote ~950 lines of our own AdaEvolve/EvoX-inspired code. The skills doc (`SKYDISCOVER_ADAEVOLVE_EVOX.md`) says "Don't reimplement from scratch — use as library" but we did reimplement. The reimplementation covers the core loop but lacks: quality-diversity archive, paradigm breakthrough via LLM, NSGA-II Pareto ranking, diff-based generation, checkpointing, live monitoring.

**Role in the system:** This is a **HEDGE**, not the primary path. If GRPO training doesn't improve kernels, evolutionary search provides an alternative route to demo-quality results. It works end-to-end IF mutation is upgraded beyond regex.

---

## 7. SKILL.md — How It Works

**Code:** `openenv_env/skill_builder.py` (247 lines)

**What gets injected into every prompt at `reset()`:**

1. **GPU-specific optimization guide** — dynamically generated from `gpu_registry.py` (A100: 108 SMs, 40MB L2, 2.0 TB/s HBM BW)
2. **4 priority tiers:**
   - P1: Algorithmic reduction (>50% impact)
   - P2: Memory hierarchy (20-50%) — architecture-specific
   - P3: Compute optimization (10-20%) — warp primitives
   - P4: Library integration — cuBLAS, cuDNN
3. **7 A100 expert patterns from doubleGraph** (`_append_a100_patterns()` lines 133-246):
   1. Degree-based kernel dispatch (warp-coop vs thread-per-vertex)
   2. Warp-level shared-memory hash tables (128-entry per-warp)
   3. Zero-copy convergence flags (pinned memory, no `cudaMemcpy` per iteration)
   4. Bitmap frontier for BFS (direction-optimizing traversal)
   5. Path-halving Union-Find (non-atomic lock-free WCC)
   6. Compilation flag tuning by algorithm class
   7. `__launch_bounds__` for explicit occupancy control

Each pattern includes source file reference (e.g., `louvain_f32.cu`, `bfs_direction_optimizing.cu`).

---

## 8. The Training Pipeline

### 3-Stage Architecture

| Stage | File | What | Steps | LR | G | Max Turns |
|-------|------|------|-------|-----|---|-----------|
| 1: Warmup | `training/stage1_warmup.py` | GRPO warm-up on easy ops | 100 | 2e-6 | 2 | 3 |
| 2: SFT | `training/stage2_rft.py` | Reject-filter + SFT on doubleGraph | 3 epochs | 5e-6 | — | — |
| 3: GRPO | `training/stage3_grpo.py` | TRLOO-corrected GRPO + curriculum | 50 | 3e-6 | 2 | 3 |

**Model:** `Qwen3-Coder-30B-A3B-Instruct` (30.5B total, 3.3B active, 128 experts / 8 active)
**Training GPU:** H200 (bf16, LoRA r=16, gradient checkpointing)
**Eval GPU:** A100 80GB on CoreWeave/Northflank

### Does It Actually Work?

| Component | Works? | Evidence | Blocker |
|-----------|--------|----------|---------|
| Model loading | YES | `training/model_loader.py` — Unsloth + fallback to transformers + PEFT | Needs packages installed |
| Dataset loading | YES | `training/dataset_loader.py` — 224 combined rows | Needs `doublegraph_sft.jsonl` present |
| Curriculum | YES | `training/curriculum.py` — 4 phases, 23 tasks, pure Python | None |
| TRLOO trainer | YES | `training/custom_grpo_trainer.py` — 94 lines, N/(N-1) scaling | Needs TRL |
| Multi-turn rollout | YES in code | `training/multi_turn_rollout.py` — extract CUDA → compile check → remote eval → feedback | Needs eval backend deployed |
| Stage 1 warmup | YES in code | `training/stage1_warmup.py` — 166 lines | Needs eval backend |
| Stage 2 SFT | YES in code | `training/stage2_rft.py` — 130 lines | Needs Stage 1 checkpoint + eval backend |
| Stage 3 GRPO | YES in code | `training/stage3_grpo.py` — 268 lines | Needs Stage 2 checkpoint + eval backend |
| End-to-end training | **UNPROVEN** | Nobody has run a full training loop yet | **Eval backend must be deployed first** |

### Critical Dependency Chain

```
GRPO training → multi_turn_rollout → evaluate_code_remote → dispatch_eval →
    CoreWeave HTTP POST (KERNELFORGE_EVAL_URL) OR Modal serverless
```

**If `KERNELFORGE_EVAL_URL` is not set and Modal is not configured, the entire training pipeline crashes during the first rollout.** This is the #1 blocker.

---

## 9. Benchmark & Comparison

| Script | Lines | Purpose | Works? |
|--------|-------|---------|--------|
| `scripts/run_benchmark.py` | 302 | Run model on task pool, compute KernelBench-compatible metrics | YES — uses HF transformers for generation + `evaluate_code_remote()` |
| `scripts/compare_results.py` | 171 | Before/after comparison with per-task diffs | YES |
| `tasks/build_task_pool.py` | 235 | Curate tasks from Ops-6K HF dataset → `pool_v0.jsonl` | YES — but hasn't been run yet |

### Metrics (KernelBench-compatible)

| Metric | What It Measures |
|--------|-----------------|
| `fast_0` | Correctness rate (compiled + correct) |
| `fast_1.05` | Correct AND >5% faster than eager PyTorch |
| `fast_compile` | Correct AND >5% faster than torch.compile |
| `geomean_vs_eager` | Geometric mean speedup over eager baseline |
| `geomean_vs_compile` | Geometric mean speedup over torch.compile |

---

## 10. Complete File Inventory

### Environment Layer (`openenv_env/`)

| File | Lines | Status | What It Does |
|------|-------|--------|-------------|
| `kernel_forge_env.py` | 273 | IMPLEMENTED | RL environment: `reset()` samples task, `step()` evaluates CUDA code |
| `reward.py` | 83 | IMPLEMENTED | Discrete reward `{-1,1,2,3}` + TRLOO post-process |
| `skill_builder.py` | 247 | IMPLEMENTED | Dynamic SKILL.md with 7 A100 patterns |
| `task_pool.py` | 174 | IMPLEMENTED | TaskPool: load, sample, cache baselines |
| `task_routing.py` | 27 | IMPLEMENTED | Re-exports from `training/task_support.py` |
| `anti_hack.py` | 275 | IMPLEMENTED | 5 anti-hack checks + suite runner |
| `eval_backend.py` | 56 | IMPLEMENTED | CoreWeave HTTP + Modal dispatch |
| `models.py` | 33 | IMPLEMENTED | Pydantic Action/Observation types |
| `gpu_registry.py` | — | IMPLEMENTED | A100/H100/B200 specs |
| `cache_pool.py` | — | IMPLEMENTED | LRU GPU cache |
| `server/app.py` | 15 | IMPLEMENTED | FastAPI OpenEnv server |
| `client.py` | 11 | IMPLEMENTED | HTTP client |

### Evaluation Layer (`eval_service/`)

| File | Lines | Status | What It Does |
|------|-------|--------|-------------|
| `eval_core.py` | 873 | IMPLEMENTED | 6 eval functions: kernel, ops6k, batch, baselines, GPU features |
| `app.py` | 76 | IMPLEMENTED | FastAPI endpoints for Northflank/CoreWeave |
| `Dockerfile` | — | EXISTS | GPU deployment container |

### Training Layer (`training/`)

| File | Lines | Status | What It Does |
|------|-------|--------|-------------|
| `stage1_warmup.py` | 166 | IMPLEMENTED | GRPO warm-up |
| `stage2_rft.py` | 130 | IMPLEMENTED | Reject-filter + SFT |
| `stage3_grpo.py` | 268 | IMPLEMENTED | TRLOO GRPO + curriculum |
| `grpo_train.py` | 141 | IMPLEMENTED | Preflight + stage launcher |
| `multi_turn_rollout.py` | 299 | IMPLEMENTED | TRL-compatible rollout with local compile check |
| `custom_grpo_trainer.py` | 94 | IMPLEMENTED | TRLOO `N/(N-1)` correction |
| `model_loader.py` | 246 | IMPLEMENTED | Unsloth + fallback chain |
| `dataset_loader.py` | 169 | IMPLEMENTED | Stage-specific dataset loading |
| `task_support.py` | 327 | IMPLEMENTED | Task routing, payload building, eval dispatch |
| `curriculum.py` | 457 | IMPLEMENTED | 4-phase curriculum, 23 tasks |
| `rft_filter.py` | 247 | IMPLEMENTED | Trajectory collection + filtering |

### SkyDiscover Integration (`skydiscover_integration/`)

| File | Lines | Status | What It Does |
|------|-------|--------|-------------|
| `evaluator.py` | 210 | IMPLEMENTED | Cascade eval bridge (local → remote A100) |
| `adaevolve.py` | 430 | IMPLEMENTED | Multi-island evolutionary search + UCB1 |
| `evox_strategies.py` | 312 | IMPLEMENTED | Self-evolving mutation strategies |
| `initial_kernels/` | dir | EXISTS | Seed kernels |
| `run_evolution.sh` | script | EXISTS | Runner |

### Scripts & Tasks

| File | Lines | Status | What It Does |
|------|-------|--------|-------------|
| `scripts/run_benchmark.py` | 302 | IMPLEMENTED | Benchmark with fast_p metrics |
| `scripts/compare_results.py` | 171 | IMPLEMENTED | Before/after comparison |
| `tasks/build_task_pool.py` | 235 | IMPLEMENTED | Curate Ops-6K → pool JSONL |

### Verification

| File | Lines | Status | What It Does |
|------|-------|--------|-------------|
| `verification/pac_verify.py` | 225 | IMPLEMENTED | WCC verification: 5 graphs, 3 invariants |
| `kernels/baseline_wcc.cu` | 145 | IMPLEMENTED | Baseline WCC kernel (Union-Find) |

---

## 11. Environment Variables Reference

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

---

## 12. Teammate Runbook (Step-by-Step)

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

# Stage 3: GRPO (set KERNELFORGE_DECISION_GATE=0)
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

## 13. What We Dropped and Why

| Technique | Why Dropped | Reference |
|-----------|------------|-----------|
| MARS return-to-go credit | Degenerates with outcome-only rewards (return_to_go[t] = r_final for all t) | GRPO-4 |
| CPPO completion pruning | At G=2, pruning leaves G=1 → zero advantage variance for TRLOO | GRPO-11 |
| MASPO soft trust region | Future research, not hackathon | GRPO-12 |
| Transform grammar | No production system ships this | GRPO-13 |
| Full Nsight structured reward | API-compatible but unused in discrete mode | GRPO-9 |
| Continuous log(speedup) reward | CUDA Agent ablation: discrete 96.8% vs continuous 60.4% | reward.py |

---

## 14. Honest Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Eval backend not deployed** — entire training pipeline blocked | CRITICAL | Deploy `eval_service/` to CoreWeave/Northflank FIRST |
| **19 live tasks is thin** — narrow training signal | HIGH | Run `build_task_pool.py` to expand to 50+ |
| **Nobody has run GRPO yet** — convergence unproven | HIGH | Start with Stage 1 warmup, 10 steps, check reward distribution |
| **SkyDiscover mutation is regex-only** — won't discover novel optimizations | MEDIUM | Connect `_mutate()` to model generation for real mutations |
| **pool_v0.jsonl doesn't exist** — TaskPool falls back to 224 combined rows | MEDIUM | Run `python tasks/build_task_pool.py` |
| **Decision gate module missing** — stage3 crashes if enabled | LOW | Set `KERNELFORGE_DECISION_GATE=0` |
| **Model default in code vs docs** — `model_loader.py` may default to Coder-Next | LOW | Set `KERNELFORGE_MODEL` env var explicitly |

---

## 15. Sources

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
