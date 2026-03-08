<chatName="CUDA-Agent Complete SKILLS.md Reference Document"/>

# CUDA-Agent → Complete SKILLS.md Reference

Here's the entire CUDA-Agent system distilled into a single, comprehensive SKILLS.md document you can use as a blueprint for replicating the RL environment:

---

```markdown
# SKILLS.md — CUDA Kernel RL Agent Environment
## Complete System Reference for Replication

> **Source:** CUDA-Agent (ByteDance Seed × Tsinghua AIR, March 2026)
> **Paper:** arxiv.org/abs/2602.24286
> **What this is:** A complete blueprint for building an RL-trained LLM agent that writes
> optimized CUDA kernels, beating torch.compile by 1.5–2.8× across difficulty levels.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Training Data Pipeline](#2-training-data-pipeline)
3. [Agent Environment Architecture](#3-agent-environment-architecture)
4. [Self-Registering Kernel System](#4-self-registering-kernel-system)
5. [Three-File Kernel Contract](#5-three-file-kernel-contract)
6. [Compilation System](#6-compilation-system)
7. [Verification System (Anti-Cheating)](#7-verification-system-anti-cheating)
8. [Profiling System](#8-profiling-system)
9. [Reward Function](#9-reward-function)
10. [Agent Loop & Episode Structure](#10-agent-loop--episode-structure)
11. [RL Training Pipeline (4 Stages)](#11-rl-training-pipeline-4-stages)
12. [Agent System Prompt (SKILL.md)](#12-agent-system-prompt-skillmd)
13. [Anti-Reward-Hacking Stack](#13-anti-reward-hacking-stack)
14. [Optimization Patterns the Agent Learns](#14-optimization-patterns-the-agent-learns)
15. [What to Build (Implementation Checklist)](#15-what-to-build-implementation-checklist)

---

## 1. System Overview

### What CUDA-Agent Is

An RL-trained LLM (Seed 1.6, 230B MoE) that operates inside a sandboxed agent loop:

```
write CUDA code → compile → verify correctness → profile performance → iterate
```

The agent receives a PyTorch model and must produce an equivalent CUDA extension that
runs ≥5% faster than `torch.compile`. Reward signals from this loop train the model via PPO.

### Three Pillars

| Pillar | Purpose |
|--------|---------|
| **Training Data** (CUDA-Agent-Ops-6K) | 6,000 synthesized operator tasks for RL training |
| **Agent Environment** (agent_workdir/) | Sandboxed workspace with frozen infrastructure + agent-writable surfaces |
| **RL Training Pipeline** (4 stages) | Multi-stage warmup preventing training collapse |

### Results

| Metric | Level 1 (Simple) | Level 2 (Fused Ops) | Level 3 (Full Models) |
|--------|-------------------|---------------------|----------------------|
| Pass Rate | 100% | 100% | 94% |
| Faster than torch.compile | 97% | **100%** | 90% |
| Geomean speedup vs compile | 1.87× | **2.80×** | 1.52× |

---

## 2. Training Data Pipeline

### Problem: Not enough CUDA optimization tasks exist for RL training

### Solution: 3-stage synthesis pipeline producing 6,000 filtered samples

```
┌─────────────────────────┐
│  Stage 1: Seed Crawling │
│  Mine nn.Module ops     │
│  from torch +           │
│  transformers libraries │
│                         │
│  Each has:              │
│  • __init__             │
│  • forward              │
│  • get_inputs()         │
│  • get_init_inputs()    │
└──────────┬──────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Stage 2: Combinatorial      │
│  Synthesis                   │
│                              │
│  LLM samples 2-5 operators  │
│  and composes them           │
│  sequentially into fused     │
│  tasks                       │
│                              │
│  Key insight: fused tasks    │
│  ≠ sum of individually       │
│  optimized parts (shared     │
│  memory, occupancy coupling, │
│  data layout dependencies)   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Stage 3: Filtering          │
│                              │
│  4 criteria:                 │
│  1. Must execute in eager    │
│     AND torch.compile        │
│  2. No stochastic ops        │
│     (dropout, etc.)          │
│  3. Runtime 1ms–100ms        │
│  4. AST similarity < 0.9    │
│     vs KernelBench           │
│     (contamination check)    │
└──────────────────────────────┘
```

### Dataset Composition

| Category | Proportion |
|----------|-----------|
| torch operators ×1 | 3.4% |
| torch operators ×2 | 83.8% |
| torch operators ×3 | 7.6% |
| torch operators ×4 | 2.8% |
| torch operators ×5 | 1.2% |
| transformers modules | 1.2% |

### Task Format

Each task is a self-contained Python file:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return self.alpha * a + b  # ← The computation to optimize

def get_inputs():
    a = torch.randn(1, 128)
    b = torch.randn(1, 128)
    return [a, b]

def get_init_inputs():
    return [2.0]  # ← Constructor arguments
```

### Why Composition Matters (from paper §3.1)

> "The combined problem is often not equivalent to trivially optimizing each operator
> in isolation and then chaining them. Fusion reshapes the optimization landscape by
> avoiding intermediate global-memory materialization, coupling stages through shared
> register/SMEM/occupancy constraints, and requiring a unified parallel mapping and
> data layout that may favor downstream consumption."

### To Replicate

1. Crawl `torch.nn` and `transformers` for `nn.Module` subclasses
2. Use an LLM to compose 2-5 operators into fused tasks
3. Filter: must run in eager + compile, deterministic, 1-100ms runtime
4. AST similarity check vs your evaluation set (threshold 0.9)

---

## 3. Agent Environment Architecture

### Directory Structure

```
agent_workdir/
├── SKILL.md              ← System prompt (rules + workflow)
├── model.py              ← FROZEN: PyTorch reference model
├── model_new.py          ← AGENT WRITES: optimized model
├── binding.cpp           ← FROZEN: pybind11 module entry
├── binding_registry.h    ← FROZEN: auto-registration macros
├── kernels/              ← AGENT WRITES: .cu + _binding.cpp pairs
│   ├── *.cu              ← Pure CUDA kernels
│   └── *_binding.cpp     ← PyTorch tensor wrappers
├── utils/                ← FROZEN: build + eval infrastructure
│   ├── __init__.py
│   ├── compile.py        ← torch cpp_extension build
│   ├── compile.sh        ← build entrypoint with timing
│   ├── verification.py   ← correctness oracle
│   └── profiling.py      ← performance measurement
└── build/                ← Generated: intermediate .o files
```

### Frozen vs Agent-Writable

| Category | Files | Rules |
|----------|-------|-------|
| **FROZEN** (read-only) | `binding.cpp`, `binding_registry.h`, `model.py`, `utils/*`, `SKILL.md` | Agent cannot modify. Protected by file permissions. |
| **AGENT-WRITABLE** | `model_new.py`, `kernels/*.cu`, `kernels/*_binding.cpp` | Agent creates/modifies freely. |
| **BUILD ARTIFACTS** | `build/`, `cuda_extension.so` | Auto-generated by compilation. |

### Tools Available to Agent

| Tool | Purpose |
|------|---------|
| **Bash** | Shell commands (compile, run verification/profiling) |
| **Read / Write** | File access (read-before-write enforced) |
| **Edit / MultiEdit** | Deterministic string-level code modifications |
| **Glob** | File discovery (`**/*.py`, `kernels/*.cu`) |
| **Grep** | Code search (ripgrep-based) |

### Sandbox Environment

- Docker-based terminal sandbox for CPU tasks (compilation)
- GPU sandbox pool (128 NVIDIA H20 GPUs) for verification/profiling
- Process-level isolation — each evaluation on dedicated GPU
- Exclusive resource allocation — no inter-process interference

---

## 4. Self-Registering Kernel System

### The Problem

If the agent had to modify a central `binding.cpp` every time it adds a kernel,
you get coordination complexity, merge conflicts, and potential corruption.

### The Solution

Kernels register themselves via C++ static initialization. The agent **never touches**
`binding.cpp`.

### `binding_registry.h` — The Registration Infrastructure

```cpp
#pragma once

// Fast-compiling binding registry header
// Avoids heavy torch/extension.h for faster compilation

#include <vector>
#include <functional>
#include <string>

// Only include minimal pybind11 headers
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Singleton registry — collects {name, lambda} pairs
class BindingRegistry {
public:
    using BindingFunction = std::function<void(pybind11::module&)>;
    
    static BindingRegistry& getInstance() {
        static BindingRegistry instance;
        return instance;
    }
    
    void registerBinding(const std::string& name, BindingFunction func) {
        bindings_.push_back({name, func});
    }
    
    void applyBindings(pybind11::module& m) {
        for (auto& [name, func] : bindings_) {
            func(m);
        }
    }
    
private:
    std::vector<std::pair<std::string, BindingFunction>> bindings_;
    BindingRegistry() = default;
};

// Static object that registers on construction (before main())
class BindingRegistrar {
public:
    BindingRegistrar(const std::string& name, BindingRegistry::BindingFunction func) {
        BindingRegistry::getInstance().registerBinding(name, func);
    }
};

// The macro the agent uses in every _binding.cpp file
#define REGISTER_BINDING(name, func) \
    static BindingRegistrar _registrar_##name(#name, [](pybind11::module& m) { func(m); })
```

### `binding.cpp` — Fixed Module Entry Point (never modified)

```cpp
#include <pybind11/pybind11.h>
#include "binding_registry.h"

PYBIND11_MODULE(cuda_extension, m) {
    BindingRegistry::getInstance().applyBindings(m);
}
```

### How It Works

```
Program loads → static BindingRegistrar objects construct →
each calls BindingRegistry::registerBinding() →
pybind11 init calls applyBindings() → all kernels exposed to Python
```

### Why This Matters for RL

- Agent adds kernels by creating file pairs — no coordination needed
- Agent removes kernels by deleting file pairs — no cleanup in other files
- Each kernel is 100% self-contained
- No merge conflicts possible

---

## 5. Three-File Kernel Contract

Every kernel the agent produces = exactly **3 files**:

| File | Purpose | Dependencies |
|------|---------|-------------|
| `kernels/foo.cu` | Pure CUDA kernel + C launcher | `<cuda_runtime.h>` only |
| `kernels/foo_binding.cpp` | PyTorch tensor → raw pointer conversion | `torch/types.h`, pybind11, `binding_registry.h` |
| `model_new.py` | Python model calling `cuda_extension.foo_forward()` | `torch`, `cuda_extension` |

### File 1: `kernels/axpby.cu` — Pure CUDA, Zero PyTorch Dependency

```cuda
#include <cuda_runtime.h>

template<int THREADS>
__global__ void axpby_kernel(float* out, const float* a, const float* b,
                              float alpha, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < size; i += stride) {
        out[i] = alpha * a[i] + b[i];
    }
}

// C-interface launcher — extern "C" prevents name mangling
extern "C" void axpby_launcher(
    float* out, const float* a, const float* b,
    float alpha, int size, int config, cudaStream_t stream
) {
    if (size <= 0) return;
    switch (config) {
        case 1: {
            int threads = 128;
            int blocks = (size + threads - 1) / threads;
            axpby_kernel<128><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
        case 2: {
            int threads = 512;
            int blocks = (size + threads - 1) / threads;
            axpby_kernel<512><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
        default: {
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            axpby_kernel<256><<<blocks, threads, 0, stream>>>(out, a, b, alpha, size);
            break;
        }
    }
}
```

**Design choices:**
- Only includes `<cuda_runtime.h>` — **zero** PyTorch headers
- Receives raw `float*` pointers, not tensors
- `config` parameter enables runtime tuning without recompilation
- `extern "C"` for clean linkage to the binding layer
- Template on `THREADS` for compile-time block size specialization

### File 2: `kernels/axpby_binding.cpp` — Thin PyTorch ↔ CUDA Bridge

```cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare the launcher from the .cu file
extern "C" void axpby_launcher(
    float* out, const float* a, const float* b,
    float alpha, int size, int config, cudaStream_t stream
);

// PyTorch tensor → raw pointer conversion
static torch::Tensor axpby_forward(torch::Tensor a, torch::Tensor b,
                                    double alpha, int config = 0) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    auto out = torch::empty_like(a);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    axpby_launcher(
        out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(),
        static_cast<float>(alpha), static_cast<int>(a.numel()),
        config, stream
    );
    return out;
}

static void register_axpby(pybind11::module& m) {
    m.def("axpby_forward", &axpby_forward,
          py::arg("a"), py::arg("b"), py::arg("alpha"), py::arg("config") = 0);
}

// Self-registration — no need to touch binding.cpp
REGISTER_BINDING(axpby, register_axpby);
```

**Design choices:**
- Uses `<torch/types.h>` + `<torch/csrc/utils/pybind.h>` (NOT `<torch/extension.h>`) for **faster compilation**
- Input validation (CUDA, contiguous, float32, same shape)
- `torch::empty_like` for output allocation — the **only** allowed torch operation
- Gets the correct CUDA stream via `c10::cuda::getCurrentCUDAStream()`

### File 3: `model_new.py` — Agent's Optimized Model

```python
import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, alpha: float) -> None:   # MUST match Model.__init__ signature
        super().__init__()
        self.alpha = alpha

    def forward(self, a, b):
        return cuda_extension.axpby_forward(a, b, self.alpha, 0)
```

**Critical constraint:** `ModelNew.__init__` must match `Model.__init__` signature exactly,
so `load_state_dict()` works for weight transfer.

---

## 6. Compilation System

### `utils/compile.sh` — Simple Wrapper with Timing

```bash
#!/bin/bash
START_TIME=$(date +%s.%N)
python3 -m utils.compile
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {printf \"%.2f\", $END_TIME - $START_TIME}")
echo "[TIME] Compilation took ${ELAPSED}s"
exit $EXIT_CODE
```

### `utils/compile.py` — Auto-Discover Sources, Force-Recompile

```python
#!/usr/bin/env python3
"""Simplified compile script: force-compile root and kernels CUDA/C++ sources."""

import shutil
import sys
from pathlib import Path
import torch.utils.cpp_extension as cpp_ext


def find_sources() -> list[str]:
    root = Path('.')
    kernels_dir = Path('kernels')
    root_sources = [str(p) for p in root.glob('*.cu')] + \
                   [str(p) for p in root.glob('*.cpp')]
    kernel_sources = []
    if kernels_dir.is_dir():
        kernel_sources = [str(p) for p in kernels_dir.glob('*.cu')] + \
                         [str(p) for p in kernels_dir.glob('*.cpp')]
    return sorted(set(root_sources + kernel_sources))


def compile_kernels() -> int:
    build_dir = Path('build/forced_compile')
    output_so = Path('cuda_extension.so')
    sources = find_sources()

    if not sources:
        print('Error: no source files found (*.cu, *.cpp in root or kernels/)')
        return 1

    print(f'Compiling {len(sources)} files: {", ".join(sources)}')

    # Force clean rebuild every time
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    if output_so.exists():
        output_so.unlink()

    try:
        cpp_ext.load(
            name='cuda_extension',
            sources=sources,
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
        )
    except Exception as exc:
        print('Compilation failed.')
        print(str(exc))
        return 1

    built_so = build_dir / 'cuda_extension.so'
    if built_so.exists():
        shutil.copy2(built_so, output_so)
        print(f'Compile success: {output_so}')
        return 0

    print('Compilation finished but cuda_extension.so was not generated.')
    return 1


def main() -> int:
    return compile_kernels()


if __name__ == '__main__':
    sys.exit(main())
```

**Key behaviors:**
- Auto-discovers all `.cu` and `.cpp` files in root + `kernels/`
- Force-deletes `build/` directory before every compile (no stale caches)
- Uses `torch.utils.cpp_extension.load()` which handles nvcc invocation, include paths, linking
- Copies resulting `.so` to working directory for `import cuda_extension`
- `TORCH_CUDA_ARCH_LIST` env var controls target GPU architecture

---

## 7. Verification System (Anti-Cheating)

### `utils/verification.py` — Complete Implementation

```python
#!/usr/bin/env python3
"""Correctness verification for CUDA extension model."""

import torch
import torch.nn.functional as F
from contextlib import contextmanager

from model import Model, get_inputs, get_init_inputs
from model_new import ModelNew


def transform_tensors(tensors, fn):
    """Recursively apply fn to all tensors in a nested structure."""
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def check_equal(actual, expected):
    """Recursive equality check with tolerance for tensors."""
    assert type(actual) == type(expected), f"{type(actual)=} != {type(expected)=}"
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), f"{len(actual)=} != {len(expected)=}"
        for x, y in zip(actual, expected):
            check_equal(x, y)
    elif isinstance(actual, dict):
        for key, val in expected.items():
            assert key in actual, f"Missing key in output: {key}"
            check_equal(actual[key], val)
    elif isinstance(actual, (str, float, int)):
        assert actual == expected, f"{actual=} != {expected=}"
    elif isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
    else:
        raise TypeError(f"Unsupported output type: {type(actual)}")


@contextmanager
def block_torch_functional(excludes=None):
    """
    THE CRITICAL ANTI-CHEAT MECHANISM.
    
    Monkey-patches EVERY callable in torch.nn.functional to raise RuntimeError
    during ModelNew.forward(). This prevents the agent from just calling F.relu(),
    F.conv2d(), etc. instead of writing real CUDA kernels.
    """
    if excludes is None:
        excludes = set()

    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith("_") and name not in excludes:
            originals[name] = attr

            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(
                    f"Function torch.nn.functional.{__name} is not allowed "
                    f"in this context."
                )

            setattr(F, name, wrapper)

    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


def initialize_models():
    """Initialize both reference and agent models with identical weights."""
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    torch_model = Model(*init_inputs).eval().cuda()
    cuda_model = ModelNew(*init_inputs).eval().cuda()
    # Force identical weights — ModelNew MUST have same parameter structure
    cuda_model.load_state_dict(torch_model.state_dict())
    return torch_model, cuda_model


def build_inputs():
    """Generate fresh random inputs for each check — can't memorize outputs."""
    torch_inputs = get_inputs()
    if not isinstance(torch_inputs, (list, tuple)):
        torch_inputs = [torch_inputs]
    torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
    cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())
    return torch_inputs, cuda_inputs


def main():
    torch_model, cuda_model = initialize_models()
    num_checks = 5

    with torch.no_grad():
        for i in range(num_checks):
            torch_inputs, cuda_inputs = build_inputs()
            torch_output = torch_model(*torch_inputs)
            # Anti-cheat: blocks torch.nn.functional.* during agent's forward
            with block_torch_functional():
                cuda_output = cuda_model(*cuda_inputs)
            check_equal(cuda_output, torch_output)
            print(f"[PASS] check {i + 1}/{num_checks}")

    torch.cuda.synchronize()
    print("[PASS] verify success")


if __name__ == "__main__":
    main()
```

### What Each Piece Prevents

| Check | Prevents |
|-------|----------|
| `block_torch_functional()` | Agent calling PyTorch ops instead of real kernels |
| `load_state_dict()` transfer | Mismatched parameter structures |
| 5 random inputs per check | Memorizing specific outputs |
| `atol=1e-2, rtol=1e-2` | Tolerates FP precision diffs, catches real bugs |
| `torch.no_grad()` | Consistent eval-mode behavior |

---

## 8. Profiling System

### `utils/profiling.py` — Complete Implementation

```python
#!/usr/bin/env python3
"""Performance profiling for CUDA extension model."""

import argparse
import torch

from model import Model, get_inputs, get_init_inputs
from model_new import ModelNew


def transform_tensors(tensors, fn):
    if isinstance(tensors, torch.Tensor):
        return fn(tensors)
    if isinstance(tensors, (list, tuple)):
        return [transform_tensors(x, fn) for x in tensors]
    if isinstance(tensors, dict):
        return {k: transform_tensors(v, fn) for k, v in tensors.items()}
    return tensors


def get_prof_ctx():
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    )


def initialize_models():
    init_inputs = get_init_inputs()
    if not isinstance(init_inputs, (list, tuple)):
        init_inputs = [init_inputs]

    torch_model = Model(*init_inputs).eval().cuda()
    cuda_model = ModelNew(*init_inputs).eval().cuda()
    cuda_model.load_state_dict(torch_model.state_dict())

    torch_inputs = get_inputs()
    if not isinstance(torch_inputs, (list, tuple)):
        torch_inputs = [torch_inputs]
    torch_inputs = transform_tensors(torch_inputs, lambda x: x.cuda())
    cuda_inputs = transform_tensors(torch_inputs, lambda x: x.clone())
    return torch_model, cuda_model, torch_inputs, cuda_inputs


def benchmark_model(model, inputs, warmup_iters, run_iters):
    """Benchmark with warmup, CUDA sync, and profiler for device-only time."""
    with torch.no_grad():
        # Warmup eliminates cold-start effects
        for _ in range(warmup_iters):
            _ = model(*inputs)

        with get_prof_ctx() as ctx:
            torch.cuda.synchronize()
            for _ in range(run_iters):
                _ = model(*inputs)
            torch.cuda.synchronize()

    # Extract CUDA device time ONLY (not CPU overhead)
    return (
        sum(e.device_time for e in ctx.events()
            if e.device_type.name == "CUDA")
        / run_iters
    )


def main():
    args = parse_args()
    torch_model, cuda_model, torch_inputs, cuda_inputs = initialize_models()

    torch_compile_model = torch.compile(torch_model)
    warmup_iters = 5
    run_iters = args.iters  # default 10

    # Benchmark all three variants
    cuda_time = benchmark_model(cuda_model, cuda_inputs, warmup_iters, run_iters)
    torch_time = benchmark_model(torch_model, torch_inputs, warmup_iters, run_iters)
    compile_time = benchmark_model(
        torch_compile_model, torch_inputs, warmup_iters, run_iters
    )
    
    print(f"Torch Baseline: {torch_time:.3f}us, "
          f"Torch Compile: {compile_time:.3f}us, "
          f"CUDA Extension: {cuda_time:.3f}us")
```

### Profiling Design Decisions

| Decision | Rationale |
|----------|-----------|
| `torch.profiler` with CUDA activity | Measures GPU device time, not CPU dispatch overhead |
| 5 warmup iterations | Eliminates JIT compilation, cache cold-start |
| `torch.cuda.synchronize()` before/after | Ensures all GPU work is captured |
| Averaging over `run_iters` | Reduces measurement noise |
| 3 variants (eager, compile, extension) | Full comparison baseline |

---

## 9. Reward Function

### Discrete Milestone-Based Rewards (Equation 1 from paper)

```python
def compute_reward(t_gen, t_eager, t_compile):
    """
    r ∈ {-1, 1, 2, 3}
    
    NOT continuous speedup — discrete milestones that normalize
    across problems of varying difficulty.
    """
    if not correctness_check():
        return -1                    # Failed correctness
    
    beats_eager   = (t_eager - t_gen) / t_eager > 0.05
    beats_compile = (t_compile - t_gen) / t_compile > 0.05
    
    if beats_eager and beats_compile:
        return 3                     # Best: beats both baselines by >5%
    elif beats_eager:
        return 2                     # Good: beats eager only
    else:
        return 1                     # Correct but not faster
```

### Why Discrete, Not Continuous?

Raw `t_compile / t_gen` ratios are:
- **Noisy** — measurement variance dominates for fast kernels
- **Biased** — trivial elementwise ops get 10× while complex convs get 1.2×
- **Outlier-prone** — extreme ratios destabilize PPO

Discrete rewards normalize difficulty. A correct-but-slow ResNet kernel (r=1) vs a
blazing-fast diagonal matmul (r=3) are on the same scale.

**Ablation proof** (Table 2): Using continuous speedup reward instead drops faster rate
from 96.8% → 60.4%.

---

## 10. Agent Loop & Episode Structure

### ReAct-Style Agent Loop

```
┌─────────────────────────────────┐
│  Episode Start                  │
│  Agent receives:                │
│  • SKILL.md (system prompt)     │
│  • model.py (frozen reference)  │
│  • model_new.py (template)      │
│  • empty kernels/ directory     │
│  • Full tool suite              │
│    (Bash, Read, Write, Edit,    │
│     Glob, Grep)                 │
└──────────┬──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Agent writes:       │◄──────────────────────┐
│  • kernels/*.cu      │                       │
│  • kernels/*_binding │                       │
│  • model_new.py      │                       │
└──────────┬───────────┘                       │
           │                                   │
           ▼                                   │
┌──────────────────────┐                       │
│  compile.sh          │──── FAIL ────► iterate │
└──────────┬───────────┘                       │
           │ SUCCESS                           │
           ▼                                   │
┌──────────────────────┐                       │
│  verification.py     │──── FAIL ────► iterate │
│  (5 random checks)   │                       │
└──────────┬───────────┘                       │
           │ PASS                              │
           ▼                                   │
┌──────────────────────┐                       │
│  profiling.py        │──── < 5% gain ► iterate
│  (3 variants)        │                       │
└──────────┬───────────┘                       │
           │ ≥ 5% faster                       │
           ▼                                   │
┌──────────────────────┐                       │
│  Continue optimizing │───────────────────────┘
│  (push for best)     │
└──────────┬───────────┘
           │ diminishing returns / max turns
           ▼
┌──────────────────────┐
│  Cleanup & Submit    │
└──────────────────────┘
```

### Typical Episode Trajectory

```
Turn  1: Read model.py → understand computation
Turn  2: Write kernels/fused_op.cu (CUDA kernel)
Turn  3: Write kernels/fused_op_binding.cpp (pybind11 wrapper)
Turn  4: Write model_new.py (call cuda_extension.fused_op_forward)
Turn  5: bash utils/compile.sh → observe errors
Turn  6: Fix compilation error in .cu file
Turn  7: bash utils/compile.sh → success
Turn  8: python3 -m utils.verification → [PASS] or [FAIL]
Turn  9: python3 -m utils.profiling → get timing numbers
Turn 10+: Iterate optimizations (shared memory, fusion, vectorization...)
```

### Episode Parameters

| Parameter | Training | Evaluation |
|-----------|----------|------------|
| Max agent turns | 150 | 200 |
| Context window | 128K tokens | 128K tokens |
| GPU pool | 128 NVIDIA H20 | Same |

---

## 11. RL Training Pipeline (4 Stages)

### Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Stage 1:   │     │  Stage 2:    │     │  Stage 3:   │     │  Stage 4:    │
│  Single-Turn│────▶│  Collect     │────▶│  Warmup     │────▶│  Full        │
│  RL (PPO)   │     │  Agent       │     │  Actor +    │     │  Agentic RL  │
│             │     │  Trajectories│     │  Critic     │     │              │
│  32K context│     │              │     │             │     │  128K context│
│  One-shot   │     │  Run Stage 1 │     │  Actor: RFT │     │  150 steps   │
│  kernel gen │     │  model in    │     │  on filtered│     │  PPO with    │
│             │     │  full agent  │     │  trajs      │     │  asymmetric  │
│             │     │  loop        │     │             │     │  clipping    │
│             │     │              │     │  Critic:    │     │              │
│             │     │              │     │  Value pre- │     │              │
│             │     │              │     │  training   │     │              │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

### Stage 1: Single-Turn RL (PPO on Base Model)

**Purpose:** Teach the base model basic CUDA generation (one-shot, no iteration)

- Base model: Seed 1.6 (230B MoE, 23B active)
- Context: 32K tokens (smaller = more stable gradients)
- PPO optimization, base model = policy + value network
- Output: model that can write CUDA kernels in a single turn

### Stage 2: Trajectory Collection

**Purpose:** Generate multi-turn agent trajectories for warmup

- Take Stage 1 model, run it in the full agent loop
- Collect complete trajectories: write → compile → verify → profile → iterate
- These trajectories are used to initialize both actor and critic

### Stage 3a: Actor Warmup (Rejection Fine-Tuning / RFT)

**Purpose:** Initialize actor with strong behavioral prior, prevent entropy explosion

**Filter trajectories by:**
1. **Outcome filtering:** Keep only R > 0 (positive reward)
2. **Pattern filtering:** Discard trajectories with:
   - Redundant multi-turn loops
   - Hallucinated tool calls
   - Invalid schemas

**Training objective:**
```
L_RFT(θ) = -E[Σ log π_θ(a_t | s_t, a_{<t})]   over filtered trajectories
```

**Without RFT:** Training collapses after ~17 steps. Actor entropy surges → incoherent outputs.

### Stage 3b: Critic Warmup (Value Pretraining)

**Purpose:** Initialize critic to provide accurate advantage estimates immediately

**Method:** GAE (λ=0.95, γ=1.0) on collected trajectories:

```
V_target_t = V_φ(s_t) + Â_t

where Â_t = Σ (γλ)^l · δ_{t+l}
and   δ_t = r_t + γV_φ(s_{t+1}) - V_φ(s_t)
```

**Loss:**
```
L_VP(φ) = (1/2) · E[(1/T) · Σ (V_φ(s_t) - V_target_t)²]
```

**Without Value Pretraining:** Critic can't distinguish good/bad states → agent generates
infinitely long trajectories (no penalty for fruitless search).

### Stage 4: Full Agentic RL

**Purpose:** Full PPO training with properly initialized actor + critic

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Algorithm | PPO with asymmetric clipping |
| ε_lower | 0.2 |
| ε_higher | 0.28 |
| Batch size | 1024 |
| Actor LR | 3 × 10⁻⁶ |
| Critic LR | 6 × 10⁻⁶ |
| Context window | 128K tokens |
| Training steps | 150 |
| Max agent turns | 150 per episode |
| γ (discount) | 1.0 |
| λ (GAE) | 0.95 |

### The Root Cause of Instability

```
CUDA code tokens ≈ 0.01% of pretraining data
    ↓
Many tokens have π_θ(a|s) ≈ 10⁻⁹ (near precision floor)
    ↓
Importance sampling ratio ρ = π_θ(a|s) / π_old(a|s)
    ↓
Small numerical errors in BF16/FP16 at 10⁻⁹ scale
    ↓
ρ fluctuates wildly or EXPLODES
    ↓
Gradient updates become unstable → collapse at step 17
```

The warmup stages shift the model's distribution closer to CUDA code, raising token
probabilities out of the "precision floor" danger zone.

### Ablation Results — Every Stage is Load-Bearing

| What's Removed | Pass Rate | Faster vs Compile | Speedup vs Compile |
|---------------|-----------|-------------------|-------------------|
| **Full system** | **98.8%** | **96.8%** | **2.11×** |
| Agent loop (single-turn only) | 77.1% | 14.1% | 0.69× |
| Robust reward (use raw speedup) | 96.8% | 60.4% | 1.25× |
| RFT warmup | 95.6% | 49.8% | 1.05× |
| Value pretraining | 98.6% | 50.9% | 1.00× |

---

## 12. Agent System Prompt (SKILL.md)

This is the complete system prompt the agent sees during each episode.

### Critical Restrictions

```
⚠️ STRICTLY FORBIDDEN:
- NO torch operators in C++ (never use torch::* in .cu files)
- NO torch operations in model_new.py (only tensor creation + custom ops)
- NO third-party libraries (except cuBLAS for GEMM, cuDNN for Conv)
- NO modifications to utils/, binding.cpp, binding_registry.h

✅ ALLOWED ONLY:
- Raw CUDA kernels, cuBLAS (GEMM), cuDNN (Conv/ConvTranspose)
- torch.tensor creation, custom extension ops, tensor properties
- torch::empty_like for allocation only
```

### Workflow (4 Repeating Steps)

```
Step 1: Write    → kernels/*.cu + *_binding.cpp + model_new.py
Step 2: Compile  → TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
Step 3: Verify   → sudo python3 -m utils.verification
         Profile → sudo python3 -m utils.profiling
Step 4: Iterate  → until ≥5% faster than torch.compile, then keep pushing
```

### Optimization Priority Order

```
Priority 1 (>50% impact):
  - Kernel fusion — reduce memory traffic
  - Shared memory tiling — improve data reuse
  - Memory coalescing — consecutive access patterns

Priority 2 (20-50% impact):
  - Vectorized loads (float2/float4)
  - Warp-level primitives (__shfl_sync, __ballot_sync)
  - Occupancy tuning (block size, register usage)

Priority 3 (<20% impact):
  - Instruction-level parallelism
  - Mixed precision (FP16/TF32)
  - Prefetching and double buffering
```

### Optimization Checklists

**Essential (Apply First):**
- [ ] Memory Coalescing: consecutive threads → consecutive addresses
- [ ] Kernel Fusion: combine ops to reduce memory traffic
- [ ] Shared Memory: cache frequently accessed data
- [ ] Grid-Stride Loops: handle data larger than grid size
- [ ] Boundary Checks: validate `tid < size`

**Performance (Apply as Needed):**
- [ ] Vectorized Memory: float2/float4
- [ ] Warp Primitives: `__shfl_sync`
- [ ] Occupancy Tuning: balance block size and resources
- [ ] Bank Conflict Avoidance: pad shared memory
- [ ] Loop Unrolling: increase ILP

**Advanced (Final Tuning):**
- [ ] Tensor Cores: WMMA/MMA for GEMM
- [ ] Mixed Precision: FP16/TF32
- [ ] Persistent Kernels: keep data in registers
- [ ] CUDA Graphs: reduce launch overhead
- [ ] Double Buffering: overlap compute + memory

### Success Criteria

```
MINIMUM:  ≤ 0.95× torch.compile time (5% faster)
TARGET:   Best possible — every microsecond counts
CORRECT:  atol=1e-2, rtol=1e-2
CLEAN:    kernels/ contains ONLY final version (no intermediate attempts)
```

---

## 13. Anti-Reward-Hacking Stack

### Complete Defense-in-Depth

| Layer | Mechanism | Implementation | Prevents |
|-------|-----------|---------------|----------|
| 1 | **torch.nn.functional blocked** | `block_torch_functional()` context manager monkey-patches ALL `F.*` callables to raise `RuntimeError` during `ModelNew.forward()` | Calling PyTorch ops instead of real CUDA kernels |
| 2 | **File permissions** | `utils/`, `binding.cpp`, `binding_registry.h` are read-only. Verification runs with `sudo`. | Modifying evaluation scripts |
| 3 | **5 random input checks** | `get_inputs()` generates fresh `torch.randn` each invocation | Memorizing specific outputs |
| 4 | **State dict transfer** | `cuda_model.load_state_dict(torch_model.state_dict())` | Mismatched parameter structures |
| 5 | **No web access** | No search or retrieval tools in agent toolkit | Copying solutions from internet |
| 6 | **Process isolation** | Each eval on dedicated GPU from 128 H20 pool | Inter-process interference / timing manipulation |
| 7 | **CUDA sync + warmup** | `torch.cuda.synchronize()` + 5 warmup iters in profiling | Measurement noise / cold-start bias |
| 8 | **Deterministic tasks only** | Stochastic ops (dropout etc.) filtered from training data | Non-reproducible correctness checks |

### Why Each is Necessary

Without `block_torch_functional()`: Agent learns to call `F.relu()` directly — passes correctness,
claims speedup from avoiding compilation overhead.

Without file permissions: Agent modifies `verification.py` to always print `[PASS]`.

Without random inputs: Agent hardcodes expected outputs for specific input values.

Without process isolation: Timing measurements contaminated by other GPU workloads.

---

## 14. Optimization Patterns the Agent Learns

### Case Studies from Paper (Appendix D)

| Task | Level | Pattern | Speedup vs compile |
|------|-------|---------|--------------------|
| Diagonal matmul | L1 | `diag(v) @ M = v * M` — algebraic simplification eliminates GEMM entirely | **73.3×** |
| Matmul + div + sum + scale | L2 | Algebraic rearrangement: `Σ(x·wⱼᵀ)/2 = x·(Σwⱼᵀ)/2`, reduces O(NM²) → O(NM) + vectorized float4 loads + shared-memory tree reduction | **24.0×** |
| ResNet BasicBlock | L3 | BN folding into conv weights + cuDNN fused conv+bias+ReLU + TF32 Tensor Cores + fused add+ReLU kernel | **3.59×** |

### Recurring Pattern Categories

1. **Algebraic Simplification:** Recognize mathematical structure to reduce op count
2. **Kernel Fusion:** Merge sequential ops into single kernel, eliminate intermediate tensors
3. **Memory Access Optimization:** Coalesced reads, shared memory, vectorized loads (float4)
4. **Hardware-Aware:** TF32 for Tensor Cores, cuDNN fused APIs, NCHW vs NHWC analysis
5. **Library-Aware:** Use cuDNN/cuBLAS where they outperform custom kernels

### ResNet BasicBlock — Multi-Technique Example

The agent applies **5 complementary techniques** to a single model:

```
1. BatchNorm folding → eliminate BN ops entirely (fuse into conv weights)
2. cuDNN fused conv+bias+ReLU → single library call instead of 3 ops
3. TF32 enabled → leverage Tensor Cores on Hopper GPUs
4. Fused add+ReLU kernel → custom CUDA for residual + activation
5. NCHW layout retained → explored NHWC but conversion overhead > benefit
```

The agent **explores and abandons** NHWC layout conversion — demonstrating search-and-evaluate
capability, not just pattern matching.

---

## 15. What to Build (Implementation Checklist)

### Phase 1: Environment Infrastructure

- [ ] Create `agent_workdir/` directory structure
- [ ] Implement `binding_registry.h` (copy exactly — this is infrastructure)
- [ ] Implement `binding.cpp` (8 lines, never modified)
- [ ] Implement `utils/compile.py` with auto-discovery + force-recompile
- [ ] Implement `utils/compile.sh` wrapper with timing
- [ ] Implement `utils/verification.py` with:
  - [ ] `block_torch_functional()` context manager
  - [ ] `load_state_dict()` weight transfer
  - [ ] 5 random input checks
  - [ ] `atol=1e-2, rtol=1e-2` tolerance
- [ ] Implement `utils/profiling.py` with:
  - [ ] 3-way comparison (eager, compile, extension)
  - [ ] `torch.profiler` for CUDA device time
  - [ ] Warmup iterations
  - [ ] `torch.cuda.synchronize()` barriers
- [ ] Set file permissions (make utils/ read-only)

### Phase 2: Task Dataset

- [ ] Crawl `torch.nn` modules for seed operators
- [ ] Build `get_inputs()` and `get_init_inputs()` generators for each
- [ ] Use LLM to compose 2-5 operators into fused tasks
- [ ] Filter: executable in eager + compile, deterministic, 1-100ms runtime
- [ ] AST similarity check vs evaluation set
- [ ] Target: 6,000 filtered samples

### Phase 3: Agent System Prompt

- [ ] Write `SKILL.md` with:
  - [ ] Critical restrictions (no torch ops in C++, no modifying infrastructure)
  - [ ] Workspace structure documentation
  - [ ] Three-file kernel contract with examples
  - [ ] Compilation commands
  - [ ] Optimization priority order
  - [ ] Checklists (essential, performance, advanced, correctness)
  - [ ] Common issues and solutions table
  - [ ] Success criteria

### Phase 4: Reward Pipeline

- [ ] Implement 3-stage evaluation: compile → verify → profile
- [ ] Implement discrete reward function: `r ∈ {-1, 1, 2, 3}`
- [ ] NOT continuous speedup — discrete milestones only
- [ ] Binary correctness gate before performance evaluation

### Phase 5: RL Training

- [ ] **Stage 1:** Single-turn PPO on base model (32K context)
- [ ] **Stage 2:** Collect agent trajectories from Stage 1 model in full loop
- [ ] **Stage 3a:** RFT on filtered trajectories (positive reward + good patterns)
- [ ] **Stage 3b:** Value pretraining on same trajectories (GAE λ=0.95, γ=1.0)
- [ ] **Stage 4:** Full PPO with asymmetric clipping (ε_lower=0.2, ε_higher=0.28)
  - Batch size 1024
  - Actor LR 3e-6, Critic LR 6e-6
  - 128K context, 150 steps, 150 turns/episode

### Phase 6: Sandbox Infrastructure

- [ ] Docker containers for CPU tasks (compilation)
- [ ] GPU pool with process-level isolation
- [ ] Exclusive GPU allocation per evaluation
- [ ] No web access in agent environment

---

## Appendix: Key Design Decisions and Rationale

| Decision | Rationale |
|----------|-----------|
| `.cu` files have zero PyTorch deps | Forces real kernel writing; enables clean compilation |
| `_binding.cpp` does pointer extraction | Thin layer, easy to template; keeps .cu pure |
| `extern "C"` launcher interface | Prevents C++ name mangling; clean .cu ↔ _binding.cpp boundary |
| `config` parameter in launchers | Runtime tuning without recompilation |
| `torch::empty_like` only allowed torch op | Minimal framework coupling in C++ layer |
| `<torch/types.h>` not `<torch/extension.h>` | 2-3× faster compilation |
| `load_state_dict()` weight transfer | Ensures identical weights for numerical comparison |
| 5 random checks with `atol=1e-2` | Catches bugs while tolerating FP precision differences |
| Discrete rewards `{-1,1,2,3}` | Normalizes difficulty; avoids outlier bias |
| Asymmetric PPO clipping | Allows slightly more exploration upward (ε_higher > ε_lower) |
| RFT before full RL | Prevents entropy explosion and policy collapse |
| Value pretraining before full RL | Prevents infinite trajectory generation |
| 32K → 128K context progression | Smaller context first for stability, then scale up |
```

---

This is the complete reference document. It contains every piece of the CUDA-Agent system: the actual source code for all infrastructure files, the paper's methodology, the training pipeline with exact hyperparameters, the anti-cheating stack, and the implementation checklist for replication. You can use it as-is as your project's `SKILLS.md`.