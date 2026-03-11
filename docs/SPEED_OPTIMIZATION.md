# KernelForge Training — Speed Optimization Plan

How to speed up GRPO training without destroying the training signal.

---

## Principles

These constraints are non-negotiable:

- **Never reduce G below 4.** GRPO needs reward variance. G=2 produces zero gradients.
- **Never reduce LoRA rank below 32.** Code generation is LoRA's hardest task (per "LoRA Learns Less and Forgets Less", 2024). Rank 64 recommended for models up to 14B.
- **Never use 4-bit quantization without testing.** INT8 is the safe middle ground.
- **Speed up generation, not the training signal.** The bottleneck is inference (generating G completions per prompt), not the optimizer step.

---

## Tier 1: Quick Wins (config changes only)

### 1a. Install causal-conv1d for Qwen 3.5 Mamba layers

The Qwen 3.5 architecture has Mamba-style linear attention layers. Without `causal-conv1d`, these fall back to pure PyTorch (significantly slower).

**File:** `modal_train.py` — image build section

```python
# BEFORE (lines 35-37):
.pip_install("xformers>=0.0.29")
# TODO: install causal-conv1d for fast linear attention (Qwen 3.5 Mamba layers).
# Skipped for now — building from source takes too long. Torch fallback works.

# AFTER:
.pip_install("xformers>=0.0.29")
.pip_install("causal-conv1d>=1.4.0")
```

Remove the TODO comment on lines 36-37.

**Expected speedup:** 10-30% faster forward pass on linear attention layers.
**Quality impact:** None — identical computation, just a faster implementation.
**Risk:** Low. If the pip install fails during image build, just revert.

---

### 1b. Use INT8 quantization for 9B model

The 9B model at bf16 uses ~18GB VRAM. INT8 reduces this to ~9GB, freeing memory for larger batches or KV cache.

**File:** `modal_train.py` — Stage 1 env var block (after the existing env vars)

```python
# Add after the existing Stage 1 env vars:
os.environ.setdefault("KERNELFORGE_QUANT_BITS", "8")
```

The model loader (`training/model_loader.py:147-155`) already supports this:

```python
elif quant_bits == 8:
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
```

**Expected speedup:** Frees ~9GB VRAM for larger batches.
**Quality impact:** Minimal. INT8 is well-tested and much better than 4-bit. The model_loader.py comments explicitly recommend INT8 for accuracy.
**Risk:** Low. If quality degrades, remove the env var to go back to bf16.

---

### 1c. Increase effective batch size to 8

With INT8 freeing VRAM (1b), we can accumulate more gradient steps.

**File:** `modal_train.py` — Stage 1 gradient accumulation setting

```python
# BEFORE:
os.environ.setdefault("KERNELFORGE_STAGE1_GRADIENT_ACCUMULATION_STEPS", "4")

# AFTER:
os.environ.setdefault("KERNELFORGE_STAGE1_GRADIENT_ACCUMULATION_STEPS", "8")
```

Effective batch = `per_device_batch_size × gradient_accumulation_steps` = 1 × 8 = 8. Must be divisible by G (4 or 8). Both work.

**Expected speedup:** Better GPU utilization per optimizer step.
**Quality impact:** Positive — more samples per update = more stable gradients.
**Risk:** Low. If OOM, reduce back to 4.

---

### Tier 1 verification

After applying 1a + 1b + 1c:

```bash
uv run modal run --detach modal_train.py --stage 1 --max-steps 5
```

Compare against baseline:
- Step time (seconds per step) — should decrease
- `reward_std > 0` frequency — should stay the same or improve
- `eval_ok` count — should stay the same
- `loss` and `grad_norm` magnitude — should be non-zero

---

## Tier 2: Medium Effort (code changes)

### 2a. Re-enable vLLM in colocate mode

vLLM provides 2-5x faster generation by using PagedAttention and continuous batching. This is the single biggest potential speedup since generation is the bottleneck.

**Current blocker:** vLLM requires `transformers<5`, but Qwen 3.5 requires `transformers>=5.2.0`.

**Options:**
1. Check if vLLM 0.8+ supports transformers 5.x (likely by now)
2. Use vLLM's external server mode with a separate inference container
3. Wait for vLLM compatibility update

**If available:**

```python
os.environ.setdefault("KERNELFORGE_USE_VLLM", "1")
os.environ.setdefault("KERNELFORGE_VLLM_MODE", "colocate")
os.environ.setdefault("KERNELFORGE_VLLM_GPU_MEMORY_UTILIZATION", "0.6")
```

**Expected speedup:** 2-5x faster generation (the #1 bottleneck).
**Quality impact:** None — identical outputs, just faster sampling.

---

### 2b. Use FP16 instead of BF16

Research shows FP16's 10 mantissa bits (vs BF16's 7) reduce rounding errors that cause mid-training reward collapses. A100 delivers identical TFLOPS for both formats.

**File:** `training/grpo_config.py:48`

```python
# BEFORE:
bf16=is_linux,

# AFTER:
bf16=False,
# And add fp16=True to the GRPOConfig kwargs
```

**Expected speedup:** None (same TFLOPS), but better training stability.
**Quality impact:** Positive — fewer reward collapses from accumulation errors.

---

### 2c. Pre-compile and cache the Modal training image

Currently each run may rebuild image layers. Pin the image hash or use `modal.Image.from_registry` with a pre-built image.

**Expected speedup:** 1-2 minutes saved per run on image build.

---

## Tier 3: Larger Changes (architecture)

### 3a. Optimal Variance Filtering (OVF)

After generating G completions, only include the highest-variance subset in the GRPO update. G=8 with OVF can match G=16 without it — same generation cost, better gradient signal.

**File:** `training/custom_grpo_trainer.py` — override advantage computation to filter low-variance groups before computing the loss.

**Expected speedup:** Same generation cost, better gradient signal per step.
**Quality impact:** Positive — focuses learning on the most informative samples.

---

### 3b. Async generation with Ray workers

Decouple generation from training. While optimizer updates run, generation for the next batch happens concurrently on separate workers.

**Expected speedup:** ~40% throughput gain (per Red Hat async-GRPO research).
**Quality impact:** Slightly off-policy (model generates with weights from 1 step ago), but convergence is guaranteed and commonly used in practice.

---

## Implementation Order

### Phase 1: Validate current state
The reverted config (v1.2 + opus_9b + modal eval) should produce positive rewards. Confirm this before any optimization.

### Phase 2: Apply Tier 1 (1a + 1b + 1c)
All are config/pip changes. Apply together, run 5-step test, compare step time.

### Phase 3: Investigate Tier 2
Check vLLM 0.8+ compatibility with transformers 5.x. If available, this is the single biggest win.

### Phase 4: Consider Tier 3
OVF and async generation are architectural changes. Only pursue after Tiers 1-2 are stable.

---

## Files to modify

| File | Change | Tier |
|------|--------|------|
| `modal_train.py` (image build) | Add `causal-conv1d>=1.4.0` pip install | 1a |
| `modal_train.py` (Stage 1 env vars) | Add `KERNELFORGE_QUANT_BITS=8` | 1b |
| `modal_train.py` (Stage 1 env vars) | Change grad accumulation 4→8 | 1c |
| `modal_train.py` (image build) | Check vLLM 0.8+ compat | 2a |
| `training/grpo_config.py` | bf16→fp16 option | 2b |
| `training/custom_grpo_trainer.py` | OVF filtering in advantage computation | 3a |

---

## References

- "LoRA Learns Less and Forgets Less" (2024) — code generation requires high LoRA rank
- Kevin-32B (2025) — used G=16, LoRA rank 64, TRLOO correction
- Red Hat async-GRPO research — 40% throughput gain from decoupled generation
- TRL GRPOTrainer docs — `vllm_mode="colocate"` for on-GPU inference acceleration
