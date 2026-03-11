# KernelForge GRPO Training — Status & Next Steps

**Last updated:** 2026-03-11
**Current version:** v2.0 (best-practices audit applied)
**Current state:** Pipeline working, no correct kernels yet

---

## 1. How We RL (Current Methodology)

### Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Model | Qwen 3.5 9B (Claude 4.6 Opus distilled) | Best reasoning model in 9B class, `<think>` chains |
| Fine-tuning | LoRA r=64, alpha=64 | Code gen requires high rank (LoRA Learns Less, 2024) |
| Precision | bf16 (no quantization) | Full precision for training stability; INT8/4-bit caused hangs |
| GPU | A100-80GB via Modal | Same GPU for training + eval eliminates cross-GPU latency |
| VRAM budget | ~10 GB model + ~6-8 GB training overhead = ~18 GB / 80 GB | Large margin for G=8 at 1024 tokens |

### GRPO Configuration

| Parameter | Value | Why |
|-----------|-------|-----|
| G (num_generations) | **8** | Consensus recommendation (Unsloth, DeepSeek-R1). Statistically robust advantage estimates |
| TRLOO correction | Enabled (G/(G-1) = 8/7 = 1.14x scaling) | Corrects GRPO self-inclusion bias (arXiv 2602.05885) |
| beta (KL penalty) | 0.0 | Matches DAPO findings for sparse-reward RL. Saves VRAM (no ref model) |
| Temperature | 1.0 (Stage 1) | High exploration for warmup phase |
| max_completion_length | 1024 tokens | CUDA kernels need 500-800 tokens minimum. Baseline WCC is ~1,100 tokens |
| Batch | 1 per_device × 8 grad_accum = 8 effective | Divisible by G=8 |
| LR | 2e-6, cosine decay | Conservative to avoid catastrophic forgetting |
| Optimizer | paged_adamw_8bit | Memory-efficient for LoRA |

### Reward Design

Discrete milestone rewards from real A100 compile/correctness/timing:

```
-1.0  no_code               No CUDA code extracted from completion
-0.7  truncated_partial      Hit max_completion_length before code complete
-0.5  local_compile_fail     Syntax error (pre-remote nvcc fast-fail)
-0.4  remote_compile_fail    Compile error on target A100
-0.3  runtime_error          CUDA runtime exception during execution
-0.2  correctness_fail       Compiles and runs but wrong output
+0.2  correct_slow           Correct but <0.98x baseline speed
+0.4  correct_parity         Correct and >=0.98x baseline
+0.7  correct_fast_eager     Correct and >1.05x vs eager PyTorch
+1.0  correct_fast_compile   Correct and >1.05x vs torch.compile
```

**Why discrete milestones:** Robust to timing noise. Each tier provides a distinct gradient signal. The 13-tier ladder gives fine-grained learning curriculum without continuous reward shaping complexity.

### Generation & Evaluation

| Component | Choice | Why |
|-----------|--------|-----|
| Generation backend | HuggingFace generate (no vLLM) | vLLM blocked by transformers 5.x incompatibility |
| Attention backend | xformers + flash-attn>=2.5.0 | flash-attn provides 10-20% speedup, xformers as fallback |
| torch.compile | Re-enabled | Bug #4025 only affects Unsloth; we bypass via SKIP_UNSLOTH=1 |
| Eval backend | Modal remote A100 | Real compile + correctness + timing on target hardware |
| Local fast-fail | nvcc -arch=sm_80 syntax check | Catches obvious compile errors before expensive remote eval |
| TRL version | 0.29.0 | 5 months of bug fixes over 0.24.0, matches pyproject.toml |

### Training Stages

| Stage | Purpose | Status |
|-------|---------|--------|
| Stage 0 | Smoke test (model load, canary gen, eval connectivity) | Working |
| Stage 1 | GRPO warmup (G=8, T=1.0, 100 steps default) | Working (mechanically), no correct kernels yet |
| Stage 2 | RFT — filter trajectories + SFT on best | Never run |
| Stage 3 | Full GRPO pilot (T=0.7, multi-turn) | Never run |

### Key Implementation Details

- **TRLOOGRPOTrainer** (`training/custom_grpo_trainer.py`): Overrides `_generate_and_score_completions` to apply N/(N-1) advantage scaling after TRL computes raw advantages. Also implements OVF (Optimal Variance Filtering) — not yet enabled.
- **Multi-turn rollout** (`training/multi_turn_rollout.py`): Manages the prompt→generate→extract→eval→feedback loop. Supports up to 3 turns per episode.
- **Eval dispatch** (`openenv_env/eval_backend.py`): Routes evaluation to Modal A100. Shared eval core in `eval_service/eval_core.py`.

---

## 2. What We Validated

These have been empirically confirmed through actual training runs on Modal A100-80GB.

### End-to-End RL Loop Works
- Model generates CUDA code from prompts
- Code extraction parses CUDA from model output
- Eval dispatch sends code to remote A100 for compilation + correctness testing
- Rewards flow back to the GRPO trainer
- Gradient updates are applied to LoRA parameters
- **Evidence:** 5-step smoke test completed (run `ap-HvwcRz5ALsCiPpW3Qzs7ds`)

### GRPO Produces Real Gradients
- reward_std = 0.15 (nonzero — different generations get different rewards)
- grad_norm increased from 0.095 → 0.136 over 5 steps (+43%)
- Loss is near-zero but negative (GRPO is slightly improving expected reward)
- No PPO clipping triggered (updates within trust region)
- **Evidence:** Per-step metrics from experiment report

### Model Learns From Reward Signal
- eval_ok rate improved: 25% → 75% (3x improvement in 5 steps)
- Truncation rate dropped: 75% → 25% (model learned to produce shorter code)
- Mean reward improved: -0.625 → -0.475 (+24%)
- **Caveat:** The model learned to avoid truncation, not to produce correct code. But this proves the gradient signal is real.

### TRLOO Correction Is Active
- With G=4 (at time of test): 4/3 = 1.33x advantage scaling applied
- Now configured for G=8: 8/7 = 1.14x scaling (weaker per-sample but more statistically robust)
- Finite-value masking handles NaN advantages correctly

### Infrastructure Works
- Modal image builds and deploys (CUDA 12.4, torch 2.9, xformers)
- Model loads via HuggingFace (VLM auto-detected, text CausalLM extracted)
- Unsloth patching correctly bypassed (SKIP_UNSLOTH=1) due to Qwen 3.5 RoPE bug
- Spawn guard prevents fork deadlocks
- Canary generation validates model before training begins
- Checkpoint persistence to Modal volumes

### Discrete Milestone Rewards Work
- Rewards of -0.70 (truncated) and -0.40 (compile fail) observed
- Different reward tiers produce nonzero advantage variance
- The reward ladder provides a gradient curriculum even when no kernels are correct

---

## 3. What We Haven't Validated Yet

### v2.0 Config Changes (Just Applied, Untested)

| Change | Risk | Mitigation |
|--------|------|------------|
| **TRL 0.29.0** in Modal image | Unsloth may cap trl<=0.24.0, causing pip conflict | API verified locally; Unsloth constraint may need override |
| **G=8** (was G=4) | 2x longer per step; possible OOM | A100 has 60+ GB free VRAM; G=4 worked at 512 tokens |
| **flash-attn>=2.5.0** | May fail to build on Modal image | xformers installed as fallback |
| **torch.compile re-enabled** | May cause shape tracing errors with VLM-extracted CausalLM | Re-disable if 1-step smoke test fails |

### Training Outcomes Not Yet Achieved

| Goal | Status | Blocker |
|------|--------|---------|
| **Positive reward (>0)** | Never achieved | 512-token limit caused 100% truncation of complete kernels. Now fixed to 1024 |
| **Correct CUDA kernel** | Never produced | Same truncation blocker + only 5 steps of training |
| **Reward improvement across a real run** | Not demonstrated | Only 5 steps completed; need 50+ |
| **Non-degenerate reward distribution** | Partial | Only 2 reward tiers observed (-0.70, -0.40). Need to see -0.2 through +1.0 |

### Pipeline Components Never Tested

| Component | Status | Notes |
|-----------|--------|-------|
| Stage 2 (RFT) | Never run | Entrypoint exists, needs Stage 1 trajectories |
| Stage 3 (full GRPO) | Never run | T=0.7, multi-turn, needs Stage 1/2 artifacts |
| OVF (Optimal Variance Filtering) | Implemented, never enabled | Enable with KERNELFORGE_OVF=1 after G=8 works |
| Multi-task training | Never tested | Only WCC kernels so far; Ops-6K tasks available |
| Checkpoint resume | Never tested | Training always starts from base model |
| WandB logging | Not configured | report_to="none" currently |
| vLLM generation | Blocked | transformers 5.x incompatibility |
| causal-conv1d | Removed | Fails to build on Modal (ptxas output rate limit) |
| FP16 training | Never tested | bf16 working; FP16 may improve precision |
| Batch eval parallelism | Partial | BATCH_EVAL=1 set but not verified for speedup |

### Unverified Assumptions

1. **1024 tokens is enough** — Baseline WCC kernel is ~1,100 tokens. Most generated kernels should complete at 1024, but complex variants may still truncate.
2. **G=8 gives better variance** — Literature says yes; we saw reward_std=0.15 with G=4. Expect 0.20-0.25 with G=8.
3. **The model has latent CUDA knowledge** — Historical Turn 2 data showed reward=+0.7, suggesting pre-training knowledge. But this was never reproduced in RL.
4. **Step time ~15 min with G=8** — Extrapolated from ~5 min/step at G=4 with 512 tokens. Actual time with G=8 × 1024 tokens unknown.
5. **The scaling law holds** — Experiment report predicts first positive reward at step 10-20 with 1024 tokens. This is an optimistic estimate.

---

## 4. What to Improve Next

### Tier 1: Validate v2.0 Config (Immediate)

These must happen before any real training run.

**1. 1-step smoke test**
```bash
uv run modal run --detach modal_train.py --stage 1 --max-steps 1
```
Validates: image builds (flash-attn + TRL 0.29.0), model loads, G=8 generation completes, torch.compile doesn't crash.

**2. 5-step validation**
```bash
uv run modal run --detach modal_train.py --stage 1 --max-steps 5
```
Validates: reward_std > 0 on every step, grad_norm > 0.05, step time ~15 min, no OOM.

**3. 50-step real run**
```bash
uv run modal run --detach modal_train.py --stage 1 --max-steps 50
```
**Target metrics by step 50:**
- reward_mean > -0.2 (at least some correct compilations)
- eval_ok rate > 50%
- First positive reward (+0.2) observed
- Truncation rate < 10%

**Estimated cost:** ~12 hours × $2.50/hr = ~$30

### Tier 2: After Positive Rewards

Only pursue these after Tier 1 demonstrates correct kernel generation.

| # | Change | Impact | Effort |
|---|--------|--------|--------|
| 4 | **Enable OVF** (`KERNELFORGE_OVF=1`) | Focus gradients on high-variance groups; G=8+OVF can match G=16 | Config change only |
| 5 | **Add WandB logging** (`report_to="wandb"`) | Track runs, compare experiments, share results | Small code change |
| 6 | **Expand to Ops-6K tasks** | Broader CUDA gen capability beyond WCC | Dataset routing change |
| 7 | **Try FP16** | 10 vs 7 mantissa bits; may prevent mid-training reward collapse | Config change, monitor stability |

### Tier 3: Scaling (After Learning Signal Confirmed)

| # | Change | Impact | Blocker |
|---|--------|--------|---------|
| 8 | **vLLM re-enablement** | 2-5x generation speedup | Blocked by transformers 5.x. Monitor vLLM releases |
| 9 | **causal-conv1d pre-built wheel** | 10-30% model inference speedup | Need to pre-build sm_80 wheel, host on private PyPI |
| 10 | **Stage 2 RFT** | Filter best trajectories, SFT on them | Needs 50+ steps of Stage 1 trajectories |
| 11 | **Stage 3 full GRPO** | T=0.7, deeper multi-turn rollouts | Needs Stage 1/2 checkpoints |
| 12 | **Curriculum progression** | Progressive difficulty across training | Needs multi-task support working first |
| 13 | **Northflank/CoreWeave migration** | Production deployment, persistent eval workers | Future work, Modal sufficient for now |

### Tier 4: Research Extensions (Future)

- KernelBench generalization beyond Ops-6K
- H100/B200/B300 retargeting
- Distillation of learned optimization behavior into smaller models
- SkyDiscover evolutionary search as parallel track
- Larger models (Qwen3-Coder-Next 80B on B200)

---

## 5. Key Lessons Learned

### What broke and why (Mar 8-10)

| Mistake | Impact | Lesson |
|---------|--------|--------|
| G=2 | Zero reward variance → zero gradients → no learning | Never go below G=4; G=8 is the safe default |
| LoRA r=16 | Insufficient capacity for code generation | r=64 minimum for code gen tasks |
| 4-bit QLoRA | Model quality too low for valid CUDA code | Use bf16 until correctness is proven |
| Chat-format prompts | Type mismatches throughout pipeline | Keep raw string prompts; chat format is cosmetic |
| max_completion_length=256 | 100% truncation before complete kernels | 1024 minimum for CUDA kernels |
| Changing 6 things at once | Impossible to diagnose which change broke training | One change at a time, smoke test after each |

### What we got right

| Decision | Why it was correct |
|----------|-------------------|
| TRLOO correction | 14-33% stronger gradients, proven in literature |
| Discrete milestone rewards | Robust to timing noise, clear gradient curriculum |
| beta=0.0 (no KL) | Saves VRAM, matches DAPO findings for sparse reward |
| LoRA r=64 | Sufficient capacity for code gen per literature |
| Smoke-test-first policy | Caught every issue before wasting GPU hours |
| Full revert over incremental fix | When 6 things break at once, revert is faster than debugging |

---

## 6. Current Config Reference

```python
# modal_train.py — Stage 1 defaults (v2.0)
KERNELFORGE_MODEL_LABEL = "opus_9b"
KERNELFORGE_LORA_R = "64"
KERNELFORGE_LORA_ALPHA = "64"
KERNELFORGE_QUANT_BITS = "0"  # bf16
KERNELFORGE_SKIP_UNSLOTH = "1"
KERNELFORGE_USE_VLLM = "0"
KERNELFORGE_USE_TRLOO = "1"
KERNELFORGE_EVAL_BACKEND = "modal"
KERNELFORGE_STAGE1_NUM_GENERATIONS = "8"
KERNELFORGE_STAGE1_PER_DEVICE_BATCH_SIZE = "1"
KERNELFORGE_STAGE1_GRADIENT_ACCUMULATION_STEPS = "8"
KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH = "1024"
KERNELFORGE_STAGE1_MAX_STEPS = "100"
KERNELFORGE_STAGE1_MAX_TURNS = "1"
KERNELFORGE_STAGE3_BETA = "0.0"
```

```bash
# Launch commands
uv run modal run --detach modal_train.py --stage 0          # smoke test
uv run modal run --detach modal_train.py --stage 1 --max-steps 1   # 1-step validation
uv run modal run --detach modal_train.py --stage 1 --max-steps 50  # real run
```
