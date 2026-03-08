# Review: Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled for KernelForge

## Model Under Review

**[Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF](https://huggingface.co/Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF)**

| Property | Value |
|----------|-------|
| Base | Qwen3.5-2B (dense) |
| Parameters | 2B total, 2B active |
| Architecture | Dense transformer (NOT MoE) |
| Distillation source | Claude 4.6 Opus reasoning chains |
| Training method | LoRA rank 64 via **Unsloth** (`train_on_responses_only`) |
| Training data | ~3,280 samples (CoT distillation) + Qwen3.5-27B reasoning trajectories |
| Reasoning format | Structured `<think>` tags |
| Quantization | GGUF (multiple quants available) |
| Context | 256K tokens (inherited from Qwen3.5) |
| Collection downloads | ~57K+ across 13 variants |

## Current Primary Model (Baseline Comparison)

**Qwen/Qwen3-Coder-30B-A3B-Instruct** (configured in `training/model_loader.py`)

| Property | Value |
|----------|-------|
| Parameters | 30.5B total, 3.3B active (MoE) |
| Architecture | Sparse MoE (256 experts, 8 routed + 1 shared) |
| Reasoning | Non-reasoning (direct responses, no CoT) |
| VRAM (bf16) | ~61GB on H200 141GB |
| Unsloth support | Yes (Faster MOE 2026 update) |
| Intelligence Index | 20 (Artificial Analysis) |

## Why the 2B Distilled Model Is Interesting for KernelForge

### 1. VRAM Budget: 61GB -> ~4GB

The current Qwen3-Coder-30B-A3B at bf16 consumes ~61GB on H200, leaving ~80GB for vLLM + GRPO rollouts. Swapping to a 2B dense model at bf16 would consume **~4GB**, freeing **~137GB** for:
- Larger batch sizes (currently batch=1 with grad_accum=4)
- Higher G (generation count) for GRPO — currently G=2, could go to G=8 or G=16
- Enabling `use_vllm=True` (currently disabled for hackathon)
- Longer `max_completion_length` (currently 768, could go to 2048+)
- Running multiple rollouts in parallel

### 2. Native Reasoning via `<think>` Tags

KernelForge's multi-turn rollout (`training/multi_turn_rollout.py`) generates CUDA code and receives feedback over 3 turns (Dr. Kernel MAX_TURN=3). The distilled model's structured `<think>` reasoning is directly beneficial here:
- Step-by-step kernel optimization planning before code emission
- Better error analysis when receiving compilation/correctness feedback
- More principled approach to A100-specific optimizations (cp.async, L2 persistence, etc.)

Qwen3-Coder-30B-A3B is a **non-reasoning model** — it gives direct responses without CoT. For iterative CUDA kernel refinement, structured reasoning is a significant advantage.

### 3. Unsloth Compatibility Is Native

The model was **trained with Unsloth** (LoRA rank 64). KernelForge already uses Unsloth's `FastLanguageModel` + `PatchFastRL("GRPO")`. The integration path is clean:
- Same `FastLanguageModel.from_pretrained()` loader
- Same LoRA target modules (q/k/v/o_proj, gate/up/down_proj)
- Same `PatchFastRL("GRPO")` patching
- **No MoE routing complexity** — dense model simplifies everything

### 4. Distillation Quality > Raw Parameter Count

The model distills Claude 4.6 Opus reasoning patterns into only 3,280 samples. Key training choices:
- `train_on_responses_only` — loss only on `<think>` sequences and final solutions
- Additional reasoning trajectories from Qwen3.5-27B (higher-quality teacher chains)
- Focused on reducing redundant cognitive loops while preserving analytical depth

For CUDA kernel generation specifically, **reasoning quality matters more than parameter count**. A 2B model that can plan "use shared memory tiling -> coalesce global loads -> exploit L2 persistence" before writing code may outperform a 30B model that just pattern-matches.

## Comparison: Qwen3.5-2B Distilled vs Qwen3-Coder-30B-A3B vs Qwen3.5-35B-A3B

| Dimension | Qwen3.5-2B Distilled | Qwen3-Coder-30B-A3B | Qwen3.5-35B-A3B |
|-----------|----------------------|----------------------|-------------------|
| Total params | **2B** | 30.5B | 35B |
| Active params | **2B** | 3.3B | 3B |
| Architecture | Dense | MoE | MoE (Gated Delta + MoE) |
| VRAM (bf16) | **~4GB** | ~61GB | ~70GB |
| Reasoning | CoT via `<think>` | None | Native thinking mode |
| Intelligence Index | N/A (community) | 20 | 37 |
| Code specialization | General (distilled reasoning) | Code-focused | General multimodal |
| Inference speed | **Fastest** (dense 2B) | Slow (~25 tok/s API) | Fast (~148 tok/s API) |
| GRPO G scaling | **G=8-16 feasible** | G=2 (VRAM limited) | G=1-2 (VRAM limited) |
| Unsloth native | Yes (trained with it) | Yes (Faster MOE) | Yes |
| Release | 2026 | Jul 2025 | Feb 2026 |

## How to Apply CUDA Agent / Dr. Kernel Techniques

### Dr. Kernel Integration Points

Dr. Kernel (arXiv:2602.05885) techniques already in KernelForge that directly benefit from the 2B model:

1. **TRLOO advantage scaling** (`custom_grpo_trainer.py`): With G=2, TRLOO corrects 50% gradient shrinkage. At G=8+ (feasible with 2B), the correction factor approaches 1.0 and GRPO becomes much more stable.

2. **Multi-turn rollout** (`multi_turn_rollout.py`): The `<think>` reasoning format naturally structures the 3-turn kernel refinement loop. Turn 1: initial kernel with reasoning. Turn 2: analyze feedback, reason about fix. Turn 3: final optimized version.

3. **Anti-hack suite** (`anti_hack.py`): Reasoning models are less likely to produce degenerate "hack" kernels (constant output, passthrough, noop) because the `<think>` process makes the model plan actual optimizations.

4. **Curriculum phases** (`curriculum.py`): The 4-phase curriculum (single_ops -> fusion_2op -> arch_specific -> advanced) benefits from a model that can reason about increasing complexity rather than just memorize patterns.

### CUDA Agent (Ops-6K) Integration

The CUDA Agent dataset (`BytedTsinghua-SIA/CUDA-Agent-Ops-6K`) loaded via `training/cuda_agent_integration.py` provides reference PyTorch code that the model must translate to optimized CUDA. The 2B distilled model's approach:

- **Reasoning step**: Analyze the PyTorch op, identify memory access patterns, determine A100-specific optimizations
- **Code step**: Emit CUDA kernel with architecture-aware choices
- **Feedback loop**: Use structured thinking to interpret eval_service results and iterate

## Implementation Plan

### Option A: Drop-in Replacement (Recommended for Pilot)

Minimal changes to `training/model_loader.py`:

```python
# In model_loader.py, change:
PRIMARY_MODEL = os.getenv(
    "KERNELFORGE_MODEL",
    "Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled"
)

# Adjust LoRA to match distillation config:
LORA_R = 64       # was 16 — match distillation rank
LORA_ALPHA = 64   # was 16
```

And in `_load_primary()`, since this is a dense model:
- `load_in_4bit=False` stays (bf16 is fine, only ~4GB)
- `fast_inference=True` can be enabled (dense model, no MoE complexity)
- `use_vllm=True` becomes feasible

### Option B: Dual-Model Configuration

Keep Qwen3-Coder-30B-A3B for Stage 2 (RFT/SFT) and use the 2B distilled model for Stage 1 and Stage 3 (GRPO), where the VRAM savings enable higher G and more exploration.

### Option C: GGUF Inference + Full-Precision Training

Use the GGUF quantized model for fast inference/generation during rollouts (via llama.cpp or similar), but train LoRA adapters on the full-precision Qwen3.5-2B base. This splits the compute:
- **Generation**: GGUF Q4_K_M (~1.5GB) for maximum throughput
- **Training**: bf16 LoRA on base Qwen3.5-2B (~4GB)

### Key Config Changes for Any Option

```python
# Stage 3 (stage3_grpo.py) — exploit VRAM savings:
G = 8                          # was 2 — more generations per prompt
MAX_COMPLETION_LENGTH = 2048   # was 768 — room for <think> + code
USE_VLLM = True                # was False — feasible now
BATCH_SIZE = 4                 # was 1 — 4x throughput
GRAD_ACCUM = 2                 # was 4 — effective batch 8
```

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| 2B too small for complex CUDA patterns | Medium | Reasoning compensates; curriculum starts easy; 3-turn refinement |
| GGUF quantization degrades generation | Low | Unsloth warns against QLoRA for Qwen3.5, but GGUF inference-only is fine |
| `<think>` tags interfere with code extraction | Low | `extract_cuda_code()` already parses ```cuda blocks; `<think>` content is separate |
| Community model, not official Qwen | Medium | Validate on held-out Ops-6K tasks before full training run |
| Reasoning overhead adds tokens | Low | 2B is fast enough; net win from better kernel quality |

## Verdict

**The Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled model is a strong candidate for KernelForge**, especially for the GRPO stages. The core argument:

1. **10x VRAM savings** -> higher G, bigger batches, vLLM enabled -> more stable GRPO training
2. **Structured reasoning** -> better multi-turn kernel refinement (matches Dr. Kernel's iterative approach)
3. **Native Unsloth** -> zero friction integration with existing pipeline
4. **Dense architecture** -> simpler, faster, no MoE routing edge cases

The main question is whether 2B parameters are sufficient for complex CUDA kernel generation (Phase 3: Flash-Attention, Batched SpMV). The recommendation is to **pilot with Option A** on Phase 0-1 curriculum tasks (vector_add, relu, softmax, matmul) and measure compile_rate + correct_rate against the current 30B baseline before committing to full training.

## Sources

- [Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF](https://huggingface.co/Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled-GGUF)
- [Qwen3.5-Claude-4.6-Opus-Reasoning-Distilled Collection](https://huggingface.co/collections/Jackrong/qwen35-claude-46-opus-reasoning-distilled)
- [Unsloth Qwen3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Unsloth Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- [Dr. Kernel Paper (arXiv:2602.05885)](https://arxiv.org/abs/2602.05885)
- [Qwen3.5-35B-A3B Intelligence Analysis](https://artificialanalysis.ai/models/qwen3-5-35b-a3b)
- [Qwen3 Coder 30B A3B Analysis](https://artificialanalysis.ai/models/qwen3-coder-30b-a3b-instruct)
