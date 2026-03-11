# A100 CUDA RL Optimization Plan

This document is a legacy reference for an earlier experimental branch. The active implementation path is defined by `docs/KERNELFORGE_FINAL_PRD.md` and `training.grpo_train`, with Stage 1 / Stage 2 / Stage 3 executed through the repo’s current training stack.

## Executive Summary

This document outlines the complete optimization strategy to improve the A100 CUDA RL system from 0% kernel compilation success to viable performance. The approach follows proven research (CUDA-L1) and leverages real DoubleGraph kernel patterns.

## Current State Analysis

### What's Working ✅
- TRLOO advantage correction fixing 25% gradient bias
- 9B Qwen3.5 model loads with LoRA adapters  
- Evaluation pipeline compiles and tests kernels
- Reward signal flows (dense ladder: -1.0 to +1.0)
- End-to-end training completes

### Critical Failures ❌
- **0% kernel compilation success**
- 75-100% token truncation at 1024 limit
- No positive rewards generated
- Model produces explanations instead of code
- Loss collapses to near-zero after first step

## Root Cause

The model lacks supervised CUDA examples before RL optimization. Current approach is like teaching optimization to someone who doesn't know the language.

## Solution: Three-Stage Pipeline

### Stage 0: Supervised Fine-Tuning (SFT)
**Goal**: Teach model CUDA syntax and A100 patterns

**Approach**:
- Use real kernel examples from DoubleGraph repository
- Create curriculum from simple → complex patterns
- Focus on compilation success over optimization

**Expected Results**:
- Compilation rate: 0% → 60-80%
- Basic CUDA syntax learned
- A100-specific patterns understood

**Implementation**:
```bash
# Preflight the active training stack
uv run python -m training.grpo_train --preflight-only

# Run the active training stages
uv run python -m training.grpo_train --stage stage1
uv run python -m training.grpo_train --stage stage2
uv run python -m training.grpo_train --stage stage3
```

### Stage 1: Enhanced GRPO with Contrastive Learning
**Goal**: Optimize for performance using RL

**Approach**:
- Build on SFT foundation
- Use TRLOO advantage correction
- Add contrastive learning (compare multiple variants)
- Performance feedback in prompts

**Expected Results**:
- Positive reward rate: 0% → 20-40%
- Speedup achievement: 0% → 10-20%
- Compilation rate maintained: 60-80%

**Key Enhancements**:
1. **Contrastive Prompts**: Show model multiple variants with performance scores
2. **Enhanced Reward Function**: Relative performance bonuses
3. **A100-Specific Context**: 40MB L2, 108 SMs guidance

### Stage 2: Advanced Optimization (Future)
**Goal**: Achieve production-level performance

**Approach**:
- Multi-turn refinement
- Architecture-specific tuning
- Advanced memory patterns

## Research Backing

### CUDA-L1 Paper Results
- **3.12× average speedup** on KernelBench
- **120× maximum gains** on specific kernels
- **99.6-100% success rates** across configurations

### Key Techniques Applied
1. **Three-stage pipeline** (SFT → Self-supervised → Contrastive RL)
2. **Performance feedback in prompts** (model learns from comparisons)
3. **Architecture-specific optimization** (A100 L2 cache, SM count)

## Implementation Details

### Data Preparation

#### SFT Dataset Structure
```json
{
  "prompt": "Write CUDA kernel for A100 to...",
  "completion": "extern \"C\" __global__ void kernel(...)",
  "kernel_id": "a100/category/algorithm",
  "curriculum_level": 0
}
```

#### Curriculum Levels
- **Level 0**: Simple operations (vector ops, basic kernels)
- **Level 1**: Medium complexity (graph traversal, reductions)
- **Level 2**: Complex algorithms (PageRank, Louvain)

### Model Architecture

#### Base Model: Qwen3.5-9B-Opus-Distilled
- 9B parameters (good balance of performance/speed)
- Strong code generation capabilities
- Fits in A100 40GB with LoRA

#### Training Configuration
```yaml
SFT:
  batch_size: 4
  learning_rate: 2e-5
  epochs: 3
  max_length: 2048

GRPO:
  batch_size: 4
  learning_rate: 1e-6
  max_steps: 50
  num_generations: 4 (G=4)
  beta: 0.01 (KL penalty)
```

### Evaluation Pipeline

#### Reward Structure
```python
def reward_function(eval_result):
    if not compiles: return -0.4
    if not correct: return -0.2
    if speedup_vs_compile > 1.05: return 1.0
    if speedup_vs_eager > 1.05: return 0.7
    if speedup_vs_eager >= 0.95: return 0.4
    return 0.2
```

#### Contrastive Enhancement
```python
def contrastive_bonus(reward, other_rewards):
    avg_other = np.mean(other_rewards)
    if reward > avg_other:
        return 0.1 * (reward - avg_other)
    return 0
```

## Expected Improvements

### Success Metrics

| Metric | Before | After SFT | After GRPO |
|--------|--------|------------|-----------|
| Compilation Rate | 0% | 60-80% | 60-80% |
| Positive Rewards | 0% | 0% | 20-40% |
| Speedup Achievement | 0% | 0% | 10-20% |
| Token Truncation | 75-100% | <20% | <20% |

### Learning Progression

1. **SFT Phase**: Model learns to write compilable CUDA
2. **GRPO Phase**: Model learns to optimize for performance
3. **Contrastive Learning**: Model learns from comparative examples

## Running the Pipeline

### Prerequisites
```bash
# Clone DoubleGraph source
git clone https://github.com/double-ai/doubleGraph.git doublegraph_src

# Install dependencies
pip install -r requirements.txt
```

### Full Pipeline
```bash
# Run the full supported pipeline
uv run python -m training.grpo_train --stage pipeline

# Or use the compatibility wrapper
python scripts/run_full_pipeline.py --stage pipeline
python scripts/run_full_pipeline.py --stage stage2
python scripts/run_full_pipeline.py --stage eval
```

### Monitoring
- **Modal Dashboard**: https://modal.com/apps/ocwc22/main
- **Weights & Biases**: Integrated for experiment tracking
- **Local Logs`: Check outputs/training_report.json`

## Risks and Mitigations

### Risk 1: SFT Overfitting
- **Mitigation**: Mix simple and complex kernels
- Use data augmentation (code transformations)

### Risk 2: Reward Hacking
- **Mitigation**: Robust correctness checks (1000 random inputs)
- Multi-metric rewards (speedup + efficiency)

### Risk 3: Computational Cost
- **Mitigation**: Start with smaller model (2B) for testing
- Use spot instances where possible

## Future Directions

### Advanced Techniques
1. **Self-supervised Learning**: Masked CUDA code completion
2. **Multi-turn Optimization**: Iterative refinement
3. **Architecture Transfer**: Adapt to H100/B200

### Production Readiness
1. **Comprehensive Testing**: 1000 random inputs per kernel
2. **Performance Profiling**: Nsight, cuProf integration
3. **Deployment**: Containerized evaluation service

## References

1. [CUDA-L1 Paper](https://arxiv.org/abs/2507.14111) - Contrastive RL for CUDA optimization
2. [DoubleGraph Repository](https://github.com/double-ai/doubleGraph) - A100-optimized graph kernels
3. [TRLOO Paper](https://arxiv.org/abs/2406.18636) - Advantage correction for GRPO

## Conclusion

This optimization plan transforms the current 0% success rate into a viable CUDA generation system by:
1. **Teaching fundamentals first** (SFT)
2. **Optimizing with proven RL techniques** (GRPO + Contrastive)
3. **Leveraging real A100 patterns** (DoubleGraph)

The expected outcome is a 60-80% compilation success rate with 10-20% speedup achievement - a significant improvement over the current state.
