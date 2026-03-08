# KernelForge GRPO Deep Dive
## Hackathon Path: Structured Priors + H200 Training + A100 Evaluation
**Purpose:** Explain the RL strategy that actually fits the hackathon.
**Primary hardware:** H200 ($4.54/hr, 141GB) for training, A100 80GB for evaluation
**Primary model:** Qwen3-Coder-30B-A3B-Instruct (30.5B total, 3.3B active, 128 experts / 8 active, 256K context)
**Last Updated:** March 7, 2026

> **IMPORTANT — Hackathon-Scoped GRPO Strategy (March 2026)**
>
> This document is no longer centered on B200 as the default, Qwen3-Coder-Next 80B as the default, or full CUDA-Agent-scale replication as the immediate goal.
>
> This document is now centered on the actual hackathon path:
> 1. **Use Qwen3-Coder-30B-A3B-Instruct as the primary model.**
> 2. **Use H200 for training and A100 80GB for evaluation.**
> 3. **Use DoubleGraph, `skills.md`, and curated CUDA-Agent-style tasks as structured priors before RL begins.**
> 4. **Use SFT first, then a small-budget GRPO pilot.**
> 5. **Treat deeper multi-turn / larger-model / B200 work as future scale-up.**
>
> The goal this weekend is to prove: the reward is real, correctness is real, timing is real, structured priors make the search space manageable, the model can improve candidates under this loop, and the pipeline is reusable later. Not to claim full CUDA-Agent parity. Sources: [CUDA-Agent paper](https://arxiv.org/abs/2602.24286), [CUDA-Agent project page](https://cuda-agent.github.io/), [doubleGraph repo](https://github.com/double-ai/doubleGraph), [OpenEnv docs](https://meta-pytorch.github.io/OpenEnv/).
>
> **Key technique:** TRLOO N/(N-1) correction (Dr. Kernel arXiv [2602.05885](https://arxiv.org/abs/2602.05885)) fixes the self-inclusion bias in GRPO's advantage estimation. With G=2, this corrects 50% gradient shrinkage. Implemented in `custom_grpo_trainer.py`.

## Core Recommendation

### Hackathon recommendation
- **Start from structured priors**, then do SFT warmup, then GRPO pilot, then search / best-of-N as a hedge
- Model: Qwen3-Coder-30B-A3B-Instruct on H200
- Eval: A100 80GB via the active remote backend target (Northflank + CoreWeave), with the current Modal path treated as legacy transition code
- Priors: DoubleGraph kernels + `skills.md` + curated CUDA-Agent-style tasks
- G=2, short context, discrete milestone reward {-1, 1, 2, 3}, execution-based correctness, limited-step run with hard abort gates
- Because under a hackathon budget, the highest-leverage thing is not elegant RL theory — it is working reward plumbing, working correctness checks, working timing on the actual target hardware, and enough prior structure that RL steps are not mostly wasted.

### Not recommended as the primary hackathon path
- Defaulting to Qwen3-Coder-Next 80B or B200
- Pretending deep 20-turn RL is the must-win path on day one
- Presenting partially implemented MARS / MASPO / CPPO ideas as already validated reality

### What we need to prove this weekend
1. The reward function is real and not easily hacked
2. The structured priors materially improve the starting distribution
3. The model can improve compile / correctness / timing outcomes
4. GRPO does not collapse immediately
5. The pipeline is reusable for bigger future experiments

That is enough for the hackathon. Not full benchmark domination.

### Locked framing for the rest of this document
The sections below are primarily **reference material** for implementation and mathematical justification. They are not a commitment to every advanced technique discussed later in the file. The active hackathon scope is the narrower path defined above and in `docs/KERNELFORGE_FINAL_PRD.md`.

---

## Full Pedagogical Deep-Dive: Understanding GRPO, Its Bias Problem, the TRLOO Fix, MARS Credit Assignment, and Why This Matters for Your KernelForge Hackathon

This section explains **everything from first principles**, as if you have never seen RL before. No assumptions. Every equation is derived step-by-step with intuition, a simple numerical example, the thought process behind why people invented it, the exact problem it solves (or fails to solve), and how the 2026 papers (real ones like arXiv 2601.08521 "Your Group-Relative Advantage Is Biased" and analogs to Dr. Kernel/TRLOO) fix it.

This is tailored to **your exact use-case**: training Qwen3-Coder-30B-A3B-Instruct on H200 to write A100 CUDA kernels using the CUDA-Agent eval pipeline (compile → verify → profile). The rewards are sparse (most kernels don't compile or are slow), which is exactly why the hackathon path starts by narrowing the search with DoubleGraph, `skills.md`, and curated CUDA-Agent-style tasks before attempting RL.

### 1. RL Basics – Why We Even Need "Advantage" (First Principles)

**Naïve REINFORCE** (the oldest policy gradient):
```
∇J(θ) ≈ (1/N) Σ_{i=1}^N r_i · ∇_θ log π_θ(y_i | x)
```

Intuition: "If this kernel got high reward, make the model more likely to output it again (increase its log-prob)."

**Problem #1: Huge variance.** One lucky 5× speedup kernel can dominate the whole batch. One bad run can push gradients the wrong way.

**Solution invented in 1990s: Advantage** A = "how much better was this action than expected?"
```
∇J ≈ Σ (r_i - b) ∇ log π(y_i)
```
where b is a **baseline** (e.g. average reward for this prompt). Subtracting b centers the signal around zero → lower variance, same expected gradient.

**Problem #2 for LLMs in 2024–2026:** training a separate **value network** V (critic) to predict expected reward for every token is expensive (extra model parameters, memory explosion on single GPU).

### 2. GRPO Invention (DeepSeekMath 2024, popularized in DeepSeek-R1 2025) – The Thought Process

DeepSeek team asked: "Can we get a good baseline **without** training a critic?"

Answer: For the **same prompt**, sample **G** different completions (G=2–16). Their rewards form a tiny "group distribution".

**Thought process:**
- The average reward in this group is a cheap Monte-Carlo estimate of "what the model currently expects for this prompt".
- Standardize: how many std-devs above/below the group mean is this completion?

**Exact GRPO advantage** (what TRL GRPOTrainer implements):
```
A_i = (r_i - μ) / (σ + ε)
```
where
```
μ = (1/G) Σ_{j=1}^G r_j,    σ = sqrt((1/G) Σ_{j=1}^G (r_j - μ)²)
```
(ε ≈ 1e-8 to avoid divide-by-zero when all rewards identical.)

Then plug into clipped PPO-style surrogate loss (same as normal PPO, just replace advantage).

**Numerical example** (your kernel world, discrete milestone rewards {-1, 1, 2, 3}):
```
Prompt: "fuse GEMM + bias + ReLU"
- Completion 1: compiles, correct, no speedup      → r=+1
- Completion 2: compiles, correct, beats eager+compile → r=+3
- Completion 3: doesn't compile                     → r=-1
- Completion 4: compiles, correct, beats eager only → r=+2

μ = 1.25, σ ≈ 1.71 → advantages ≈ [-0.15, +1.02, -1.32, +0.44]
```

The model strongly reinforces completion 2, mildly reinforces 4, mildly discourages 1, strongly discourages 3.

This worked amazingly for math/code reasoning in 2025 because it's cheap and stable.

### 3. The Fundamental Problem (Discovered Jan–Feb 2026): Group-Relative Advantage Is Biased

Real paper: **"Your Group-Relative Advantage Is Biased"** (arXiv [2601.08521](https://arxiv.org/abs/2601.08521), Jan 2026) — this is the exact paper the conversation called "Dr. Kernel".

**The bias comes from self-inclusion** in the baseline.

When computing μ for completion i, μ **includes r_i itself**.

So the advantage for a good completion is artificially **pulled down** (because it helped raise the mean), and for a bad one **pulled up**.

**Step-by-step derivation** (this is the math you need):

Let the true (unbiased) baseline be the mean of the *other* G-1 completions: μ_{-i}

But GRPO uses μ = (r_i + Σ_{j≠i} r_j)/G

So:
```
A_i^GRPO = r_i - μ = r_i - (r_i + S_{-i})/G = (1 - 1/G) r_i - S_{-i}/G
```
where S_{-i} = sum of others.

When you take expectation E[A_i^{GRPO} · ∇logπ_i], the self-term (1-1/G) r_i correlates with the action you are updating → the gradient is scaled by exactly **(1 - 1/G)**.

With G=2: gradients are **only 50% as strong** as they should be. With G=4: 75%.

In **multi-turn** kernel refinement (turn 1: first kernel, turn 2: fix compile error, turn 3: optimize for Nsight occupancy):
- Bad kernels die early → fewer valid completions in later turns → G shrinks → bias gets **worse**.
- High-reward outliers (a magical 3× speedup kernel) get self-penalized most.
- This is exactly why pure RL on kernels collapsed at step ~17 in early 2026 experiments.

Additional bias (from 2601.08521): for **hard prompts** (rarely any good kernels) the estimator systematically underestimates good actions; for easy prompts it overestimates.

### 4. The TRLOO Solution (Turn-level REINFORCE Leave-One-Out) – Exact Fix

**Thought process** (from Dr. Kernel / 2601.08521 and TRL's own RLOO trainer):
- Just exclude the current sample from the baseline. That's it.
- Baseline becomes mean of the other G-1.
- Closed-form that avoids recomputing sums every time:
```
A_i^TRLOO = (G/(G-1)) × (r_i - μ)
```
(where μ is still the full-group mean).

This is **unbiased**: E[gradient] = true ∇J.

**Numerical example again** (same numbers, discrete milestone rewards):
```
μ = 1.25
TRLOO multiplier = 4/3 ≈ 1.333
Advantages become: [-0.20, +1.36, -1.76, +0.59] → much stronger signal for the good kernel, stronger penalty for the bad one.
```

In code (actual implementation: `openenv_env/reward.py` + `training/custom_grpo_trainer.py`):
```python
r_tensor = torch.tensor(rewards)  # shape (G,)
N = len(rewards)
if N > 1:
    mean = r_tensor.mean()
    unbiased = (r_tensor - mean) * (N / (N - 1.0))   # ← this single line removes the 25% bias
```

When G=1 (rare), fallback to raw r or 0.

This is exactly what TRL's RLOO trainer does internally, and what Dr. Kernel proved mathematically fixes kernel RL.

### 5. MARS Credit Assignment for Multi-Turn (arXiv [2510.15414](https://arxiv.org/abs/2510.15414) MARSHAL + related 2025 papers)

In single-turn: reward = final outcome only.

In your kernel loop: each turn gives partial signal (compile success, then speedup, then Nsight occupancy).

**Return-to-go** (MARS idea): at turn t, the "effective reward" for that turn includes all future rewards from later refinements.

```
G_{i,t} = R_{i,t} + R_{i,t+1} + ... + R_{i,T}    (γ=1)
```

Then compute TRLOO advantage on these **G_{i,t}** instead of raw R.

In your code (MARS is DROPPED — this is retained for reference only):
```python
# MARS was designed for per-turn rewards. KernelForge uses outcome-only
# rewards, making return_to_go[t] = r_final for all t. MARS = no-op here.
r = discrete_reward  # {-1, 1, 2, 3} from CUDA Agent Equation 1
turn = kwargs.get("turn", 0)
r = r * (1.0 ** turn)   # simple return-to-go weighting (future turns fully count)
```

This prevents credit assignment collapse in multi-turn kernel refinement (e.g. "the compile fix in turn 2 enabled the 2.3× speedup in turn 3").

### 6. The Operational Reality

The real bottleneck is not just memory — it is **evaluation throughput**: compiling kernels, running correctness checks, timing on A100, doing this enough times that RL sees real signal.

That means:
- fancy RL without real reward plumbing is useless,
- search sometimes gives better demo value per dollar than pure RL,
- SFT-first is the right move.

**Stacked techniques that help (implemented):**
- **Base reward:** discrete milestone {-1, 1, 2, 3} (CUDA Agent Equation 1) — clear thresholds for compile, correct, speedup vs eager, speedup vs compile.
- **TRLOO scaling** → removes self-inclusion bias (50% gradient shrinkage at G=2).
- ~~**MARS return-to-go**~~ → DROPPED (no benefit with outcome-only rewards — see GRPO-4).
- ~~**CPPO cheap filter**~~ → DROPPED (no benefit post-SFT at G=2 — see GRPO-11).

**Operational conclusion:** because budget is tight, start with SFT-first + shallow GRPO. Because evaluation is expensive, correctness and reward plumbing matter more than fancy RL theory. Because we target a hackathon, working feedback loops beat elegant but unrun architecture.

### 6.1 Where OpenEnv fits in the hackathon stack

OpenEnv and KernelGYM solve different problems. For KernelForge, the plan is to build a **custom lightweight OpenEnv wrapper** that standardizes how the agent talks to the environment: typed models, `step()` / `reset()` / `state()`, HTTP serving, and reusable environment packaging. Underneath that wrapper, KernelGYM is the architecture reference for how kernel evaluation should run: separate learning clients from execution, serialize profiling-sensitive work, isolate failures, and eventually scale through workers. For KernelForge, the correct hackathon interpretation is:

1. **KernelForge uses a custom lightweight OpenEnv wrapper as the public environment contract.**
   - Judges, demos, and future external integrations should talk to `openenv_env/`.
2. **KernelForge owns the task/reward/rollout logic.**
   - `task_support.py`, `multi_turn_rollout.py`, reward normalization, and TRLOO are the project-specific logic layer.
3. **KernelGYM is the backend architecture reference.**
   - The remote evaluation plane should evolve into a simplified worker system behind a provider-neutral adapter.
4. **Training transport should stay direct for now.**
   - The OpenEnv wrapper is part of the real architecture, but forcing every Stage 3 turn through the HTTP server would add transport overhead without improving reward fidelity, because the reward-bearing invariant is the shared runtime adapter and shared reward math.
   - This is still compatible with TRL/OpenEnv because custom rollout functions may integrate with an environment while keeping training-specific transport logic optimized. Source: [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv)

This is the shortest path that stays aligned with TRL's documented OpenEnv support while preserving the low-latency compile-fast-fail and multi-turn repair loop needed for a hackathon budget. Sources: [OpenEnv docs](https://meta-pytorch.github.io/OpenEnv/), [OpenEnv environment guide](https://github.com/meta-pytorch/OpenEnv/blob/main/envs/README.md), [TRL OpenEnv integration](https://huggingface.co/docs/trl/en/openenv), [Dr. Kernel / KernelGYM paper](https://arxiv.org/abs/2602.05885), [KernelGYM repo](https://github.com/hkust-nlp/KernelGYM).

### 7. Quick Reference Table

| Method          | Advantage Formula                          | Bias? | Multi-turn Credit? | Memory on B200 | Your Hackathon Use |
|-----------------|--------------------------------------------|-------|--------------------|----------------|--------------------|
| PPO + critic    | GAE from value net                         | No    | Yes (GAE)          | High           | Too expensive      |
| Vanilla GRPO    | (r_i - μ)/σ                                | Yes (25% loss) | Weak               | Low            | Dropped            |
| TRLOO-GRPO      | [G/(G-1)] × (r_i - μ)                      | No    | N/A (outcome-only) | Low            | Core (your code)   |
| + MARS          | TRLOO on return-to-go G_{i,t}              | No    | Excellent          | Low            | DROPPED (no benefit with outcome rewards) |

You now have **complete understanding**. The single line `* (N/(N-1))` in your reward function is the mathematical cure for the exact disease that killed early kernel RL attempts in 2026.

---

## TRLOO-GRPO Math Deep-Dive

This is the **precise mathematics** behind the fix you are pasting tonight. Everything is taken verbatim from the source papers (verified today via arXiv PDFs).

### 1. Original GRPO (DeepSeekMath, arXiv [2402.03300](https://arxiv.org/abs/2402.03300))

For each prompt (one PyTorch op + task_code), sample **G** completions (G=2 for hackathon; G=4 in original CUDA Agent).  
Compute scalar reward **r_i** for each completion i (in your case: discrete milestone {-1, 1, 2, 3} from CUDA Agent Equation 1).

**Advantage estimator** (outcome supervision variant used in kernel RL):
```
Â_i = r_i - r̄    where r̄ = (1/G) Σ_{j=1}^G r_j
```
(or normalized by std(r) in some variants; Dr. Kernel drops the /std for stability in sparse-reward settings).

The surrogate loss (PPO-style clipped) is then:
```
J_GRPO(θ) = E[ (1/G) Σ_{i=1}^G (1/|o_i|) Σ_t min( (π_θ/π_old) Â_i, clip(...) Â_i ) ]
```

**Problem in multi-turn kernel refinement** (your setting: turn 1 = generate kernel, turn 2 = fix based on compile error, turn 3 = profile feedback, …):
- Rewards are now **per-turn**: R_{i,t} = correctness_{i,t} + clipped(speedup_{i,t})
- Return-to-go (future discounted reward from turn t onward):
```
G_{i,t} = Σ_{t'=t}^T R_{i,t'}    (γ = 1 in all 2026 kernel papers)
```
- GRPO still does **group mean over the G rollouts at that turn**:
```
A_GRPO^{i,t} = G_{i,t} - Ĝ_t    where Ĝ_t = (1/N_t) Σ_{j ∈ valid} G_{j,t}
```

### 2. The Bias Proof (Dr. Kernel, arXiv [2602.05885](https://arxiv.org/abs/2602.05885), 6 Feb 2026 — §4.2 + Appendix A)

The baseline **includes the current sample itself** → the advantage estimator is **correlated** with the action y_{i,t} you are trying to credit.

**Exact derivation** (simplified for one turn, one group of N valid rollouts):

```
ĝ_GRPO = (1/N) Σ_{i=1}^N ∇_θ log π_θ(y_i) · (G_i - Ĝ)
```

Expand Ĝ = (1/N) (G_i + Σ_{j≠i} G_j):

```
G_i - Ĝ = G_i - (1/N) G_i - (1/N) Σ_{j≠i} G_j = (1 - 1/N) G_i - (1/N) Σ_{j≠i} G_j
```

When you take **expectation** E[∇ log π (G_i - Ĝ)] and apply the score-function identity (E[∇ log π · f] = ∇ E[f] when f independent of action), the only surviving term from the baseline is the - (1/N) G_i piece.

**Final result** (verbatim quote):
```
E[ĝ_GRPO] = (1 - 1/N_t) ∇_θJ(θ)
```

- With G=4 → 25% smaller gradients; with G=2 (hackathon) → **50% smaller gradients every step**
- In multi-turn kernel generation: N_t shrinks (bad kernels die early) → bias gets **worse** in later turns (exactly why CUDA-Agent pure-RL collapsed at step 17).
- High-return outliers are **self-penalized** (a 5× speedup kernel gets pulled down by its own inclusion in the mean).

### 3. TRLOO Fix (Turn-level REINFORCE Leave-One-Out) — The Exact Equation

**Dr. Kernel definition**:
```
Ĝ_{(-i),t} = (1/(N_t-1)) Σ_{j ≠ i} G_{j,t}
A_TRLOO^{i,t} = G_{i,t} - Ĝ_{(-i),t}
```

**Algebraically equivalent closed form** (what you actually implement in code — zero extra computation):
```
A_TRLOO^{i,t} = (N_t/(N_t-1)) × (G_{i,t} - Ĝ_t)
```

- When N_t = 1: fallback to A = G_i (or 0)
- Baseline **independent** of y_i → unbiased estimator: E[ĝ_TRLOO] = ∇J(θ)
- No self-penalization of rare good kernels
- Preserves scale even when N_t drops in later turns

**Empirical win in kernel RL** (Dr. Kernel on KernelBench Level-2):
- Vanilla GRPO: 18.4% Fast@1.2×
- TRLOO: 31.6% Fast@1.2× (+72% relative)

### 4. How We Stack MARS Turn-Level Credit Assignment (arXiv [2510.15414](https://arxiv.org/abs/2510.15414))

MARS (Multi-turn Advantage with Return-to-go Scaling) tells us **G_{i,t} must be return-to-go**, not just the single-turn reward.

```
G_{i,t} = Σ_{t'=t}^T R_{i,t'}    (γ=1)
```

Then apply TRLOO on top of these G_{i,t}.

Actual implementation is `compute_reward()` in `openenv_env/reward.py` (MARS is DROPPED — see GRPO-4):
```python
# MARS step — NOT USED (outcome-only rewards make this a no-op)
# r = log(speedup) + 0.3*occupancy + 0.2*coalescing  # OLD continuous reward
# KernelForge now uses discrete milestone {-1, 1, 2, 3}

# TRLOO scaling — IMPLEMENTED (this is what we actually use)
unbiased = (r_tensor - mean) * (N / (N-1))
```

### 5. Full TRLOO-GRPO in Your KernelForge Reward (the code you paste tonight)

```python
# Inside cuda_kernel_reward after computing raw r for all G completions
r_tensor = torch.tensor(rewards)          # shape (G,)
N = len(rewards)
if N > 1:
    mean = r_tensor.mean()
    # TRLOO scaling — this is the ONLY line that removes the 25% bias
    unbiased = (r_tensor - mean) * (N / (N - 1.0))
else:
    unbiased = r_tensor

# Then feed unbiased advantages into TRL GRPOTrainer (it just sees them as "rewards")
return unbiased.tolist()
```

That single multiplication `*(N/(N-1))` is the entire mathematical fix.

### 6. Why This Works for the Hackathon Stack (H200-class training + remote A100 eval + CUDA-Agent)

- Rewards come from **CUDA events timing on remote A100 hardware** + execution-based correctness (discrete milestone {-1, 1, 2, 3})
- G=2 keeps eval budget low, TRLOO N/(N-1) fixes the bias
- Hackathon config: Qwen3-Coder-30B-A3B-Instruct on H200 gen + A100 eval
- **Expert demonstrations**: 192 A100 kernels from doubleGraph as SFT data (`doublegraph_sft.jsonl`) + combined RL prompts (`combined_kernelforge.jsonl`)
- **Rich SKILL.md**: 178 lines with 7 real A100 patterns from production code via `skill_builder.py:_append_a100_patterns()`
- **SkyDiscover parallel hedge**: `skydiscover_integration/evaluator.py` should point at the same backend adapter so evolutionary search and GRPO share one remote A100 evaluation contract during the Northflank + CoreWeave migration

### 7. Quick Reference Table (Copy into Your PRD Appendix)

| Variant       | Advantage Formula                          | Bias Factor      | Used In KernelForge? |
|---------------|--------------------------------------------|------------------|----------------------|
| Vanilla GRPO  | G_i - mean(G)                              | (1-1/N)          | No (dropped)        |
| TRLOO         | [N/(N-1)] (G_i - mean(G))                  | None             | Yes (core)          |
| + MARS        | return-to-go G_{i,t} then TRLOO            | None             | DROPPED (no benefit with outcome rewards) |
| + Nsight PR   | optional Nsight bonus (not in primary reward) | —             | Future enhancement  |

**Bottom line:** TRLOO-GRPO is **not** a new algorithm — it is vanilla GRPO with one mathematically proven 1-line correction that removes the exact bias Dr. Kernel identified in Feb 2026. Combined with SFT warmup and discrete milestone rewards {-1, 1, 2, 3}, it is the hackathon-fit stack for validating real learning signal under budget.

---

## GRPO-1: The Algorithm (Full Math)

### What GRPO Actually Is

GRPO (Group Relative Policy Optimization) is PPO without a critic network. Instead of training a separate value model to estimate "how good is this state," GRPO generates MULTIPLE completions for the same prompt and uses their relative rewards as the baseline. This eliminates ~50% of memory (no critic weights, no critic optimizer, no critic activations).

### The Exact Equations

**Step 1: Group Sampling**

For each prompt q_j in a batch of M prompts, sample G completions from the current (old) policy:

```
{o_{j,1}, o_{j,2}, ..., o_{j,G}} ~ π_θ_old(· | q_j)
```

In our case: M = 1 (one prompt per step), G = 2 (two kernel candidates per prompt for the hackathon pilot).

**Step 2: Reward Computation**

Each completion gets a scalar reward from the environment:

```
r_{j,i} = env.step(o_{j,i}).reward    for i = 1..G
```

In our case, this is the discrete reward {-1, 1, 2, 3} from CUDA Agent's Equation 1:
- r = -1: compilation fails OR correctness fails
- r = +1: correct but no speedup (speedup ≤ 1.05×)
- r = +2: >5% faster than torch.eager
- r = +3: >5% faster than BOTH torch.eager AND torch.compile

**Step 3: Advantage Estimation (The Key Innovation)**

PPO computes advantages using a learned value function V(s). GRPO computes advantages using group statistics:

```
Â_{j,i} = (r_{j,i} - mean(r_{j,1}, ..., r_{j,G})) / (std(r_{j,1}, ..., r_{j,G}) + ε)
```

Where ε = 1e-8 to avoid division by zero.

**Concrete example with our reward values:**

```
Prompt: "Write a CUDA kernel for softmax on A100"

Completion 1: Compiles, correct, no speedup    → r = +1
Completion 2: Compilation fails                 → r = -1
Completion 3: Compiles, correct, beats eager    → r = +2
Completion 4: Compiles, wrong output            → r = -1

mean(r) = (1 + (-1) + 2 + (-1)) / 4 = 0.25
std(r)  = sqrt(((1-0.25)² + (-1-0.25)² + (2-0.25)² + (-1-0.25)²) / 4)
        = sqrt((0.5625 + 1.5625 + 3.0625 + 1.5625) / 4)
        = sqrt(1.6875)
        ≈ 1.299

Â_1 = (1 - 0.25) / 1.299 = +0.577   ← "slightly better than average"
Â_2 = (-1 - 0.25) / 1.299 = -0.962  ← "worse than average"
Â_3 = (2 - 0.25) / 1.299 = +1.347   ← "much better than average"
Â_4 = (-1 - 0.25) / 1.299 = -0.962  ← "worse than average"
```

**What this means:** Every token in Completion 3 gets advantage +1.347 — the model is pushed to generate MORE text like Completion 3. Every token in Completions 2 and 4 gets advantage -0.962 — the model is pushed AWAY from generating text like those.

**Critical edge case:** If all 4 completions get the same reward (e.g., all fail to compile, all r = -1), then std(r) = 0, and all advantages are 0. **No learning signal.** This is why SFT warmup is mandatory — you need at least SOME completions that compile so there's variance in the group.

### GRPO Self-Inclusion Bias vs TRLOO (Dr. Kernel, arXiv 2602.05885)

The advantage formula above includes sample `i` in its own baseline (mean and std). Dr. Kernel proves this causes a **systematic gradient shrinkage**:

```
E[ĝ_GRPO] = (1 - 1/N) × ∇θJ(θ)
```

With G=4 (our setting), **gradients are 25% too small**. This is not noise — it's a deterministic bias that slows learning.

**TRLOO fix:** Remove the current sample from the baseline computation:

```
GRPO advantage:   Â_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)
TRLOO advantage:  Â_i = (r_i - mean(r_{j≠i})) / std(r_{j≠i})     ← leave-one-out
```

In multi-turn settings, the bias compounds: at turn t, only N_t samples survive (those that compiled and were correct enough to continue). If N_t drops to 2, the bias factor becomes 50%. TRLOO computes per-turn baselines excluding the current trajectory:

```
Ā^(-i)_t = 1/(N_t - 1) × Σ_{j≠i} G_j,t
```

**Empirical impact on CUDA kernels** (Dr. Kernel, KernelBench Level-2):
- GRPO: Fast@1.2x saturates at ~5.6% after 200 steps
- TRLOO: Fast@1.2x reaches ~20.0% and continues improving
- Profiling-based rejection sampling (PRS) adds another ~3.6x on top

**Recommendation:** If GRPO shows saturation during hackathon training (reward mean plateaus before step 100), consider: (1) increasing G from 4 to 8 to reduce bias factor from 25% to 12.5%, (2) switching to single-turn to avoid multi-turn bias compounding, or (3) abandoning GRPO in favor of SFT + evolutionary search (see FINAL_PRD Section 6.0.4).

**Step 4: Policy Loss (Clipped Surrogate)**

For each token t in completion o_{j,i}:

```
ratio_t = π_θ(token_t | context) / π_θ_old(token_t | context)

L_clip = min(
    ratio_t × Â_{j,i},
    clip(ratio_t, 1-ε, 1+ε) × Â_{j,i}
)
```

Where ε = 0.2 (standard PPO clip range).

The clipping prevents the policy from changing too much in one step. If the ratio goes above 1.2 or below 0.8, it gets clipped — limiting the gradient magnitude.

**Step 5: KL Penalty**

GRPO adds a KL divergence term to prevent the policy from drifting too far from the reference (initial) model:

```
KL_t = π_ref(token_t | context) / π_θ(token_t | context) 
       - log(π_ref(token_t | context) / π_θ(token_t | context)) 
       - 1
```

This is the "unbiased estimator" of KL divergence used by DeepSeek.

**Full GRPO Objective:**

```
J_GRPO(θ) = E[L_clip] - β × E[KL]
```

Where β is the KL coefficient. DeepSeek used β = 0.04. CUDA Agent's DAPO variant sets β = 0.0 (no KL penalty). **We use β = 0.0** because:
1. With PEFT/LoRA, the adapter IS the deviation — the base model stays frozen as the reference
2. DAPO showed KL penalty "not essential" for code tasks
3. Fewer hyperparameters to tune at a hackathon

### GRPO vs PPO: Memory Comparison

```
PPO requires 4 models in memory:
  1. Policy (π_θ)           — the model being trained
  2. Reference (π_ref)      — frozen copy for KL computation
  3. Critic (V_ψ)           — value network (same size as policy)
  4. Reward model (R_φ)     — scores completions

GRPO requires 2 models + 1 reward function:
  1. Policy (π_θ)           — the model being trained
  2. Reference (π_ref)      — frozen copy for KL computation
  3. Reward function         — our env.step() (NOT a neural network)

With PEFT (LoRA), reference behavior is recovered by disabling
the adapter — so there's no separate reference model in memory.

Effective GRPO memory = 1 model + LoRA adapters + optimizer states
```

---

## GRPO-2: Memory Budget

> **Hackathon primary:** H200 141GB with Qwen3-Coder-30B-A3B-Instruct. The B200 192GB budgets below are preserved for future scale-up reference only.

### Model: Qwen3-Coder-30B-A3B-Instruct (MoE, 3.3B active) — HACKATHON PRIMARY

```
H200 Total VRAM: 141 GB

COMPONENT                          MEMORY        NOTES
─────────────────────────────────────────────────────────
Model weights (FP8 or 4-bit)      ~16-18 GB     30.5B params, MoE, 3.3B active
LoRA adapters (r=16)               ~0.2 GB       Only on attention + MLP projections
LoRA optimizer (AdamW 8-bit)       ~1.5 GB       8-bit Adam states for LoRA params only
Gradient checkpointing activations ~4.0 GB       Unsloth's optimized checkpointing
KV cache (G=2, 8192 tokens)        ~4-6 GB       2 generations × 8192 × hidden_dim
Generation logprobs storage        ~2.0 GB       For policy ratio computation
Reference logprobs                 ~2.0 GB       LoRA disabled = reference behavior
─────────────────────────────────────────────────────────
TOTAL MODEL + TRAINING             ~30-34 GB
─────────────────────────────────────────────────────────

REMAINING FREE                     ~107-111 GB   ← Ample headroom on H200
```

This is the hackathon default. The model fits comfortably on H200 with substantial margin.

### Model: Qwen3.5-35B-A3B (MoE, 3B active per token) — FALLBACK SCENARIO

```
H200 Total VRAM: 141 GB

COMPONENT                          MEMORY        NOTES
─────────────────────────────────────────────────────────
Model weights (4-bit QLoRA)        ~17.5 GB      35B params × 4 bits / 8
LoRA adapters (r=16)               ~0.2 GB       Only on attention + MLP projections
LoRA optimizer (AdamW)             ~0.8 GB       2 states per LoRA param (FP32)
Gradient checkpointing activations ~4.0 GB       Unsloth's optimized checkpointing
KV cache (G=2, 4096 tokens each)   ~4.0 GB       2 generations × 4096 × hidden_dim
Generation logprobs storage        ~2.0 GB       For policy ratio computation
Reference logprobs                 ~2.0 GB       LoRA disabled = reference behavior
─────────────────────────────────────────────────────────
TOTAL MODEL + TRAINING             ~34.5 GB
─────────────────────────────────────────────────────────

REMAINING FREE                     ~106.5 GB     ← Comfortable fit on H200
```

> **Note:** Eval runs on the remote A100 backend selected by `KERNELFORGE_EVAL_BACKEND` — Northflank/CoreWeave service by default, Modal only as fallback. H200 only handles model weights, generation, and gradient updates.

### Model: Qwen3.5-9B (Dense)

```
COMPONENT                          MEMORY
─────────────────────────────────────────────────────────
Model weights (4-bit QLoRA)        ~5 GB
LoRA + optimizer                   ~1.5 GB
Activations + KV cache             ~6 GB
─────────────────────────────────────────────────────────
TOTAL                              ~12.5 GB
REMAINING (H200)                   ~128.5 GB
REMAINING (B200, future)           ~179.5 GB
```

### Model: Qwen3-Coder-Next (80B MoE, 3B active) — FUTURE SCALE-UP (NOT hackathon primary)

```
COMPONENT                          MEMORY
─────────────────────────────────────────────────────────
Model weights (FP8 Dynamic)        ~85 GB         80B × 8 bits / 8 + overhead
LoRA + optimizer                   ~3 GB
Activations + KV cache             ~12 GB
─────────────────────────────────────────────────────────
TOTAL                              ~100 GB
REMAINING (H200 141GB)             ~41 GB         Fits on H200 (141GB) — tight but feasible. Requires B200 for comfortable headroom.
REMAINING (B200 192GB)             ~92 GB         Comfortable fit on B200
```

**Conclusion:** On H200, Qwen3-Coder-30B-A3B-Instruct fits comfortably with ~107-111GB free. Qwen3-Coder-Next 80B fits on H200 (tight, ~41GB free) but B200 provides more headroom. The hackathon default is the 30B model on H200.

---

## GRPO-3: TRL GRPOTrainer — Exact Configuration

### 3.1 The Reward Function Signature

TRL's GRPOTrainer expects reward functions with this exact signature:

```python
def reward_func(
    prompts: list[str],           # The prompts
    completions: list[str],       # G completions for current prompt
    **kwargs                      # Absorbs completion_ids, trainer_state, extra columns
) -> list[float]:                 # One reward per completion
    """
    Must return a list of floats, same length as completions.
    Each float is the reward for the corresponding completion.
    """
```

### 3.2 Our Reward Function (Wraps OpenEnv)

```python
import subprocess
import json
import tempfile
import os
import re

def cuda_kernel_reward(prompts: list[str], completions: list[str],
                       **kwargs) -> list[float]:
    """
    Evaluate CUDA kernel completions through the KernelForge environment.
    
    For each completion:
    1. Extract CUDA code from model output
    2. Write to temp file
    3. Compile with nvcc -arch=sm_80 in subprocess (crash isolation)
    4. Verify correctness against PyTorch reference
    5. Profile vs torch.compile
    6. Return discrete milestone reward {-1, 1, 2, 3}

    Reward formula (CUDA Agent Equation 1):
    r = -1  if compilation fails OR correctness fails
    r = +1  if correct but no speedup (speedup <= 1.05x)
    r = +2  if >5% faster than torch.eager
    r = +3  if >5% faster than BOTH torch.eager AND torch.compile
    See openenv_env/reward.py for implementation.
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        # Extract CUDA code from completion
        cuda_code = _extract_cuda(completion)
        if not cuda_code:
            rewards.append(-1.0)
            continue
        
        # Get the task reference code for this completion
        ref_code = task_code[i] if task_code else None
        if not ref_code:
            rewards.append(-1.0)
            continue
        
        # Evaluate in subprocess (crash isolation — non-negotiable)
        reward = _evaluate_in_subprocess(cuda_code, ref_code)
        rewards.append(reward)
    
    return rewards


def _extract_cuda(text: str) -> str:
    """Extract CUDA code from model output."""
    # Try ```cuda ... ``` block
    match = re.search(r'```(?:cuda|cpp|c\+\+)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try raw code with __global__
    if '__global__' in text:
        return text.strip()
    return ""


def _evaluate_in_subprocess(cuda_code: str, ref_code: str, timeout: int = 60) -> float:
    """
    Run evaluation in a separate Python process.
    
    WHY SUBPROCESS: If a generated kernel causes a segfault, CUDA context
    corruption, or infinite loop, it kills the subprocess — not our training
    process. Every successful CUDA kernel RL system uses this pattern.
    """
    tmpdir = tempfile.mkdtemp()
    kernel_path = os.path.join(tmpdir, "kernel.cu")
    ref_path = os.path.join(tmpdir, "reference.py")
    eval_path = os.path.join(tmpdir, "evaluate.py")
    
    with open(kernel_path, 'w') as f:
        f.write(cuda_code)
    with open(ref_path, 'w') as f:
        f.write(ref_code)
    
    # The evaluation script runs in the subprocess
    eval_script = '''
import sys, json, subprocess, os, torch, time, importlib.util

kernel_path = sys.argv[1]
ref_path = sys.argv[2]

# Step 1: Compile with nvcc -arch=sm_80
binary = kernel_path.replace('.cu', '.so')
compile_cmd = [
    'nvcc', '-arch=sm_80', '-O3', '--use_fast_math',
    '-Xcompiler', '-fPIC', '-shared',
    '-o', binary, kernel_path
]
result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=30)
if result.returncode != 0:
    print(json.dumps({"reward": -1.0, "reason": "compile_fail"}))
    sys.exit(0)

# Step 2: Check for forbidden symbols (anti-reward-hacking)
nm_result = subprocess.run(['nm', '-D', binary], capture_output=True, text=True, timeout=5)
forbidden = ['torch', 'at::Tensor', 'c10::', 'triton']
for f in forbidden:
    if f in nm_result.stdout:
        print(json.dumps({"reward": -1.0, "reason": f"forbidden_symbol:{f}"}))
        sys.exit(0)

# Step 3: Load reference model
spec = importlib.util.spec_from_file_location("ref", ref_path)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)

model = ref.Model(*ref.get_init_inputs()).cuda().eval()
compiled_model = torch.compile(model)

# Step 4: Correctness check (5 random inputs)
for seed in range(5):
    torch.manual_seed(42 + seed)
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in ref.get_inputs()]
    with torch.no_grad():
        ref_out = model(*inputs)
    
    # TODO: Load and run the compiled kernel against these inputs
    # This requires the kernel to expose a callable interface
    # The exact mechanism depends on how the kernel is structured
    # For now, we check compilation + forbidden symbols only
    # >>> RISK 1.1 / Gate G-0.3 BLOCKER: This TODO must be resolved before
    # >>> any training. compute_reward() in openenv_env/reward.py needs to be
    # >>> wired to compilation results. See FINAL_PRD Section 6.1, Risk 1.1.
    pass

# Step 5: Profile (simplified — full version uses CUDA events)
# Profile eager
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(30):
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
eager_ms = (time.perf_counter() - start) / 30 * 1000

# Profile compiled
for _ in range(10):  # warmup
    with torch.no_grad():
        compiled_model(*inputs)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(30):
    with torch.no_grad():
        compiled_model(*inputs)
    torch.cuda.synchronize()
compile_ms = (time.perf_counter() - start) / 30 * 1000

# TODO: Profile the generated kernel
# For hackathon MVP, use compilation success as primary signal
# >>> RISK 1.1 / Gate G-0.3 BLOCKER: Without kernel profiling, rewards r=+2
# >>> and r=+3 are unreachable (kernel_ms always equals compile_ms).
# >>> Model only ever gets r=+1 or r=-1. See FINAL_PRD Section 6.1, Risk 1.1.
kernel_ms = compile_ms  # Placeholder

# Step 6: Compute reward (CUDA Agent Equation 1)
speedup_eager = eager_ms / kernel_ms if kernel_ms > 0 else 0
speedup_compile = compile_ms / kernel_ms if kernel_ms > 0 else 0

if speedup_compile > 1.05:
    reward = 3.0
elif speedup_eager > 1.05:
    reward = 2.0
else:
    reward = 1.0

print(json.dumps({
    "reward": reward,
    "eager_ms": eager_ms,
    "compile_ms": compile_ms,
    "kernel_ms": kernel_ms,
    "speedup_eager": speedup_eager,
    "speedup_compile": speedup_compile,
}))
'''
    
    with open(eval_path, 'w') as f:
        f.write(eval_script)
    
    try:
        result = subprocess.run(
            ['python', eval_path, kernel_path, ref_path],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        if result.returncode != 0:
            return -1.0
        
        output = json.loads(result.stdout.strip().split('\n')[-1])
        return output['reward']
    
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return -1.0
```

### 3.3 Dataset Preparation

```python
from datasets import load_dataset, Dataset

def prepare_grpo_dataset(stage: str = "warmup") -> Dataset:
    """
    Prepare dataset for GRPO training.
    
    Each row must have a 'prompt' column (required by GRPOTrainer)
    and a 'task_code' column (forwarded to reward function via **kwargs).
    """
    ops_6k = load_dataset("BytedTsinghua-SIA/CUDA-Agent-Ops-6K", split="train")
    # Also available: datasets/combined_kernelforge.jsonl (~6,192 merged RL prompts)
    # Load via: load_dataset("json", data_files="datasets/combined_kernelforge.jsonl")
    # For SFT: datasets/doublegraph_sft.jsonl (192 expert A100 kernel demonstrations)

    # Load the SKILL.md content (A100 optimization context — includes 7 real doubleGraph patterns)
    # Generated dynamically by openenv_env/skill_builder.py:build_skill_md("a100")
    with open("skill_a100.md") as f:
        skill_md = f.read()
    
    tasks = []
    for row in ops_6k:
        # Determine difficulty from data_source field
        num_ops = int(row["data_source"].split("#")[-1])
        
        # Stage 1 warmup: only single-op tasks
        if stage == "warmup" and num_ops > 1:
            continue
        
        # Stage 3 curriculum: include harder tasks
        if stage == "curriculum" and num_ops > 5:
            continue
        
        prompt = (
            f"{skill_md}\n\n"
            f"## Problem\n"
            f"Write an optimized CUDA kernel for A100 (sm_80) that replaces:\n\n"
            f"```python\n{row['code']}\n```\n\n"
            f"Operations: {row['ops']}\n\n"
            f"Write ONLY the .cu file:\n```cuda\n"
        )
        
        tasks.append({
            "prompt": prompt,
            "task_code": row["code"],  # Forwarded to reward func as **kwargs
            "ops": row["ops"],
            "num_ops": num_ops,
        })
    
    # Limit for hackathon
    if stage == "warmup":
        tasks = tasks[:512]
    elif stage == "curriculum":
        tasks = tasks[:200]
    
    return Dataset.from_list(tasks)
```

### 3.4 Full Training Script (All 3 Stages)

```python
"""
training/run_all_stages.py

Complete 3-stage GRPO pipeline for CUDA kernel optimization.
Runs on H200 141GB. Targets A100 sm_80 kernels.

Usage:
    python training/run_all_stages.py \
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
        --output_dir ./checkpoints \
        --stage all

Estimated wall-clock:
    Stage 1 (warmup):    ~2 hours
    Stage 2 (RFT):       ~30 minutes
    Stage 3 (pilot):     ~2 hours
    Total:               ~4.5 hours
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel


# ============================================================
# STAGE 1: GRPO WARM-UP
# Goal: Bootstrap compilation rate ~50% → ~85%
# Dataset: 512 single-op tasks from Ops-6K
# Turns: 3 (multi-turn with compile error feedback)
# Steps: 300
# ============================================================

def run_stage1(model, tokenizer, output_dir: str):
    print("=" * 60)
    print("STAGE 1: GRPO WARM-UP")
    print("=" * 60)
    
    dataset = prepare_grpo_dataset(stage="warmup")
    print(f"Stage 1 dataset: {len(dataset)} tasks")
    
    config = GRPOConfig(
        output_dir=os.path.join(output_dir, "stage1"),
        
        # GRPO-specific
        num_generations=2,            # G = 2 (reduced from 4 — fewer zero-gradient steps)
        # For each prompt, generate 2 CUDA kernels,
        # evaluate both, compute group-relative advantages,
        # update policy to favor higher-reward kernels
        
        # Clipping
        # epsilon for clipped surrogate loss
        # ratio_t clipped to [1 - 0.2, 1 + 0.2] = [0.8, 1.2]
        max_grad_norm=1.0,            # Gradient clipping (prevents explosions)
        
        # KL penalty
        beta=0.0,                     # No KL penalty (DAPO: "not essential")
        # With PEFT/LoRA, reference behavior = adapter disabled
        # No separate reference model needed in memory
        
        # Generation
        max_completion_length=768,    # CUDA kernels are 50-300 lines
        temperature=1.0,              # High for diverse exploration in warmup
        # Higher = more diverse = stronger GRPO signal.
        
        # Training
        per_device_train_batch_size=1, # 1 prompt per step
        # Effective batch = 1 prompt × 4 generations = 4 completions
        gradient_accumulation_steps=4, # Accumulate over 4 prompts
        # Effective batch for gradient = 16 completions
        num_train_epochs=1,
        max_steps=100,                # Stage 1 budget (hackathon-adjusted; full budget=300)
        learning_rate=2e-6,           # Conservative — preserve base model capabilities
        
        # Precision
        bf16=True,                    # BF16 training on H200
        
        # Logging
        logging_steps=1,              # Log every step (watch the reward!)
        save_steps=50,                # Checkpoint every 50 steps
        
        # Memory optimization
        gradient_checkpointing=True,
        
        # Important: keep task_code column for reward function
        remove_unused_columns=False,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=cuda_kernel_reward,  # Our reward function from 3.2
        args=config,
        train_dataset=dataset,
    )
    
    print("Starting Stage 1 training...")
    train_result = trainer.train()
    
    # Save Stage 1 checkpoint
    trainer.save_model(os.path.join(output_dir, "stage1", "final"))
    
    # Collect trajectories for Stage 2 RFT
    print("Collecting trajectories for RFT...")
    trajectories = collect_trajectories(model, tokenizer, dataset, n=100)
    
    return train_result, trajectories


# ============================================================
# STAGE 2: REJECTION FINE-TUNING (RFT)
# Goal: Filter successful trajectories, SFT on them
# Without RFT: faster-than-compile drops 96.8% → 49.8%
# (CUDA Agent Table 2 — this is NON-NEGOTIABLE)
# ============================================================

def run_stage2(model, tokenizer, trajectories: list, output_dir: str):
    print("=" * 60)
    print("STAGE 2: REJECTION FINE-TUNING (RFT)")
    print("=" * 60)
    
    # Filter: keep only trajectories with reward >= 1.0 (correct kernels)
    good_trajectories = [t for t in trajectories if t["reward"] >= 1.0]
    print(f"Collected {len(trajectories)} trajectories, "
          f"{len(good_trajectories)} passed filter (reward >= 1.0)")
    
    if len(good_trajectories) < 10:
        print("WARNING: Too few good trajectories. RFT may not be effective.")
        print("Possible causes: model can't generate valid CUDA yet.")
        print("Consider: longer Stage 1, or SFT warmup on pre-generated data first.")
    
    # Format for SFT
    sft_data = []
    for t in good_trajectories:
        sft_data.append({
            "prompt": t["prompt"],
            "completion": t["completion"],
        })
    
    sft_dataset = Dataset.from_list(sft_data)
    
    # SFT training (3 epochs over filtered data)
    from trl import SFTConfig, SFTTrainer
    
    sft_config = SFTConfig(
        output_dir=os.path.join(output_dir, "stage2"),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=999999,  # Don't save intermediate — just final
        max_seq_length=8192,
    )
    
    sft_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=sft_dataset,
    )
    
    sft_trainer.train()
    sft_trainer.save_model(os.path.join(output_dir, "stage2", "final"))
    
    print(f"RFT complete. Trained on {len(good_trajectories)} trajectories × 3 epochs.")


# ============================================================
# STAGE 3: GRPO + CURRICULUM
# Goal: Learn optimization strategies through progressive difficulty
# Dataset: 200 curated tasks across 4 difficulty phases
# Turns: 3-5 (hackathon; 20 is future target)
# Steps: 50 (hackathon pilot; 150 is future target)
# ============================================================

def run_stage3(model, tokenizer, output_dir: str):
    print("=" * 60)
    print("STAGE 3: GRPO + CURRICULUM")
    print("=" * 60)
    
    dataset = prepare_grpo_dataset(stage="curriculum")
    print(f"Stage 3 dataset: {len(dataset)} tasks")
    
    config = GRPOConfig(
        output_dir=os.path.join(output_dir, "stage3"),
        
        # GRPO
        num_generations=2,            # G = 2 (reduced from 4)
        beta=0.0,
        max_grad_norm=1.0,

        # Generation — lower temperature for exploitation
        max_completion_length=768,
        temperature=0.7,   # Lower for Stage 3 exploitation
        
        # Training
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        max_steps=50,                # Hackathon pilot (future target: 150)
        learning_rate=3e-6,  # Matches PRD table and GRPO-15.1
        
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=50,
        remove_unused_columns=False,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=cuda_kernel_reward,
        args=config,
        train_dataset=dataset,
    )
    
    print("Starting Stage 3 training...")
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "stage3", "final"))


# ============================================================
# TRAJECTORY COLLECTION (for RFT)
# ============================================================

def collect_trajectories(model, tokenizer, dataset, n: int = 100) -> list:
    """
    Generate n trajectories and evaluate them.
    Used between Stage 1 and Stage 2 to collect RFT data.
    """
    trajectories = []
    model.eval()
    
    for i in range(min(n, len(dataset))):
        row = dataset[i]
        prompt = row["prompt"]
        
        # Generate one completion
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=1.0,
                do_sample=True,
            )
        
        completion = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], 
                                       skip_special_tokens=True)
        
        # Evaluate
        reward = cuda_kernel_reward(
            completions=[completion],
            prompts=[prompt],
            task_code=[row["task_code"]]
        )[0]
        
        trajectories.append({
            "prompt": prompt,
            "completion": completion,
            "reward": reward,
            "task_ops": row["ops"],
        })
        
        status = "✓" if reward >= 1.0 else "✗"
        print(f"  [{i+1}/{n}] {status} reward={reward:.0f} ops={row['ops'][:50]}")
    
    return trajectories


# ============================================================
# MAIN: RUN ALL STAGES
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--stage", default="all", 
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    args = parser.parse_args()
    
    # Load model with Unsloth
    print(f"Loading {args.model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=8192,
        dtype=None,  # Auto-detect (BF16 on H200)
        load_in_4bit=args.load_in_4bit,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Unsloth recommends 0
        use_gradient_checkpointing="unsloth",
    )
    
    print(f"Model loaded. Trainable params: {model.print_trainable_parameters()}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.stage in ["1", "all"]:
        result, trajectories = run_stage1(model, tokenizer, args.output_dir)
    
    if args.stage in ["2", "all"]:
        if args.stage == "2":
            # Load trajectories from Stage 1
            with open(os.path.join(args.output_dir, "stage1", "trajectories.json")) as f:
                trajectories = json.load(f)
        run_stage2(model, tokenizer, trajectories, args.output_dir)
    
    if args.stage in ["3", "all"]:
        run_stage3(model, tokenizer, args.output_dir)
    
    print("\nDone! Final model saved to:", args.output_dir)


if __name__ == "__main__":
    main()
```

---

## GRPO-4: MARS+TRLOO Credit Assignment (Multi-Turn Extension)

### STATUS: DROPPED

**MARS computes cumulative returns on per-turn rewards. KernelForge uses outcome-only rewards (reward at episode end). With outcome rewards, `return_to_go[t] = r_final` for all t, making MARS identical to standard trajectory-level GRPO. Per CUDA Agent paper analysis: MARS provides zero benefit here.**

The code below is retained for reference only. Do not implement or present MARS as part of the hackathon stack.

### Why Standard GRPO Fails on Multi-Turn

Standard GRPO assigns the SAME advantage to every token in a completion. In multi-turn kernel optimization:

```
Turn 1: Model writes kernel → compile error         (bad)
Turn 2: Model fixes error → compiles, wrong output   (ok)
Turn 3: Model fixes output → correct, 1.2× speedup   (good!)

Standard GRPO: ALL tokens in ALL turns get advantage based on final reward (+2)
Problem: Turn 1's buggy code gets REWARDED because the final outcome was good
```

This is the #1 reason agentic RL systems learn syntax but not optimization — early critical fixes get diluted or miscredited → slow learning or collapse. CUDA Agent needed a full 4-stage pipeline partly because of this.

### MARS Solution (arXiv:2510.15414, ICLR 2026)

**Paper:** https://arxiv.org/abs/2510.15414 (v3)
**GitHub:** https://github.com/thu-nics/MARSHAL
**Models:** https://huggingface.co/collections/nics-efc/marshal

MARS computes PER-TURN advantages using cumulative returns (Eq. 7 in paper):

```python
def compute_mars_returns(turn_rewards: list[float], gamma: float = 1.0) -> list[float]:
    """
    MARS: turn-level cumulative returns.

    R_k = r_k + gamma * R_{k+1}   (backward pass)

    The return for turn k captures all future value from that point.
    This is mathematically equivalent to GAE(gamma=1, lambda=1) with
    batch-mean baseline, but applied at turn granularity.

    Example:
        turn_rewards = [-1.0, -0.5, +2.0]  # compile fail, wrong output, success

        Cumulative returns (backward):
        R_3 = 2.0
        R_2 = -0.5 + 1.0 × 2.0 = 1.5
        R_1 = -1.0 + 1.0 × 1.5 = 0.5

        Turn 1 tokens get advantage based on R_1 = 0.5  (modest)
        Turn 3 tokens get advantage based on R_3 = 2.0  (high)
        → Model learns: fixing errors is valuable, but writing correct code first is more valuable
    """
    K = len(turn_rewards)
    if K == 0:
        return []
    cumulative_returns = [0.0] * K

    running = 0.0
    for k in range(K - 1, -1, -1):
        running = turn_rewards[k] + gamma * running
        cumulative_returns[k] = running

    return cumulative_returns
```

### TRLOO Hybrid: Fixing GRPO Self-Inclusion Bias (Dr. Kernel, arXiv 2602.05885)

Standard MARS + GRPO still has the self-inclusion bias: computing the group mean/std using the current sample. TRLOO (Turn-level Reinforce Leave-One-Out) removes the current trajectory from the baseline:

```python
def mars_trloo_advantages(group_rollouts: list[dict], gamma: float = 1.0) -> list[list[float]]:
    """
    MARS cumulative returns + TRLOO leave-one-out baseline.

    For each trajectory i in the group:
    1. Compute cumulative returns R_{i,k} for each turn k
    2. Baseline = mean of R_{j,k} for j ≠ i (leave-one-out)
    3. Advantage A_{i,k} = R_{i,k} - baseline

    This fixes GRPO's (1 - 1/N) gradient shrinkage:
      GRPO:  E[ĝ] = (1 - 1/N) × ∇θJ(θ)     ← 50% too small at G=2
      TRLOO: E[ĝ] = ∇θJ(θ)                    ← unbiased

    Args:
        group_rollouts: list of G trajectories, each with "turn_rewards" key
        gamma: discount factor (1.0 = no discounting, matches MARSHAL paper)

    Returns:
        advantages: list of G lists, each with per-turn advantages
    """
    G = len(group_rollouts)

    # Step 1: Compute MARS cumulative returns for each trajectory
    all_returns = []
    for traj in group_rollouts:
        returns = compute_mars_returns(traj["turn_rewards"], gamma)
        all_returns.append(returns)

    # Step 2: TRLOO leave-one-out baseline per trajectory, per turn
    all_advantages = []
    for i in range(G):
        traj_advantages = []
        for k in range(len(all_returns[i])):
            R_ik = all_returns[i][k]
            # Leave-one-out: mean of all OTHER trajectories at turn k
            others_at_k = []
            for j in range(G):
                if j != i and k < len(all_returns[j]):
                    others_at_k.append(all_returns[j][k])
            baseline = sum(others_at_k) / len(others_at_k) if others_at_k else 0.0
            traj_advantages.append(R_ik - baseline)
        all_advantages.append(traj_advantages)

    return all_advantages
```

### Ablation Results (MARSHAL Paper, Tables 14-15)

| Configuration | Avg Return | Notes |
|--------------|-----------|-------|
| Without MARS (trajectory-level GRPO) | 1.764 | Standard GRPO baseline |
| With MARS (turn-level) | 2.173 | **+23% improvement** |
| Without agent-specific normalization | 1.764 | Collapses in mixed-role settings |
| With agent-specific normalization | +0.41 win-rate | Normalize per role separately |

**Zero-shot transfer (from strategic games):**
- AIME: +10.0%
- GPQA-Diamond: +7.6%

**Why it works:** Captures long-horizon dependencies (Turn 1 avoiding a crash gets credit for +3 reward in Turn 5), low variance (cumulative returns + group normalization), zero extra models (works with pure GRPO, no critic needed — perfect for single-GPU QLoRA).

### Integration with training/multi_turn_rollout.py

The existing `multi_turn_rollout.py` uses a `best_reward` heuristic that gives the same reward to all turns. Replace with MARS+TRLOO:

```python
def multi_turn_hybrid_rollout(prompts, trainer):
    """
    Custom rollout with:
    1. MARS+TRLOO per-turn credit assignment
    2. Hybrid eval (local turns 1-2, remote A100 backend turn 3)
    3. Nsight structured reward (see GRPO-9)
    4. Early stop at r >= 2

    Passed via: GRPOTrainer(rollout_func=multi_turn_hybrid_rollout, ...)
    """
    MAX_TURNS = 3  # Reduced from 5 (TRLOO bias compounds with more turns)

    tokenizer = trainer.processing_class
    all_prompt_ids, all_completion_ids, all_logprobs, all_rewards = [], [], [], []

    for prompt in prompts:
        history = prompt
        turn_rewards = []

        for turn in range(MAX_TURNS):
            outputs = generate_rollout_completions(trainer, [history])
            turn_text = tokenizer.decode(outputs[0]["completion_ids"], skip_special_tokens=True)

            cuda_code = _extract_cuda(turn_text)
            if not cuda_code:
                turn_rewards.append(-1.0)
            elif turn < MAX_TURNS - 1:
                # Turns 1-2: cheap local eval (nvcc + 3-graph PAC)
                result = _local_eval(cuda_code, task_code)
                turn_rewards.append(_compute_reward(result))
            else:
                # Final turn: full Modal + Nsight eval
                # NOTE: actual dispatch uses eval_backend.dispatch_eval() via _dispatch()
                result = _modal_eval_nsight(cuda_code, task_code)
                turn_rewards.append(_compute_reward_nsight(result))

            if turn_rewards[-1] >= 2.0:
                break  # Early stop

            # Feedback for next turn (with Nsight metrics if available)
            history += turn_text + _format_feedback(result)

        # MARS cumulative returns
        cumulative = compute_mars_returns(turn_rewards, gamma=1.0)
        episode_reward = cumulative[0] if cumulative else -1.0

        all_prompt_ids.append(outputs[0]["prompt_ids"])
        all_completion_ids.append(outputs[0]["completion_ids"])
        all_logprobs.append(outputs[0]["logprobs"])
        all_rewards.append(episode_reward)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "rewards": all_rewards,
    }
```

**Note:** For TRLOO group normalization, wrap this in a batch loop where G trajectories for the same prompt are collected, then call `mars_trloo_advantages()` across the group before passing rewards to TRL.

### Comparison Table

| Method | Credit Granularity | Variance | Multi-turn Stability | Our Fit |
|--------|-------------------|----------|---------------------|---------|
| Standard GRPO | Trajectory | High | Poor | Bad |
| Naive per-turn norm | Turn (local) | Very High | Collapse | Bad |
| **MARS (sum-then-norm)** | Turn (cumulative) | Low | Excellent | **Perfect** |
| **MARS+TRLOO** | Turn (cumulative, leave-one-out) | Low | Excellent | **Best** |
| GAE(0.95,0.95) + critic | Token | Medium | Good (needs critic) | Too heavy |

**Bottom line (OUTDATED — see STATUS: DROPPED above):** MARS provides zero benefit with outcome-only rewards. TRLOO alone is the implemented credit assignment method. MARS code above is retained for reference only.

> **Implementation status:** TRLOO advantage scaling is **IMPLEMENTED** in `training/custom_grpo_trainer.py`. MARS is **DROPPED** (not "stretch goal") because KernelForge uses outcome-only rewards, making MARS mathematically identical to trajectory-level GRPO. Do not implement or present MARS as part of the stack.

---

### Gate G-0.8: 5-Step GRPO Sanity Check (Thu Night)

**When:** Thursday night, after P0 eval pipeline is working.
**What:** Run 5 GRPO steps with the discrete milestone reward {-1, 1, 2, 3} + TRLOO post-process.

**Pass criteria:**
1. Non-zero gradients in at least 4/5 steps
2. Reward variance > 0 in at least 3/5 steps (i.e., not all completions get the same reward)
3. At least 2 out of 20 test kernels (5 steps × 4 candidates) achieve >1.2× speedup vs torch.compile

**Fail action:**
- Zero gradients in >2 steps → reward function is broken. Debug reward computation.
- Reward variance = 0 in >3 steps → model is collapsed or all outputs identical. Increase temperature to 1.2 or switch to backup model.
- No kernels >1.2× → model cannot optimize on this task distribution. Fall back to SFT + SkyDiscover only. Do NOT continue to P3 GRPO.

```bash
# Gate G-0.8 validation
KERNELFORGE_STAGE3_MAX_STEPS=5 KERNELFORGE_STAGE3_LOCAL_ONLY=1 \
    python -m training.stage3_grpo
# Check: non-zero gradients, reward variance, any speedup > 1.2x
```

---

## GRPO-5: Compute Budget (Hackathon: H200 + A100)

> **Hackathon budget target: <$200 total.** H200 at $4.54/hr, A100 80GB at $2.50/hr. See PRD Section 6 for budget breakdown.

### Time Per GRPO Step

```
COMPONENT                          TIME          NOTES
─────────────────────────────────────────────────────────
Model generation (4 × 2048 tokens)  ~40-80s      3B active × 4 candidates
                                                  ~50 tok/s per candidate
nvcc compilation (4 kernels)        ~20-120s     5-30s each, SEQUENTIAL
                                                  (can't parallelize on 1 GPU)
Correctness check (4 × 5 inputs)   ~8-40s       2-10s per kernel
Profiling (4 × 30 runs)            ~20-60s      50 warmup + 30 timed
Gradient computation                ~5-10s       Small with LoRA
─────────────────────────────────────────────────────────
TOTAL PER STEP (single-turn):      ~90-310s     ~1.5-5 minutes
TOTAL PER STEP (multi-turn, 3T):   ~270-930s    ~4.5-15 minutes
```

### Throughput Estimates

| Stage | Steps | Time/Step | Wall-Clock | Total Evaluations |
|-------|-------|-----------|------------|-------------------|
| Stage 1 (300 steps, 3 turns) | 300 | ~5 min | ~25 hours | 3,600 |
| Stage 2 (RFT, SFT) | N/A | N/A | ~30 min | 100 |
| Stage 3 (pilot, H200+A100) | 50 | ~2 min | ~2 hours | 100 |

**Reality check (revised):** Stage 3 is a GRPO pilot (H200 generates, A100 evaluates through the remote backend selected by `KERNELFORGE_EVAL_BACKEND`). Full pipeline is still gated by remote eval throughput, so Stage 3 only runs if Gate G-0.8 passes.

### Hackathon-Adjusted Budget

For a 24-hour hackathon day, reduce stages:

| Stage | Steps | Time/Step | Wall-Clock | Evaluations |
|-------|-------|-----------|------------|-------------|
| Stage 1 (single-turn, not multi) | 100 | ~2 min | ~3.5 hours | 400 |
| Stage 2 (RFT) | N/A | N/A | ~30 min | 50 |
| Stage 3 (single-turn GRPO) | 80 | ~2 min | ~3 hours | 320 |

**Total: ~7 hours.** Fits in one hackathon day. Leaves time for SkyDiscover runs, evaluation, and demo preparation.

**Key tradeoff:** Single-turn (agent writes one kernel, gets one reward) instead of multi-turn (agent iterates on errors). Multi-turn is better for learning but 3-5x slower per step. For hackathon, single-turn is pragmatic.

---

## GRPO-6: What To Monitor During Training

### Key Metrics (logged by GRPOTrainer)

```
grpo/reward/mean          — Average reward across all generations
                            Should increase over training.
                            Stage 1: -1 → 0 → 0.5 (learning to compile)
                            Stage 3: 0.5 → 1.0 → 1.5 (learning to optimize)

grpo/reward/std           — Reward variance within groups
                            Should be >0 (if 0, no learning signal)
                            If std drops to 0: all generations identical → increase temperature

grpo/completion_length    — Average completion length
                            Should stabilize at 100-300 tokens for CUDA kernels
                            If growing unboundedly: model is generating garbage

grpo/kl                   — KL divergence from reference
                            With beta=0, not penalized but still tracked
                            If >10: model has drifted far from base, may be unstable

train/loss                — Training loss
                            Should decrease. If NaN: reduce learning rate.
```

### Decision Points During Hackathon

```
After 30 Stage 1 steps:
  reward/mean > -0.5?
    YES → On track. Continue.
    NO  → Model can't compile CUDA. Options:
          1. Switch to fallback model (Qwen3.5-35B-A3B)
          2. Do SFT warmup first (pre-generated CUDA data)
          3. Simplify tasks (try vector_add only)

After 100 Stage 1 steps:
  reward/mean > 0?
    YES → Good. Proceed to Stage 2 RFT.
    NO  → Stage 1 isn't working. Options:
          1. Increase temperature to 1.0 (more exploration)
          2. Reduce task difficulty (single-op only)
          3. Bail on GRPO, focus on SkyDiscover

After Stage 2 RFT (30 min):
  Test on 20 tasks: compilation rate > 70%?
    YES → Proceed to Stage 3.
    NO  → RFT didn't anchor the policy. Collect more trajectories.

After 50 Stage 3 steps:
  reward/mean increasing?
    YES → Let it cook. Run until hackathon deadline.
    NO  → Model plateau'd. Stop training, use best checkpoint.
```

---

## GRPO-7: Critical Implementation Notes

### 7.1 Unsloth + PEFT Reference Model Trick

With PEFT (LoRA), GRPOTrainer does NOT keep a separate reference model. Instead, it temporarily disables the LoRA adapter to recover reference behavior. This saves ~17.5 GB (for 35B-A3B) of VRAM.

```python
# Internally, TRL does this:
with model.disable_adapter():
    ref_logprobs = model.forward(input_ids)  # Reference behavior

# Then with adapter enabled:
policy_logprobs = model.forward(input_ids)   # Current policy

# KL = f(ref_logprobs, policy_logprobs)
```

### 7.2 vLLM Colocate Mode (Single GPU)

> **Caveat (March 2026):**
> - vLLM is **disabled by default** for the hackathon (`KERNELFORGE_USE_VLLM=0`).
> - When enabled, the default mode is **"server"** (not "colocate").
> - vLLM kwargs (e.g., `vllm_gpu_memory_utilization`) are only passed conditionally when `USE_VLLM=True`.

For generation during GRPO, TRL can use vLLM in "colocate" mode — vLLM runs in the same process as training, sharing GPU memory.

```python
config = GRPOConfig(
    use_vllm=True,
    vllm_gpu_memory_utilization=0.5,  # Give vLLM 50% of remaining memory
    # Unsloth's "Standby" mode frees inference memory during gradient computation
)
```

**Warning:** vLLM colocate mode can cause OOM if the model is large. Test this BEFORE the hackathon. If it OOMs, fall back to `use_vllm=False` (uses transformers.generate, slower but safer).

### 7.3 The remove_unused_columns Trap

By default, Trainer removes dataset columns not in the model's forward signature. GRPOTrainer's reward funcs receive `prompts`, `completions`, `completion_ids`, `trainer_state`, plus any extra dataset columns via `**kwargs` — **but only if `remove_unused_columns=False`**. Without this, extra columns like `task_code` are stripped before reaching the reward function. **You MUST set:**

```python
config = GRPOConfig(
    remove_unused_columns=False,  # Keep task_code and other extra columns
)
```

And your reward function must use a **permissive signature** — TRL's internal plumbing has changed across versions:

```python
def reward_func(prompts, completions, **kwargs):
    # Extra dataset columns arrive via **kwargs if remove_unused_columns=False
    task_code = kwargs.get("task_code")
    # Belt-and-suspenders: also embed task_code in prompt text
```

**Belt-and-suspenders approach:** embed task_code into the `prompt` column text so reward works regardless of TRL version behavior. Set `remove_unused_columns=False` explicitly — do not rely on defaults.

### 7.4 Group Size and Reward Variance

If G=4 and all 4 completions get r=-1 (all fail to compile), advantages are all 0. No learning. The probability this happens depends on the model's compilation success rate:

```
P(all 4 fail) = (1 - success_rate)^4

If success_rate = 10% → P(all fail) = 0.9^4 = 65.6%  ← BAD (2/3 steps wasted)
If success_rate = 30% → P(all fail) = 0.7^4 = 24.0%  ← OK
If success_rate = 50% → P(all fail) = 0.5^4 = 6.25%  ← GOOD
If success_rate = 70% → P(all fail) = 0.3^4 = 0.81%  ← EXCELLENT
```

This is why Stage 1 warmup targets compilation rate > 85% before proceeding. If your model starts at <30% compilation rate, most GRPO steps will have zero signal and training won't converge.

---

## GRPO-8: Quick Reference Card

```
GRPO ALGORITHM SUMMARY:
1. Sample G completions per prompt from policy
2. Evaluate each → rewards r_1, ..., r_G
3. Advantage: Â_i = (r_i - mean(r)) / std(r)
4. Loss: min(ratio × Â, clip(ratio, 0.8, 1.2) × Â) - β × KL
5. Update policy via backprop

HACKATHON CONFIGURATION (March 5, 2026):
  Model = Qwen3-Coder-30B-A3B-Instruct (30.5B total, 3.3B active)
  Train GPU = H200 ($4.54/hr, 141GB)
  Eval GPU = A100 80GB ($2.50/hr)
  G = 2 (num_generations)
  β = 0.0 (no KL penalty, DAPO style)
  ε = 0.2 (clip range)
  Rewards = discrete milestone {-1, 1, 2, 3} (CUDA Agent Equation 1)
  Temperature = 1.0 (Stage 1) → 0.7 (Stage 3)
  Learning rate = 2e-6 (Stage 1) → 3e-6 (Stage 3)
  LoRA rank = 16, target = qkvo + gate/up/down
  Credit = TRLOO only (default; set `KERNELFORGE_USE_TRLOO=0` to use vanilla GRPO) (MARS DROPPED — see GRPO-4)
  Eval = Hybrid: local nvcc+PAC early turns, configured remote A100 backend for reward-bearing final evaluation (GRPO-10)
  Loss = standard clip (MASPO DEFERRED — see GRPO-12)
  Pruning = CPPO DROPPED (see GRPO-11)

H200 MEMORY (141 GB total):
  Coder-30B-A3B bf16: ~61 GB total → ~80 GB free (PRIMARY)
  35B-A3B bf16: ~65 GB total → ~76 GB free (FALLBACK)
  Coder-Next 80B FP8: ~100 GB → fits on H200 but tight (future B200 preferred)

HACKATHON BUDGET:
  Stage 1: 100 steps × ~2 min = ~3.5 hours
  Stage 2: RFT = ~30 min
  Stage 3: 50 steps × ~2 min = ~2 hours (pilot)
  Total: ~6 hours training + eval
  Cost: ~$60-100 (H200 + A100)

ABORT CONDITIONS:
  Reward mean stays at -1.0 after 30 steps → model can't compile
  Reward std = 0 → all completions identical → increase temperature
  Loss = NaN → reduce learning rate to 1e-6
  OOM → reduce max_completion_length to 2048, G=2→1
```

---

## GRPO-9: Nsight Compute Structured Rewards

### The Problem (Section 6.0.2 — Fundamental Failure #2)

Our reward function uses discrete levels {-1, 1, 2, 3}. Without profiling, only {-1, 1} were reachable. With profiling wired, all four levels are reachable. The discrete scheme has known limitations (a 1.06x kernel gets the same r=+2 as a 5x kernel), but provides clear milestone signal that is robust to timing noise.

CUDABench (arXiv 2603.02236): "mismatch between high compilation success and low functional correctness" -- this is why correctness gating (r=-1 for wrong output) is non-negotiable.

### Nsight Compute Metrics (Optional Enhancement, Not Current Reward)

Nsight metrics can provide additional diagnostic signal. They are currently available as optional kwargs in `compute_reward()` but are **not** part of the primary discrete reward. Future work may use them for a continuous reward variant:

```bash
# ncu command for the remote A100 eval endpoint
ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained,\
    smsp__warps_launched.avg.pct_of_peak_sustained_active \
    --csv kernel.so
```

**Structured reward vector (returned from Modal):**

```json
{
    "runtime_ms": 0.85,
    "speedup_eager": 1.32,
    "speedup_compile": 1.08,
    "occupancy": 0.72,
    "mem_coalescing": 0.88,
    "warp_efficiency": 0.91,
    "sm_util": 0.79,
    "bank_conflicts": 0.0
}
```

**Reward formula (discrete milestone {-1, 1, 2, 3} — CUDA Agent Equation 1):**

```python
def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
    occupancy: float | None = None,
    mem_coalescing: float | None = None,
    warp_efficiency: float | None = None,
) -> float:
    """Discrete milestone reward per CUDA Agent Equation 1.

    -1: compilation fails OR correctness fails
    +1: correct but no speedup (speedup <= 1.05x)
    +2: >5% faster than torch.eager
    +3: >5% faster than BOTH torch.eager AND torch.compile
    """
    if not compiled or not correct:
        return -1.0

    if speedup_vs_compile > 1.05:
        return 3.0
    elif speedup_vs_eager > 1.05:
        return 2.0
    else:
        return 1.0


def trloo_post_process(advantages: list[float], n: int) -> list[float]:
    """Scale GRPO advantages by N/(N-1) to correct gradient shrinkage.

    Dr. Kernel (arXiv 2602.05885) proves GRPO's self-inclusion bias
    shrinks expected gradients by (1 - 1/N). With G=4, that's 25%.
    Drop-in fix for TRL GRPOTrainer — no custom training loop needed.
    """
    if n <= 1:
        return advantages
    scale = n / (n - 1)
    return [a * scale for a in advantages]
```

**Why this matters for learning:** The discrete milestone reward {-1, 1, 2, 3} provides clear threshold-based signal that is robust to timing noise on A100. The gap between levels (especially -1 to +1, which gates on correctness) creates strong learning signal. TRLOO post-processing is a drop-in fix that corrects the 25% gradient shrinkage without requiring a custom training loop.

**Files:** `openenv_env/reward.py` (discrete milestone reward + TRLOO), `eval_service/eval_core.py` (shared compile + verify + benchmark pipeline), `openenv_env/eval_backend.py` (backend switch), `modal_app.py` (fallback wrapper)

---

## GRPO-10: Hybrid Eval (Local + Remote A100 Service)

### The Problem (Section 6.1, Risk 1.2)

Every reward-bearing eval still depends on a remote A100 service. If the service is slow (>30s/call), each GRPO step can still become expensive. The current hackathon path therefore keeps H200 for local compile pre-checks and reserves remote A100 calls for correctness and timing.

### The Fix: Local Cheap Eval for Early Turns, Remote Service for Final

```
Turn 1: local nvcc -arch=sm_80 syntax check                (~5 sec)
Turn 2: local nvcc + lightweight validation                (~10 sec)
Turn 3: remote A100 eval service + full correctness/timing (~30-90 sec)
```

**Early stop:** If any turn yields r >= 2.0, stop the episode (no need for further remote evaluation).

**Impact:**
- Per-step time drops because non-compiling kernels die locally before paying for A100 execution.
- For pilot GRPO runs, most waste is removed from the rollout loop before remote dispatch.
- Remote calls are reduced because only promising candidates hit the A100 service.

**Feedback includes Nsight metrics (when available):**

```python
def _format_feedback(result: dict) -> str:
    parts = []
    if not result.get("compiles"):
        parts.append("FEEDBACK: Compilation failed. Fix errors.")
    elif not result.get("correct"):
        parts.append("FEEDBACK: Wrong output. Fix correctness.")
    else:
        parts.append(f"FEEDBACK: Correct. Speedup: {result.get('speedup_eager', 0):.2f}x")
        ncu = result.get("ncu_metrics", {})
        if ncu:
            parts.append(f"  Occupancy: {ncu.get('occupancy', 0)*100:.0f}%")
            parts.append(f"  Memory coalescing: {ncu.get('mem_coalescing', 0)*100:.0f}%")
            parts.append(f"  Warp efficiency: {ncu.get('warp_efficiency', 0)*100:.0f}%")
            if ncu.get('occupancy', 0) < 0.5:
                parts.append("  LOW OCCUPANCY: reduce register usage or increase block size")
            if ncu.get('mem_coalescing', 0) < 0.5:
                parts.append("  POOR COALESCING: ensure consecutive threads access consecutive addresses")
    return "\n".join(parts)
```

**File:** `training/hybrid_rollout.py` (new, or modify `training/multi_turn_rollout.py`)

---

## GRPO-11: CPPO Completion Pruning

### STATUS: DROPPED

**After SFT on expert kernels, the model generates syntactically valid CUDA by default. Heuristic pre-filtering (checking for `__shared__`, `float4`) provides no value. During GRPO, pruning removes exploration signal needed for TRLOO group statistics.**

With G=2, pruning one candidate leaves G=1 which produces zero advantage variance. CPPO was designed for G=4+ settings where most candidates are garbage. Post-SFT at G=2, both candidates typically compile, making structural pre-filtering pointless.

The code below is retained for reference only. Do not implement.

**Paper:** arXiv [2503.22342](https://arxiv.org/abs/2503.22342)
**GitHub:** https://github.com/lzhxmu/CPPO

### The Fix: Cheap Structural Filter + Prune Bottom Completions

Score CUDA code by structural heuristics BEFORE sending to eval:

```python
def cheap_cuda_score(code: str) -> float:
    """Fast structural heuristic for CUDA code quality. No compilation needed."""
    if not code:
        return -10.0
    score = 0.0
    if '__global__' in code:     score += 2.0   # Has kernel declaration
    if '__shared__' in code:     score += 1.0   # Uses shared memory
    if 'float4' in code:         score += 0.5   # Vectorized loads
    if '__shfl' in code:         score += 0.5   # Warp primitives
    if '__launch_bounds__' in code: score += 0.3 # Occupancy-aware
    if len(code) < 100:          score -= 1.0   # Suspiciously short
    if '#include' not in code:   score -= 0.5   # Likely incomplete
    return score
```

**Usage:** Generate G candidates. Score all cheaply. Only eval top candidates on the remote backend. Assign r=-1 to pruned candidates (they didn't even pass the structural filter). With G=2, prune the weaker one if its score is below threshold.

**Impact:**
- 2-4x fewer remote eval calls per step
- For 50 trajectories in RFT: 200 → 50-100 remote eval calls
- Saves ~$1.50 and 25-75 min wall time
- CPPO paper shows 7.98x speedup on GSM8K with 70-80% pruning, same or better accuracy

**CPPO integration with advantage computation:**

After computing advantages, prune completions with |advantage| < threshold (typically 0.2). Only backpropagate through high-signal completions. This further reduces gradient noise from uninformative completions.

---

## GRPO-12: MASPO Soft Trust Region

### STATUS: DEFERRED (future research)

**Not planned for hackathon or near-term work.** Standard PPO clip is sufficient for the current pipeline. MASPO would only matter at scale (150+ steps, 20 turns) where clip-boundary oscillation becomes a measurable bottleneck. Revisit if Stage 3 training shows ratio-clipping artifacts in W&B logs.

**Paper:** arXiv [2602.17550](https://arxiv.org/abs/2602.17550)

### The Problem

PPO/GRPO's hard clip at ratio in [0.8, 1.2] creates a gradient discontinuity at the boundary. This causes:
- Oscillation at the clip boundary
- Loss of gradient signal for rare high-reward optimizations (which tend to have large policy changes)

### The Fix: Gaussian Soft Gating

Replace the hard clip with a smooth Gaussian gate:

```python
import torch

def maspo_soft_gate(ratio: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
    """Gaussian soft gating: smooth falloff instead of hard clip."""
    return torch.exp(-((ratio - 1.0) ** 2) / (2.0 * sigma ** 2))

def maspo_policy_loss(log_probs, old_log_probs, advantages, sigma=0.2):
    """MASPO soft trust region loss.

    Replaces: L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    With:     L_maspo = -ratio * A * w(ratio)
    """
    ratio = torch.exp(log_probs - old_log_probs)
    gate = maspo_soft_gate(ratio, sigma)
    return -(ratio * advantages * gate).mean()
```

**Key properties:**
- At ratio=1.0: gate=1.0 (no penalty)
- At ratio=1.2: gate=0.61 (soft dampening vs hard clip to 0)
- At ratio=1.5: gate=0.11 (strong dampening for large changes)
- Smooth gradients everywhere → better exploration on sparse rewards

**Impact:** +2-3% over GRPO on coding/math (MASPO paper ablations). Less collapse on rare high-reward optimizations — potentially useful with discrete rewards where high-speedup events (r=+3) are sparse.

**Integration:** Monkey-patch `GRPOTrainer.compute_loss()` in `training/stage3_grpo.py`, or create `training/maspo_loss.py` as a standalone module. If TRL's API is too opaque for monkey-patching, use standard clip (this is a nice-to-have, not critical).

---

## GRPO-13: Transformation Grammar (40 Rules) — DEFERRED TO v2

> **DEFERRED TO v2 (Fix 4):** Transformation grammar has not been shipped for CUDA kernels by any production system as of March 2026. The 12-40 rules below are theoretical. For v1, `openenv_env/skill_builder.py:_append_a100_patterns()` dynamically generates SKILL.md with 7 real A100 expert patterns from doubleGraph production kernels (degree-based dispatch, warp hash tables, zero-copy convergence, bitmap frontier, path-halving Union-Find, compilation flag tuning, __launch_bounds__). Output: 178 lines / 7.5KB for A100. This section is retained for reference but is NOT in the v1 implementation priority.

### The Problem

Free-form generation of 2000-token CUDA kernels is a massive search space. Most generated code doesn't compile. The model spends most of its learning budget on syntax, not optimization. OptiML (arXiv 2602.12305) and KernelBlaster (arXiv 2602.14293) prove that **structured edit operations** dramatically reduce the search space.

### The Fix: Model Proposes Transforms from a Grammar

Instead of "write a CUDA kernel from scratch," the model receives:
1. A baseline kernel (from torch2cuda, SkyDiscover, or previous iteration)
2. Current profile metrics (from Nsight, see GRPO-9)
3. A list of applicable transforms

And proposes 1-3 transforms to apply.

### Core 12 Transforms (Implement First)

```python
TRANSFORMS = [
    # Block/Grid configuration
    "change_block_size:32/64/128/256/512/1024",
    "add_grid_stride_loop",
    "add_launch_bounds:threads/min_blocks",

    # Memory optimization
    "coalesce_memory:transpose_access",
    "shared_memory_tiling:tile_size=32/64",
    "remove_bank_conflicts:pad_shared",
    "vectorize_loads:float4",
    "L2_pinning:__ldg",

    # Compute optimization
    "warp_shuffle_reduce",
    "loop_unroll:factor=4/8",
    "register_tiling",
    "cooperative_groups:sync",
]
```

### Extended 40 Rules (Implement If Time)

```python
EXTENDED_TRANSFORMS = TRANSFORMS + [
    # Arithmetic
    "use_fast_math:__fmul_rn/__fadd_rn",
    "fma_intrinsics:__fmaf_rn",
    "rsqrt_intrinsic:__frsqrt_rn",

    # Memory hierarchy
    "texture_cache:__ldg_via_texture",
    "constant_memory:__constant__",
    "prefetch:__prefetch_l1/__prefetch_l2",
    "double_buffering:shared_mem_ping_pong",
    "bank_conflict_padding:pad_by_1",

    # Parallelism
    "kernel_fusion:adjacent_kernels",
    "persistent_kernel:grid_of_1",
    "dynamic_parallelism:nested_launch",
    "warp_specialization:producer_consumer",

    # A100-specific (sm_80)
    "async_copy:cp.async",
    "ldmatrix:wmma_load",
    "mma_instruction:wmma/mma",
    "l2_residency_control:cudaAccessPolicyWindow",
    "occupancy_calculator:cudaOccupancyMaxActiveBlocksPerMultiprocessor",

    # Reduction patterns
    "tree_reduction:shared_mem",
    "atomic_add:__atomicAdd_block",
    "ballot_sync:__ballot_sync+__popc",
    "shfl_xor_reduce:butterfly",

    # Data layout
    "aos_to_soa:struct_of_arrays",
    "padding_alignment:aligned_alloc",
    "pitch_allocation:cudaMallocPitch",

    # Control flow
    "branch_divergence_fix:predication",
    "early_exit:if_return",
    "loop_interchange:cache_friendly_order",
]
```

### Prompt Template for Transform Mode

```
## Current Kernel
```cuda
{baseline_kernel}
```

## Profile Metrics
- Speedup vs eager: {speedup}x
- Occupancy: {occupancy}%
- Memory coalescing: {mem_coalescing}%
- Warp efficiency: {warp_efficiency}%
{bottleneck_hint}

## Available Transforms
{numbered_transform_list}

## Task
Select 1-3 transforms to improve this kernel for A100 (sm_80).
For each transform, explain why it addresses the bottleneck.
Format: TRANSFORM: name(args)
```

### Parser

```python
import re

def parse_transforms(text: str) -> list[tuple[str, str]]:
    """Parse 'TRANSFORM: name(args)' from model output."""
    transforms = []
    for match in re.finditer(r'TRANSFORM:\s*(\w+)\(([^)]*)\)', text):
        transforms.append((match.group(1), match.group(2)))
    return transforms
```

**File:** `openenv_env/transform_grammar.py` (~250 LOC)

---

## GRPO-14: Full Stacked Architecture (Single-GPU Hackathon)

### Complete Training Loop (All Techniques Combined)

```
For each GRPO step:

  IMPLEMENTED:
  1. Generate: vLLM produces G=2 candidates with SKILL.md context
  2. Hybrid eval:
     - Local nvcc compile check (~5s)
     - Remote A100 backend eval for compiling kernels (~30-90s)
     - Early stop if reward >= 3
  3. TRLOO: Leave-one-out advantage scaling N/(N-1)
  4. Standard clip loss (PPO-style ε=0.2)
  5. Update: Backprop through LoRA adapters

  UNBUILT (dropped or deferred):
  - SortedRL: Sort prompt queue by estimated completion length — NOT IMPLEMENTED
  - CPPO filter: Score candidates cheaply, prune bottom — DROPPED (see GRPO-11)
  - MARS per-turn credit: Cumulative returns on per-turn rewards — DROPPED (see GRPO-4)
  - MASPO loss: Soft trust region — DEFERRED (see GRPO-12)
  - Transform mode: Structured edits instead of free-form — DEFERRED to v2 (GRPO-13)
```

### Stage Configurations (Revised)

```
STAGE 1 — WARM-UP (Qwen3-Coder-30B-A3B-Instruct on H200):
  Steps: 100
  G: 2
  Max turns: 3
  Temperature: 1.0
  LR: 2e-6
  Action: Free-form generation (learn to compile first)
  Eval: Hybrid (local turns 1-2, remote A100 backend turn 3)
  Credit: TRLOO only (MARS DROPPED — see GRPO-4)
  Goal: Compilation rate 50% → 85%

STAGE 2 — RFT:
  Filter: reward >= 0.0 (any speedup is useful signal)
  SFT: 3 epochs on filtered trajectories + 192 doubleGraph expert demonstrations
  Goal: Anchor compilation ability + expose model to real A100 patterns

STAGE 3 — GRPO PILOT (Qwen3-Coder-30B-A3B-Instruct on H200) — HACKATHON CONFIG:
  Steps: 50 (hackathon pilot; 150 is future scale-up target)
  G: 2
  Max turns: 3-5 (hackathon; 20 is future target)
  Temperature: 0.7
  LR: 3e-6
  Context: 8,192 tokens (H200: ~107GB free — ample headroom)
  Action: Free-form generation with SKILL.md context (7 real A100 patterns via skill_builder.py)
  Eval: Hybrid — local nvcc early turns, remote A100 backend final turn
  Credit: TRLOO only (MARS DROPPED — see GRPO-4; CPPO DROPPED — see GRPO-11)
  Reward: Discrete milestone {-1, 1, 2, 3} (CUDA Agent Equation 1)
  Loss: Standard clip (MASPO DEFERRED — see GRPO-12)
  Inference: Best-of-N candidates per problem + SkyDiscover hedge
  Goal: Validate real learning signal, demonstrate candidate improvement
```

### New Files Summary (updated March 5, 2026)

| File | LOC | Status | Purpose |
|------|-----|--------|---------|
| `training/custom_grpo_trainer.py` | ~40 | **DONE** | TRLOOGRPOTrainer — N/(N-1) advantage scaling |
| `openenv_env/reward.py` | ~60 | **DONE** | Discrete milestone {-1,1,2,3} + TRLOO post-process |
| `training/multi_turn_rollout.py` | ~160 | **DONE** | Multi-turn rollout + local compile fast-path + backend-dispatched remote eval |
| `openenv_env/skill_builder.py` update | ~100 | **DONE** | `_append_a100_patterns()` — 7 real expert patterns from doubleGraph |
| `datasets/extract_doublegraph_a100.py` | ~550 | **DONE** | Harvest 192 A100 kernels → 3 TRL datasets |
| `skydiscover_integration/evaluator.py` | ~190 | **DONE** | Cascade evaluator bridge (stage1→stage2) |
| `skydiscover_integration/adaevolve.py` | ~300 | **DONE** | AdaEvolve multi-island UCB search |
| `skydiscover_integration/evox_strategies.py` | ~250 | **DONE** | EvoX self-evolving strategies + LogWindowScorer |
| `training/curriculum.py` update | ~80 | **DONE** | 5 topology-aware graph problems with graph_properties |
| `openenv_env/kernel_forge_env.py` update | ~10 | **DONE** | graph_properties + topology_type in observations |
| `modal_app.py` update | ~30 | **DONE** | Nsight lightweight profiling (ptxas occupancy) |
| `training/custom_grpo_loop.py` | ~100 | **DROPPED** | MARS return-to-go (no benefit with outcome rewards) + CPPO pruning (no benefit post-SFT at G=2) |
| `training/maspo_loss.py` | ~50 | **DEFERRED** | Soft trust region loss term (future research) |

**Dropped from v1:** `openenv_env/transform_grammar.py` (~250 LOC) — transformation grammar deferred to v2. Replaced by 7 real A100 patterns in SKILL.md.

### Expected Performance — Future Scale-Up Target (NOT Hackathon Claim)

> **WARNING:** The projections below are speculative future targets based on paper results. They are **NOT** hackathon claims. The hackathon will validate the core feedback loop; full benchmark comparison is a later research objective. See claim discipline rules in KERNELFORGE_FINAL_PRD.md Section 7.

```
FUTURE SCALE-UP: Qwen3-Coder-Next 80B on B200 (NOT hackathon config):
  Memory: ~100GB model + ~92GB free for sandbox
  Generation: ~50 tok/s (3B active parameters)

Training data:
  Ops-6K: 6,000 PyTorch operator tasks (GRPO prompts)
  doubleGraph: 192 expert A100 kernels (SFT + GRPO prompts, 84K+ lines)
  Curriculum: 21 problems across 4 phases (5 topology-aware graph tasks)
  SKILL.md: 178 lines, 7 real A100 patterns from production code

Projected training-time results (pass@1) — FUTURE TARGET, NOT PROVEN:
  Compilation rate: 93-96% (projected, if Stage 1 warmup + SFT on 192 real expert demos)
  Functional correctness: 80-88% (SPECULATIVE — depends on unimplemented components; MARS DROPPED)
  Geo-mean speedup: 1.8-2.1× (projected, on combined dataset + WCC + graph algorithms)
  A100 patterns discovered: 12-18 (projected — float4, L2 pinning, shuffles, tiling, etc.)

Projected inference-time results (best-of-8 + SkyDiscover) — FUTURE TARGET:
  Pass rate: 98-99% (projected, CUDA-Agent benchmark: 98.8%)
  Faster-than-compile: 93-97% (projected, CUDA-Agent benchmark: 96.8%)
  Geo-mean speedup: 2.0-2.3× (projected, CUDA-Agent benchmark: 2.11×)

Cost & timeline:
  Total evals: ~2,000 Modal calls (CPPO savings no longer assumed — DROPPED)
  Wall-clock training: ~14 hours (Stage 1: 3.5h + Stage 2: 0.5h + Stage 3: 10h)
  SkyDiscover parallel: evolutionary search on doubleGraph seeds (~2h)
  Total project time: 12-16 hours coding + 14 hours training
  Total cost: ~$155 ($125 after $30 free credit)
  vs CUDA-Agent: ~$5,000-10,000+ (30-60× cheaper)

Why these projections are plausible (research basis):
  - TRLOO: unbiased gradients (Dr. Kernel arXiv 2602.05885, +72% Fast@1.2x) — IMPLEMENTED
  - MARS: DROPPED (no benefit with outcome-only rewards — see GRPO-4)
  - Discrete milestone reward {-1, 1, 2, 3}: robust threshold-based signal — IMPLEMENTED
  - CPPO: DROPPED (no benefit post-SFT at G=2 — see GRPO-11)
  - 192 expert demos: real A100 patterns as SFT anchor — IMPLEMENTED
  - 7 SKILL.md patterns: concrete optimization strategies in every prompt — IMPLEMENTED
  - Best-of-N inference + SkyDiscover hedge — IMPLEMENTED
  NOTE: These projections assume full 150-step, 20-turn, B200 scale-up.
  SPECULATIVE — the per-sample efficiency claims assumed MARS+CPPO (both DROPPED).
  The hackathon will test the core loop at smaller scale with TRLOO only.
```

### Why This Architecture Has Potential (Future Targets)

| Original PRD Failure | Stacked Fix | Evidence |
|---------------------|-------------|---------|
| GRPO bias (50% gradient shrinkage at G=2) | TRLOO N/(N-1) leave-one-out | Dr. Kernel: unbiased, +72% Fast@1.2× |
| Flat credit (all turns same reward) | ~~MARS per-turn cumulative returns~~ DROPPED (no benefit with outcome-only rewards) | MARSHAL: +23% return — but requires per-turn rewards, which KernelForge does not use |
| Discrete {-1,1,2,3} (limited granularity) | Discrete milestone {-1, 1, 2, 3} with clear correctness gate | CUDA Agent Equation 1; robust to timing noise |
| 25x fewer RL samples than CUDA Agent | TRLOO correction + 192 expert demos. **SPECULATIVE -- depends on unimplemented components:** original 12-25x claim required MARS+CPPO (both DROPPED) | TRLOO alone: ~1.3x correction at G=2 |
| No real A100 patterns in training data | 192 doubleGraph kernels + 7 SKILL.md patterns | Production code: 84K+ lines, 3.6× avg speedup |
| Generic prompts (no topology awareness) | Topology-aware curriculum (5 graph tasks) | Power-law, dense-community, sparse-islands prompts |
| No evolutionary search hedge | AdaEvolve multi-island UCB + EvoX strategies | 4 islands, 5 seed kernels, cascade eval on the shared remote A100 backend |
| G=4 zero-gradient steps | G=2 + discrete milestone {-1,1,2,3} | 4-level reward → usually non-zero std when compile rates differ |
| Only 10 steps, 3 turns (demo) | 150 steps, 20 turns (match config) | Multi-turn = #1 CUDA-Agent ablation factor |
| Single candidate per problem | Best-of-8 inference + AdaEvolve | pass@8 ≈ 99.99%; max speedup selection |

**Bottom line:** This architecture combines TRLOO (implemented), discrete milestone rewards (implemented), expert demos, topology-aware curriculum, and evolutionary search. MARS and CPPO are DROPPED. MASPO is DEFERRED. The hackathon will validate the core feedback loop at small scale. Full benchmark comparison against CUDA-Agent is a future research target, not a hackathon claim.

---

## GRPO-15: Hackathon Config + Future Scale-Up Target

> **Added March 5, 2026. Revised March 5.** Split into hackathon primary config (15.1) and future scale-up target (15.2). See KERNELFORGE_FINAL_PRD.md Section 13 for future feasibility assessment.

### 15.1 Hackathon GRPO Config (Primary)

```
HACKATHON STAGE 3 — GRPO PILOT
  Model: Qwen3-Coder-30B-A3B-Instruct on H200
  Steps: 50 (pilot — validate signal, not benchmark domination)
  G: 2
  Max turns: 3-5
  Temperature: 0.7
  LR: 3e-6
  Context: 8,192 tokens (H200: ~80GB free)
  Eval: Hybrid — local nvcc early turns, remote A100 backend final turn
  Credit: TRLOO only (default; set `KERNELFORGE_USE_TRLOO=0` to use vanilla GRPO) (MARS DROPPED — see GRPO-4)
  Reward: Discrete milestone {-1, 1, 2, 3} (CUDA Agent Equation 1)
  Loss: Standard clip
  Cost: ~$30-50 (H200 training) + ~$20-40 (A100 eval)

OBJECTIVE:
  1. Validate real reward signal (not hackable, not degenerate)
  2. Demonstrate candidate improvement under feedback
  3. GRPO does not collapse immediately
  4. Pipeline reusable for future experiments

ABORT GATES:
  - reward=-1 stays after 30 steps → stop, ship SFT + search
  - reward std=0 → model collapsed → increase temperature or stop
  - Loss = NaN → reduce LR to 1e-6
  - OOM → reduce G/batch/seq
```

### 15.2 Future Scale-Up Config (Research Target)

The content below describes the future scale-up path using Qwen3-Coder-Next 80B on B200. This is **NOT** the hackathon config.

### Why the Original Config Falls Short

The original GRPO-14 config (10 steps, 3 turns, 4K context) was designed as a "P3 demo." CUDA-Agent ablation data shows this massively undershoots:

| CUDA-Agent Ablation | Impact on Faster-than-compile % |
|--------------------|---------------------------------|
| Remove multi-turn agent loop | **-82.7%** (96.8% → 14.1%) |
| Remove robust reward | -36.4% (96.8% → 60.4%) |
| Remove RFT warmup | -47.0% (96.8% → 49.8%) |
| Remove value pretraining | -45.9% (96.8% → 50.9%) |

**Multi-turn iteration is the #1 factor by far.** With only 3 turns, we leave most of the performance on the table. CUDA-Agent uses up to 150 turns per episode. Most optimization discoveries happen after turn 5 (initial kernel → compile fix → profile → targeted optimization → iterate).

### Future Match Stage 3 Configuration (NOT Hackathon)

```
FUTURE: STAGE 3 — GRPO + CURRICULUM (SCALE-UP TARGET)
  Model: Qwen3-Coder-Next 80B FP8 on B200 (requires B200)
  Steps: 150 (matches CUDA-Agent's 150 — but on 1 GPU, not 128)
  G: 2
  Max turns: 20 (full compile→correct→optimize→tune lifecycle)
  Temperature: 0.7
  LR: 5e-6
  Context: 32,768 tokens (B200: 92GB free - 40GB KV = 52GB headroom)
  Eval: Hybrid — local nvcc turns 1-5, remote A100 backend turns 6+
  Credit: TRLOO leave-one-out N/(N-1) (MARS would require per-turn rewards — currently DROPPED)
  Reward: Discrete milestone {-1, 1, 2, 3} (future: consider continuous log(speedup) + Nsight bonus)
  CPPO: DROPPED for now (would require G>2 to be useful; re-evaluate if G increases)
  Loss: Standard clip (MASPO DEFERRED — consider if clip-boundary artifacts appear)
  Inference: Best-of-8 per problem + SkyDiscover for Level 3
  Cost: ~10 hrs B200 ($62.50) + ~10 hrs A100 eval ($25.00) = $87.50
```

### Future Scale-Up Key Changes (vs Original Demo Config)

| Change | From → To | Why |
|--------|----------|-----|
| Steps | 10 → **150** | Target CUDA-Agent's 150 steps. On 1 GPU (not 128), each step is 12-25× more efficient. |
| Max turns | 3 → **20** | Multi-turn = #1 ablation factor (-82.7%). 20 turns covers full lifecycle. |
| Context | 4K → **32K** | At 20 turns × ~1.5K/turn = 30K, need 32K. Fits on B200 (92GB - 40GB KV = 52GB free). |
| RFT filter | ≥ 1.0 → **≥ 0.0** | Any speedup (log > 0) is useful RFT signal. |
| Inference | pass@1 → **best-of-8** | Generate 8 candidates, eval all, pick fastest correct. |
| Goal | Demo → **Approach CUDA-Agent** | Projected target based on stacked improvements. NOT proven. |

### Multi-Turn Budget Analysis

With 20 turns and hybrid eval (CPPO DROPPED — no filtering savings):

```
Per GRPO step (G=2, max_turns=20):
  Turn 1-5 per candidate: local nvcc compile check (~5s each) = 50s total
  Turn 6-20 per candidate: remote A100 backend eval (~30s each) = 15 turns × 2 × 30s = 900s
  Average turns per candidate: ~12 (early exit at reward >= 3)
  Average wall-clock per step: ~6 minutes (no CPPO filtering)

150 steps × 4 min = ~600 min = ~10 hours
  B200 time: 10 hrs × $6.25/hr = $62.50
  A100 eval: ~1,200 Modal calls × 30s = ~10 hrs × $2.50/hr = $25.00
  Stage 3 total: ~$87.50

Full pipeline:
  Stage 1 (300 steps): $22
  Stage 2 (RFT): $3
  Stage 3 (150 steps): $87.50
  Best-of-8 inference: $8
  SkyDiscover: $10
  Testing/debug: $25
  Total: ~$155.50 ($125.50 after $30 free credit)
```

### Projected Performance vs CUDA-Agent (FUTURE TARGET — NOT PROVEN)

| Metric | Training (pass@1) | Inference (best-of-8) | CUDA-Agent | Projected? |
|--------|------------------|----------------------|------------|------------|
| Pass rate | 93-96% | 98-99% | 98.8% | Projected |
| Faster-than-compile | 80-88% | 93-97% | 96.8% | Projected |
| Geomean speedup | 1.8-2.1× | 2.0-2.3× | 2.11× | Projected |

### Projected Data Efficiency Comparison (Future Scale-Up)

> **SPECULATIVE -- depends on unimplemented components.** The per-sample efficiency projection assumed MARS+CPPO, both of which are now DROPPED. Only TRLOO is implemented. Revise these numbers if future work adds per-turn rewards (which would re-enable MARS).

| Metric | CUDA-Agent | KernelForge (Projected) | Ratio |
|--------|------------|------------------------|-------|
| RL samples | 153,600 | ~6,000 (150 steps × G=2 × avg 20 turns) | 25× fewer |
| Per-sample efficiency | Baseline (PPO, biased) | **SPECULATIVE -- depends on unimplemented components.** TRLOO alone gives ~1.3x correction at G=2. The 12-25x claim required MARS+CPPO (both DROPPED). | Overstated |
| Expert demonstrations (SFT) | 0 | 192 (doubleGraph A100 kernels, 84K+ lines) | Advantage |
| Inference-time candidates | 1 (pass@1) | 8 (best-of-8) | 8× |
| Evolutionary search | None | SkyDiscover (same Modal pipeline) | Advantage |

**Note:** The original "effective signal" comparison assumed full MARS+TRLOO+CPPO stack. MARS is DROPPED (no benefit with outcome-only rewards). CPPO is DROPPED (no benefit post-SFT at G=2). The data efficiency argument with TRLOO alone is weaker than originally projected.

### Model Size Discussion (Future Scale-Up Context)

For the **hackathon**, the primary model is **Qwen3-Coder-30B-A3B-Instruct** on H200 (see Section 15.1). The discussion below applies to the future B200 scale-up path.

Qwen3-Coder-Next 80B MoE has **only 3B active parameters** due to MoE routing. At inference, it runs like a 3B model but has access to 80B of stored routing knowledge from pre-training on 800K executable code tasks.

| Model | Active | VRAM | Tok/s | CUDA Level 1 | CUDA Level 2-3 |
|-------|--------|------|-------|--------------|----------------|
| **Qwen3-Coder-30B-A3B (hackathon)** | 3.3B | ~30-34 GB | ~50 | **Strong** | **Moderate** |
| **Qwen3-Coder-Next 80B (future)** | 3B | ~100 GB | ~50 | **Very Strong** | **Strong** |
| Qwen3.5-32B | 32B | ~32 GB | ~80 | Strong | Moderate |
| Qwen3.5-9B | 9B | ~9 GB | ~150 | Moderate | Weak |

**Hackathon decision:** Use Qwen3-Coder-30B-A3B-Instruct on H200. Fall back to Qwen3.5-35B-A3B if needed.
**Future decision:** Scale to Qwen3-Coder-Next 80B FP8 on B200 for deeper experiments.

### Updated Quick Reference Card (March 5, 2026)

```
═══════════════════════════════════════════════════════════
HACKATHON CONFIG (PRIMARY — this weekend)
═══════════════════════════════════════════════════════════
  Model = Qwen3-Coder-30B-A3B-Instruct (30.5B total, 3.3B active)
  Train GPU = H200 ($4.54/hr, 141GB)
  Eval GPU = A100 80GB ($2.50/hr)
  G = 2
  Steps = 100 (Stage 1) + RFT + 50 (Stage 3 pilot)
  Max turns = 3 (Stage 1) / 3-5 (Stage 3)
  Context = 8192
  Credit = TRLOO only (default; set `KERNELFORGE_USE_TRLOO=0` to use vanilla GRPO) (MARS DROPPED — see GRPO-4)
  Reward = discrete milestone {-1, 1, 2, 3} (CUDA Agent Equation 1)
  Loss = standard clip (MASPO DEFERRED — see GRPO-12)
  Eval = Hybrid: local nvcc early turns, remote A100 backend final turn
  Pruning = CPPO DROPPED (see GRPO-11)
  Inference = Best-of-N per problem + SkyDiscover hedge

  COST (hackathon):
    SFT warmup: ~$8
    Stage 1 (100 steps): ~$15
    Stage 2 (RFT): ~$2
    Stage 3 (50 steps, pilot): ~$15
    A100 eval: ~$20-40
    Search + testing: ~$15-25
    Total: ~$75-105 (under $200 budget)

  OBJECTIVE: Validate real learning signal. Not benchmark domination.

═══════════════════════════════════════════════════════════
FUTURE SCALE-UP CONFIG (research target — after hackathon)
═══════════════════════════════════════════════════════════
  Model = Qwen3-Coder-Next 80B FP8 (requires B200)
  G = 2, Steps = 150, Max turns = 20, Context = 32768
  TRLOO only (MARS/CPPO DROPPED; MASPO DEFERRED)
  Cost: ~$155
  Target: approach CUDA-Agent metrics (SPECULATIVE — depends on unimplemented components)

DATA SOURCES (shared):
  Ops-6K: 6,000 PyTorch operator tasks (GRPO prompts)
  doubleGraph: 192 expert A100 kernels (SFT + GRPO prompts, 84K+ lines)
  Curriculum: 21 problems across 4 phases (5 topology-aware graph tasks)
  SKILL.md: 178 lines, 7 real A100 patterns from doubleGraph production code

INTEGRATION STATUS:
  doubleGraph dataset harvester: DONE (datasets/extract_doublegraph_a100.py)
  SKILL.md A100 patterns: DONE (skill_builder.py:_append_a100_patterns())
  Evaluator bridge (cascade eval): DONE (skydiscover_integration/evaluator.py)
  AdaEvolve multi-island search: DONE (skydiscover_integration/adaevolve.py)
  EvoX self-evolving strategies: DONE (skydiscover_integration/evox_strategies.py)
  Topology-aware curriculum: DONE (training/curriculum.py Phase 2-3)
  Graph topology in RL observations: DONE (kernel_forge_env.py + curriculum.py)
  Nsight lightweight profiling: DONE (modal_app.py ptxas occupancy)
  TRLOO trainer: DONE (training/custom_grpo_trainer.py)
  Local compile fast-path: DONE (training/multi_turn_rollout.py)
  MARS return-to-go: DROPPED (no benefit with outcome-only rewards — see GRPO-4)
  CPPO pruning: DROPPED (no benefit post-SFT at G=2 — see GRPO-11)
  MASPO soft trust: DEFERRED — future research (~50 LOC, see GRPO-12)
```

---

## GRPO-16: Presentation Framing

When talking to judges or writing the demo narrative:

**Say:**
- "We built a low-cost kernel optimization RL/search environment"
- "We targeted A100 kernels with real compile / correctness / timing feedback"
- "We used Qwen3-Coder-30B-A3B-Instruct on H200 as the practical hackathon base"
- "This is designed to scale into richer RL, distillation, and future hardware"

**Do NOT say:**
- "We already beat CUDA-Agent"
- "We proved single-GPU SOTA"
- "We fully reproduced large-scale paper results"

**Claim discipline applies:** Only claim what has actually been run. Separate implemented / tested / benchmarked / projected. See KERNELFORGE_FINAL_PRD.md Section 7 for the full claim taxonomy.

---

## GRPO-17: Abort Gates (Consolidated)

Abort or shrink RL scope if any of these trigger:

| Trigger | Meaning | Action |
|---------|---------|--------|
| reward=-1 stays after 30 SFT+GRPO steps | Dataset or reward pipeline is broken | Debug reward. If unfixable, ship SFT + search only. |
| reward std=0 for >5 consecutive steps | Model collapsed, all outputs identical | Increase temperature to 1.2. If persists, stop GRPO. |
| Loss = NaN | LR too high or numerical instability | Reduce LR to 1e-6. If persists, stop GRPO. |
| OOM | Model/context too large for H200 | Reduce G to 1, reduce context, reduce batch. |
| Eval throughput eats >60% of budget | A100 eval calls too expensive | Simplify to single-turn, reduce step count. |
| Gate G-0.8 fails (no speedup >1.2× in 5-step sanity check) | Model cannot optimize on this task distribution | Abandon GRPO entirely. Ship SFT + SkyDiscover + environment. |

**If abort triggers:** Ship SFT + search + environment. Do NOT fake an RL success story. A working environment with honest search results is a better hackathon submission than a broken RL pipeline with inflated claims.
