<chatName="SkyDiscover AdaEvolve & EvoX Deep Dive"/>

# SkyDiscover: AdaEvolve & EvoX — Complete Technical Reference

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [AdaEvolve: Adaptive Multi-Island Search](#2-adaevolve)
3. [EvoX: Self-Evolving Search Strategies](#3-evox)
4. [Tuning for Search Accuracy](#4-tuning)
5. [Practical Usage Guide](#5-usage)
6. [Integration with CUDA Kernel Optimization](#6-integration)

---

## 1. Architecture Overview

SkyDiscover runs a single loop regardless of algorithm:

```
sample → prompt → LLM generate → evaluate → add
  ↑                                           │
  └───────────────────────────────────────────┘
```

Every algorithm plugs into this loop by implementing two things:

| Level | What you implement | Class |
|-------|-------------------|-------|
| **Database only** | `add()` + `sample()` | Subclass `ProgramDatabase` |
| **Database + Controller** | `add()` + `sample()` + `run_discovery()` | Subclass both |

The `Program` dataclass is the universal unit:

```python
@dataclass
class Program:
    id: str                          # UUID
    solution: str                    # Source code / prompt text
    metrics: Dict[str, Any]          # Must include 'combined_score'
    parent_id: Optional[str]         # Who was mutated to create this
    other_context_ids: List[str]     # Additional context programs shown to LLM
    iteration_found: int             # When discovered
    artifacts: Dict[str, Any]        # Evaluator diagnostic output
    generation: int                  # Depth in mutation tree
```

The key metric is **`combined_score`** — this is what every algorithm maximizes. Your evaluator must return it.

---

## 2. AdaEvolve: Adaptive Multi-Island Search

**Paper:** [arxiv.org/abs/2602.20133](https://arxiv.org/abs/2602.20133)

AdaEvolve treats LLM-driven search as **non-stationary zeroth-order optimization** and adapts at three levels simultaneously.

### 2.1 Core Data Structures

#### Per-Island Adaptive State (`AdaptiveState` in `adaptation.py`)

Each island maintains an **accumulated improvement signal G** — an exponential moving average of squared normalized improvements:

```
When program improves island's best score:
    δ_raw = fitness - best_score
    δ_normalized = min(δ_raw / (|best_score| + ε), 1.0)    # Scale-invariant, capped
    G_t = ρ · G_{t-1} + (1 - ρ) · δ²_normalized           # EMA of squared deltas

Search intensity:
    I = I_min + (I_max - I_min) / (1 + √(G + ε))
```

**Interpretation:**
- **High G** (island is finding improvements) → intensity approaches `I_min` → **exploit**
- **Low G** (island is stagnating) → intensity approaches `I_max` → **explore**

This is analogous to how AdaGrad/Adam adapt learning rates, but for gradient-free search.

```python
# From adaptation.py — the actual implementation
class AdaptiveState:
    accumulated_signal: float = 0.0    # G
    best_score: float = float("-inf")
    decay: float = 0.9                 # ρ
    intensity_min: float = 0.1         # I_min
    intensity_max: float = 0.7         # I_max

    def record_evaluation(self, fitness: float) -> float:
        if fitness > self.best_score:
            raw_delta = fitness - self.best_score
            normalized_delta = self._normalize_delta(raw_delta)  # scale-invariant
            self.best_score = fitness
            self.accumulated_signal = (
                self.decay * self.accumulated_signal 
                + (1 - self.decay) * (normalized_delta ** 2)
            )
            return normalized_delta
        return 0.0

    def get_search_intensity(self) -> float:
        return self.intensity_min + (self.intensity_max - self.intensity_min) / (
            1 + math.sqrt(self.accumulated_signal + self.epsilon)
        )
```

#### Multi-Dimensional Adapter (`MultiDimensionalAdapter` in `adaptation.py`)

Manages adaptive state across all islands with **UCB island selection**. Critical design: **dual normalization**:

1. **Search Intensity** (per-island): Uses **local** best for scale-invariant adaptation
2. **UCB Rewards** (cross-island): Uses **global** best for fair comparison

This fixes the "Poor Island Bias" where trash islands with high local percentage gains would dominate UCB:

```python
# UCB formula
reward_avg = decayed_rewards[i] / decayed_visits[i]   # Recent reward per recent visit
exploration_bonus = C * sqrt(ln(N) / raw_visits[i])    # Classic UCB exploration
ucb_score = reward_avg + exploration_bonus
```

**Key:** Both rewards and visits decay at the same rate (`ρ`). Without decayed visits, `reward_avg = decaying_sum / growing_count → 0`, making all islands look equally unproductive.

#### UnifiedArchive (`archive/unified_archive.py`)

Each island has a **quality-diversity archive** that maintains both high-fitness and high-novelty programs. Programs are scored by:

```
elite_score = fitness_weight · fitness_percentile 
            + novelty_weight · novelty_percentile
            + pareto_weight  · pareto_percentile    (if multi-objective)
```

Where:
- **fitness_percentile**: position when sorted by `combined_score` / n
- **novelty_percentile**: position when sorted by k-NN distance / n (using pluggable `DiversityStrategy`)
- **pareto_percentile**: NSGA-II rank + crowding distance (opt-in via `pareto_objectives`)

**Eviction** uses **deterministic crowding**: when at capacity, find the most similar NON-PROTECTED program and replace it if the new program has higher elite score. Top programs by elite score + the overall best by fitness + Pareto front members are protected.

### 2.2 The Sampling Decision

When the controller calls `database.sample()`, the decision cascade is:

```
1. Get search intensity I for current island:
     adaptive → I = I_min + (I_max - I_min) / (1 + √(G + ε))
     fixed    → I = fixed_intensity (ablation)

2. Roll random r ∈ [0, 1):
     r < I                        → EXPLORATION  (probability = I)
     r < I + (1-I) × 0.7         → EXPLOITATION  (probability = (1-I) × 0.7)
     else                         → BALANCED       (probability = (1-I) × 0.3)

3. Sample parent based on mode:
     EXPLORATION  → novelty-proportional sampling from archive
     EXPLOITATION → random from top-25% by fitness (or Pareto front)
     BALANCED     → 50/50 coin flip between exploration and exploitation

4. Sample context programs (hybrid):
     local_count  = num_context × local_ratio (default 60%)  → most different from parent in archive
     global_count = num_context × (1 - local_ratio)          → top programs across ALL islands
```

The mode label (`EXPLORE_LABEL` or `EXPLOIT_LABEL`) is injected into the LLM prompt as structured guidance:

```
## PARENT SELECTION CONTEXT
This parent was selected through diversity-driven sampling to explore different regions.

### EXPLORATION GUIDANCE
- Consider alternative algorithmic approaches
- Don't be constrained by the parent's approach
...
```

### 2.3 Island Lifecycle

```
for each iteration:
    1. Check paradigm stagnation → generate breakthrough ideas if needed
    2. Generate child (sample → prompt → LLM → evaluate)
    3. Add to current island's archive
    4. end_iteration():
        a. Check dynamic island spawning (if global productivity < threshold)
        b. Select next island (UCB with decayed rewards OR round-robin)
        c. Ring migration (every migration_interval iterations)
```

**Migration** uses ring topology: island `i` copies top `migration_count` programs to island `(i+1) % num_islands`. Migrants are marked as external improvements — they update the destination's `best_score` and `G` but NOT UCB rewards or visit counts:

```python
def receive_external_improvement(self, fitness):
    # Updates best_score and G → triggers exploitation mode
    # Does NOT update improvement_count or total_evaluations
    # → UCB stats remain unaffected (island didn't earn this)
```

**Dynamic Island Spawning** triggers when:
1. `use_dynamic_islands = true`
2. `num_islands < max_islands`
3. Cooldown elapsed since last spawn
4. `global_productivity < spawn_productivity_threshold`

New islands are created with **heterogeneous configurations** from presets: `balanced`, `quality`, `diversity`, `pareto`, `exploration` — each with different weight distributions for elite score computation. The system picks the least-used preset.

### 2.4 Paradigm Breakthrough System

When the global improvement rate drops below threshold, the system generates high-level strategy shifts:

```
ParadigmTracker monitors:
    improvement_history: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  (last window_size iterations)
    improvement_rate = sum(history) / len(history)

    Stagnation triggers when:
        len(history) >= window_size  AND
        improvement_rate < improvement_threshold  AND
        no active paradigms available

ParadigmGenerator produces (via LLM):
    [
        {
            "idea": "Use scipy.optimize.minimize with SLSQP",
            "description": "Apply scipy.optimize.minimize directly...",
            "what_to_optimize": "combined_score",
            "approach_type": "scipy.optimize.minimize"
        },
        ...  (num_paradigms_to_generate ideas)
    ]
```

Paradigms are **round-robin rotated** and each used `max_paradigm_uses` times. When a paradigm is active:
- The **best program** (not a random parent) is used as parent
- The paradigm guidance is injected into the prompt
- After all paradigms are exhausted, they're archived with outcome data (SUCCESS/FAILED + score improvement)
- Previously tried ideas are fed back to the generator to avoid repeating failures

### 2.5 AdaEvolve Control Flow Summary

```
AdaEvolveController.run_discovery():
│
├── _setup_iteration_stats_logging()    # JSONL file for post-hoc analysis
├── _ensure_all_islands_seeded()        # Copy seed to empty islands
│
└── for iteration in range(start, total):
    │
    ├── if paradigm_stagnating:
    │   └── _generate_paradigms_if_needed()
    │       └── ParadigmGenerator.generate()  →  database.set_paradigms()
    │
    ├── _run_normal_step(iteration):
    │   └── _generate_child(iteration, error_context):
    │       ├── database.sample()               # Returns parent_dict + context_dict
    │       ├── context_builder.build_prompt()   # Injects mode label + paradigm + siblings
    │       ├── llm.generate()                   # LLM call
    │       ├── parse response (diff or full rewrite)
    │       └── evaluator.evaluate_program()     # Score the candidate
    │
    ├── _process_result():
    │   └── database.add(child, iteration)       # Archive handles quality-diversity
    │
    └── database.end_iteration(iteration):
        ├── _should_spawn_island() → _spawn_island()
        ├── adapter.select_dimension_ucb()       # Pick next island
        └── if migration_interval: _migrate()     # Ring migration
```

---

## 3. EvoX: Self-Evolving Search Strategies

**Paper:** [arxiv.org/abs/2602.23413](https://arxiv.org/abs/2602.23413)

EvoX takes a radical approach: instead of designing the search strategy, it **co-evolves** the search strategy alongside the solutions. The LLM writes new `ProgramDatabase` subclasses as Python code, which are hot-swapped into the running system.

### 3.1 Two Levels of Co-Evolution

**Inner loop — solution discovery:**
A solution database (initially `EvolvedProgramDatabase` from `initial_search_strategy.py`) manages the population. The standard `sample → prompt → LLM → evaluate → add` loop runs.

**Outer loop — search strategy evolution:**
When solutions stagnate, the LLM generates an entirely new database class. This new class is:
1. **Validated** by `search_strategy_evaluator.py` (structural checks + functional tests)
2. **Hot-swapped** — all programs migrated to the new database
3. **Scored** by how much solutions improved during its active window

### 3.2 The Search Strategy Scoring Formula

From `search_scorer.py`:

```python
improvement = running_best - start_score
log_weight  = 1.0 + log(1.0 + max(0.0, start_score))
score       = improvement * log_weight / sqrt(horizon)
```

**Interpretation:**
- `improvement`: Raw score gain during this strategy's active window
- `log_weight`: Algorithms that improve an already-strong solution are rewarded more (diminishing returns at high scores are harder)
- `sqrt(horizon)`: Normalizes by window length (longer windows need proportionally more improvement)

### 3.3 Stagnation Detection & Strategy Switching

```python
DEFAULT_SWITCH_RATIO = 0.10          # 10% of total iterations
DEFAULT_IMPROVEMENT_THRESHOLD = 0.01

_switch_interval = max(1, int(max_iterations * 0.10))

def _should_evolve_search(self) -> bool:
    current = self._get_best_score()
    if (current - last_tracked) > 0.01:
        stagnant_count = 0      # Reset on improvement
    else:
        stagnant_count += 1     # Increment on stagnation
    
    return stagnant_count >= _switch_interval
```

### 3.4 Variation Operators

Before the main loop, EvoX generates **problem-specific variation operators** via LLM. These are structured prompt templates that tell the LLM HOW to vary solutions:

- **DIVERGE (exploration):** "Use a FUNDAMENTALLY DIFFERENT APPROACH" — lists alternative libraries, algorithms, paradigms specific to the problem
- **REFINE (exploitation):** "Keep the same fundamental approach, SQUEEZE the last percent" — lists hyperparameter tuning, computational budget, polish strategies

The generation prompt (`variation_operator_generator.py`) is sophisticated — it analyzes:
- Problem type (optimization, combinatorial, graph, etc.)
- Available packages from `requirements.txt`
- Evaluator scoring function components
- Constraint types and handling strategies

### 3.5 Database Hot-Swapping

When a new strategy is generated, the switch process:

```python
def _switch_to_new_search_algorithm(self, result):
    # 1. Write generated code to temp file
    # 2. load_database_from_file() — imports the new class
    # 3. Validate: must have EvolvedProgramDatabase(ProgramDatabase), 
    #    correct sample() signature, etc.
    # 4. Create new database instance
    # 5. Migrate ALL programs from old → new database
    # 6. Wrap add() to ensure _update_best_program always called (safety)
    # 7. Save fallback (old database + old code) for rollback
    # 8. Switch: self.database = new_db
```

If the new database crashes at runtime, the system **automatically restores** the previous strategy and preserves any programs that were successfully found:

```python
def _restore_fallback_database(self):
    # Migrate new programs found during broken strategy's runs
    for pid, program in broken_db.programs.items():
        if pid not in old_db.programs:
            old_db.add(program)
    self.database = old_db  # Restore
```

### 3.6 The Evaluator for Search Strategies

`search_strategy_evaluator.py` is ~700 LOC of validation that runs BEFORE a new strategy goes live:

1. **Structural checks:** Class named `EvolvedProgramDatabase`, inherits `ProgramDatabase`, `sample()` has correct signature
2. **Add operations:** Metrics preserved after `add()` (no accidental mutation)
3. **Sample contract:** Returns `Tuple[Dict[str, Program], Dict[str, List[Program]]]`, exactly one parent, context count ≤ requested
4. **Error handling:** Programs with error strings in metrics don't crash sampling
5. **Migration compatibility:** Must handle base `Program` instances (not just `EvolvedProgram`)
6. **Metric immutability:** All original metric values preserved through add→store→sample→retrieve cycle

### 3.7 EvoX Control Flow Summary

```
CoEvolutionController.run_discovery():
│
├── _generate_variation_operators()     # LLM generates problem-specific explore/exploit templates
├── _initialize_first_search_program()  # Score the seed strategy
│
└── for iteration in range(start, total):
    │
    ├── _run_iteration(iteration)       # Standard solution generation
    ├── _process_iteration_result()     # Add to database
    ├── _record_search_window_step()    # Track best score for strategy scoring
    │
    └── if _should_evolve_search():     # Stagnation detected
        └── _evolve_search():
            ├── _finalize_pending_search()    # Score previous strategy
            ├── _reset_search_window()        # Start new scoring window
            └── _generate_and_validate_search_algorithm():
                ├── search_controller.run_discovery(max_iterations=1)  # Generate new DB class
                ├── Validate generated code
                ├── _switch_to_new_search_algorithm()  # Hot-swap with fallback
                └── On failure: _restore_fallback_database()
```

---

## 4. Tuning for Search Accuracy

### 4.1 Algorithm Selection Guide

| Scenario | Best Algorithm | Why |
|----------|---------------|-----|
| **Quick runs (<50 iter)** | `topk` or `best_of_n` | Minimal overhead, works well with few iterations |
| **Standard optimization** | `adaevolve` | Adaptive exploration/exploitation, paradigm breakthroughs for escaping local optima |
| **Unknown problem structure** | `evox` | Self-discovers what sampling strategy works for your specific problem |
| **Multi-objective** | `adaevolve` with `pareto_objectives` | NSGA-II ranking + crowding distance in archive |
| **Very long runs (500+ iter)** | `evox` | Strategy adaptation becomes valuable over long horizons |
| **Prompt optimization** | `adaevolve` | Has specific prompt-optimization labels and paradigm generators |

### 4.2 AdaEvolve Key Parameters

```yaml
search:
  type: "adaevolve"
  database:
    # === MOST IMPACTFUL PARAMETERS ===
    
    num_islands: 2
    # More islands = more diversity but slower convergence per island
    # 2-3 for focused problems, 4-5 for open-ended problems
    
    decay: 0.9
    # How quickly past improvements are forgotten
    # Higher (0.95) = longer memory, slower adaptation
    # Lower (0.8) = shorter memory, more reactive but noisy
    
    intensity_min: 0.15    # Exploitation floor (how much explore even when productive)
    intensity_max: 0.5     # Exploration ceiling (how much explore when stagnating)
    # Narrow range [0.3, 0.4] → conservative, always balanced
    # Wide range [0.1, 0.7] → dramatic shifts between explore/exploit
    
    # === MIGRATION (cross-pollination) ===
    migration_interval: 15   # Too frequent = homogenization, too rare = isolation
    migration_count: 5       # Programs to migrate (ring topology)
    
    # === ARCHIVE QUALITY-DIVERSITY ===
    fitness_weight: 1.0      # Pure fitness selection (quality-focused)
    novelty_weight: 0.0      # No novelty bonus (set to 0.3 for diversity)
    diversity_strategy: "code"  # "code" (structure), "metric" (performance), "hybrid"
    
    # === PARADIGM BREAKTHROUGH ===
    use_paradigm_breakthrough: true
    paradigm_window_size: 10        # Iterations to measure improvement rate over
    paradigm_improvement_threshold: 0.12  # Below this rate → trigger paradigm generation
    paradigm_max_uses: 2            # Uses per paradigm idea
    paradigm_num_to_generate: 3     # Ideas per paradigm batch
    
    # === DYNAMIC ISLANDS ===
    use_dynamic_islands: true
    max_islands: 5
    spawn_productivity_threshold: 0.015  # Spawn if no island is productive
    spawn_cooldown_iterations: 30        # Min iterations between spawns
```

**Tuning heuristics:**
- If the best score plateaus early → **lower** `paradigm_improvement_threshold` (trigger breakthroughs sooner), **increase** `intensity_max`
- If solutions oscillate / regress → **increase** `decay` (longer memory), **decrease** `intensity_max`
- If all islands converge to same solution → **increase** `novelty_weight` to 0.3, **increase** `migration_interval`
- If progress is steady but slow → **decrease** `num_islands` to 2, focus compute

### 4.3 EvoX Key Parameters

```yaml
search:
  type: "evox"
  database:
    auto_generate_variation_operators: true  # LLM generates problem-specific templates
    # Set to false for default templates (faster startup, less tailored)
```

EvoX has fewer explicit knobs — its core mechanism IS adaptation. The main levers:
- **`max_iterations`**: Needs enough iterations for strategy evolution to pay off (≥100)
- **LLM quality**: Strategy generation benefits from stronger models (guide_models config)
- **`auto_generate_variation_operators`**: Significantly impacts quality for domain-specific problems

### 4.4 LLM Configuration Tips

```yaml
llm:
  models:
    - name: "gpt-5"           # Strong model for solution generation
      weight: 1.0
  guide_models:
    - name: "gpt-5"           # Can use a different (cheaper/stronger) model for paradigm/strategy generation
      weight: 1.0
  temperature: 0.7             # Higher → more diverse solutions, lower → more conservative
  top_p: 0.95
  max_tokens: 32000            # Increase for complex solutions (CUDA kernels need ~8-16K)
```

**Multi-model strategy** (supported natively):
```yaml
llm:
  models:
    - name: "gpt-5"
      weight: 0.6             # 60% of generations
    - name: "gemini/gemini-3-pro"
      weight: 0.4             # 40% of generations — adds diversity
```

---

## 5. Practical Usage Guide

### 5.1 CLI Usage

```bash
# Basic run with TopK (simplest)
uv run skydiscover-run initial_program.py evaluator.py \
  --model gpt-5 --iterations 50

# AdaEvolve with config
uv run skydiscover-run initial_program.py evaluator.py \
  --config config_adaevolve.yaml --search adaevolve --iterations 100

# EvoX
uv run skydiscover-run initial_program.py evaluator.py \
  --search evox --model gpt-5 --iterations 200

# Resume from checkpoint
uv run skydiscover-run initial_program.py evaluator.py \
  --config config.yaml --checkpoint outputs/adaevolve/problem_0305_1430/checkpoints/checkpoint_50

# From scratch (no initial program)
uv run skydiscover-run evaluator.py --search adaevolve --model gpt-5
```

### 5.2 Python API

```python
from skydiscover import run_discovery

# Minimal usage
result = run_discovery(
    evaluator="evaluator.py",
    initial_program="initial_program.py",
    model="gpt-5",
    iterations=100,
    search="adaevolve",
)
print(f"Best score: {result.best_score:.4f}")
print(f"Improvement: {result.initial_score:.4f} → {result.best_score:.4f}")
print(result.best_solution)

# With full config control
from skydiscover.config import Config, load_config

config = load_config("config_adaevolve.yaml")
# Override anything programmatically:
config.search.database.num_islands = 3
config.search.database.paradigm_improvement_threshold = 0.08
config.llm.temperature = 0.9

result = run_discovery(
    evaluator="evaluator.py",
    initial_program="initial_program.py",
    config=config,
    iterations=200,
    output_dir="my_experiment/",
)
```

### 5.3 Writing an Evaluator

Your evaluator must define an `evaluate(program_path: str) -> dict` function:

```python
# evaluator.py
def evaluate(program_path: str) -> dict:
    """
    Args:
        program_path: Path to the generated solution file
    
    Returns:
        Dict with at minimum 'combined_score' (float, higher is better)
    """
    # Load the generated solution
    with open(program_path, 'r') as f:
        code = f.read()
    
    # Execute and test it
    try:
        exec(compile(code, program_path, 'exec'), namespace := {})
        func = namespace['solve']
        
        # Run on test cases
        correct = 0
        total = 10
        for case in test_cases:
            result = func(case.input)
            if result == case.expected:
                correct += 1
        
        accuracy = correct / total
        
        return {
            'combined_score': accuracy,      # REQUIRED: main metric
            'accuracy': accuracy,            # Optional: extra metrics
            'runs_successfully': 1.0,        # Optional: for diagnostics
        }
    except Exception as e:
        return {
            'combined_score': 0.0,
            'error': str(e),
            'runs_successfully': 0.0,
        }
```

**For CUDA kernel evaluation**, you can delegate to Modal or local GPU:

```python
# evaluator.py for CUDA kernels
def evaluate(program_path: str) -> dict:
    """Stage 1: local compile check. Stage 2: GPU execution."""
    with open(program_path) as f:
        code = f.read()
    
    # Stage 1: Syntax + compile check (fast, no GPU)
    try:
        compile(code, program_path, 'exec')
    except SyntaxError as e:
        return {'combined_score': 0.0, 'error': f'Syntax error: {e}'}
    
    # Stage 2: Run on GPU (slow, but accurate)
    try:
        result = run_on_gpu(code)  # Your GPU evaluation function
        speedup = result.reference_time / result.kernel_time
        correctness = result.max_error < 1e-3
        
        return {
            'combined_score': speedup if correctness else 0.0,
            'speedup': speedup,
            'correctness': float(correctness),
            'max_error': result.max_error,
        }
    except Exception as e:
        return {'combined_score': 0.0, 'error': str(e)}
```

### 5.4 Config File Structure

A complete config for CUDA kernel optimization:

```yaml
max_iterations: 100
checkpoint_interval: 5
log_level: "INFO"

llm:
  models:
    - name: "gpt-5"
      weight: 1.0
  temperature: 0.8          # Slightly creative for kernel optimization
  max_tokens: 32000
  timeout: 600

search:
  type: "adaevolve"
  num_context_programs: 4
  database:
    num_islands: 3
    population_size: 20
    decay: 0.9
    intensity_min: 0.15
    intensity_max: 0.5
    use_paradigm_breakthrough: true
    paradigm_window_size: 8
    paradigm_improvement_threshold: 0.10
    paradigm_num_to_generate: 3
    use_dynamic_islands: true
    max_islands: 5
    migration_interval: 10
    migration_count: 3

prompt:
  system_message: |
    You are an expert CUDA/Triton kernel engineer. Given a kernel implementation,
    optimize it for maximum throughput on NVIDIA H200 GPU.
    
    Focus on: memory coalescing, shared memory tiling, warp-level primitives,
    register pressure, occupancy, instruction-level parallelism.
    
    The kernel will be scored by speedup over a reference implementation.
    Correctness (max error < 1e-3) is required.

evaluator:
  timeout: 120              # CUDA compilation can be slow
  max_retries: 3
  cascade_evaluation: true
  cascade_thresholds: [0.3, 0.6]

diff_based_generation: true   # Generate diffs, not full rewrites
max_solution_length: 60000

monitor:
  enabled: true               # Live dashboard at http://localhost:8765
  port: 8765
```

### 5.5 Monitor Dashboard

Enable with `monitor.enabled: true` in config. Provides:
- Real-time score trajectory
- Per-iteration program details
- Best solution code viewer
- Human feedback injection (write guidance to `human_feedback.md`)

---

## 6. Integration with CUDA Kernel Optimization (KernelForge)

### 6.1 Why NOT Reimplement From Scratch

The plan proposes implementing simplified AdaEvolve (~200 LOC) and EvoX (~150 LOC). This would miss critical capabilities:

| Feature | Simplified Version | Full SkyDiscover |
|---------|-------------------|------------------|
| Adaptive intensity | ❌ | ✅ Scale-invariant G signal with dual normalization |
| UCB island selection | ❌ | ✅ Decayed magnitude rewards preventing breakthrough memory |
| Quality-diversity archive | ❌ | ✅ UnifiedArchive with deterministic crowding + k-NN novelty |
| Paradigm breakthroughs | ❌ | ✅ LLM-generated strategy shifts with outcome tracking |
| Dynamic island spawning | ❌ | ✅ Heterogeneous island configs from presets |
| Search strategy co-evolution | ❌ | ✅ Hot-swappable database classes with fallback |
| Checkpointing & resume | ❌ | ✅ Full state serialization including archive genealogy |
| Live monitoring | ❌ | ✅ WebSocket dashboard with human feedback |
| Diff-based generation | ❌ | ✅ SEARCH/REPLACE blocks for targeted edits |
| Multi-model support | ❌ | ✅ Weighted model pool with provider-agnostic routing |
| Evaluator feedback | ❌ | ✅ Artifacts injected into prompts for targeted improvement |

### 6.2 Integration Architecture

The correct approach is to use SkyDiscover as a library:

```
KernelForge Project
├── kernels/                     # CUDA kernel benchmarks
│   ├── bfs/
│   │   ├── initial_program.py   # Seed kernel
│   │   ├── evaluator.py         # Compile + run + measure speedup
│   │   ├── config.yaml          # SkyDiscover config for this kernel
│   │   └── reference.py         # Reference implementation
│   ├── pagerank/
│   └── ...
├── run_kernelforge.py           # Orchestration script
└── evaluation/
    └── modal_bridge.py          # Modal GPU evaluation backend
```

**Evaluator bridge** — the key integration point:

```python
# kernels/bfs/evaluator.py
import subprocess
import json

def evaluate(program_path: str) -> dict:
    """Two-stage evaluation: local compile + remote GPU execution."""
    
    with open(program_path) as f:
        code = f.read()
    
    # Stage 1: Local compile check (fast, catches syntax errors)
    try:
        compile(code, program_path, 'exec')
    except SyntaxError as e:
        return {'combined_score': 0.0, 'error': str(e), 'validity': 0}
    
    # Stage 2: GPU execution via Modal
    result = subprocess.run(
        ['modal', 'run', 'evaluation/modal_bridge.py', '--kernel', program_path,
         '--graph-type', 'power-law', '--num-vertices', '1000000'],
        capture_output=True, text=True, timeout=300
    )
    
    if result.returncode != 0:
        return {'combined_score': 0.0, 'error': result.stderr, 'validity': 0}
    
    metrics = json.loads(result.stdout)
    
    # Build combined score from multiple objectives
    speedup = metrics.get('speedup', 0.0)
    correctness = metrics.get('correctness', 0.0)
    
    combined = speedup * correctness  # Zero if incorrect
    
    return {
        'combined_score': combined,
        'speedup': speedup,
        'correctness': correctness,
        'execution_time_ms': metrics.get('execution_time_ms', 0),
        # Evaluator feedback for paradigm generator
        'feedback': metrics.get('failure_analysis', ''),
    }
```

**Orchestration script:**

```python
# run_kernelforge.py
from skydiscover import run_discovery

KERNELS = [
    ("kernels/bfs/initial_program.py", "kernels/bfs/evaluator.py", "kernels/bfs/config.yaml"),
    ("kernels/pagerank/initial_program.py", "kernels/pagerank/evaluator.py", "kernels/pagerank/config.yaml"),
]

for initial, evaluator, config in KERNELS:
    result = run_discovery(
        evaluator=evaluator,
        initial_program=initial,
        config=config,
        iterations=100,
        output_dir=f"outputs/{initial.split('/')[1]}",
    )
    print(f"{initial}: {result.initial_score:.2f} → {result.best_score:.2f}")
```

### 6.3 Topology-Aware Optimization

To add graph topology data to the RL observation (observation #10678), use **evaluator artifacts**:

```python
# In evaluator.py, return topology info as feedback
def evaluate(program_path: str) -> dict:
    ...
    return {
        'combined_score': combined,
        # This gets injected into the LLM prompt via AdaEvolveContextBuilder
        'feedback': f"""
Graph topology: {graph_type} (avg_degree={avg_degree}, max_degree={max_degree})
Optimal pattern: {'warp-level for hubs' if graph_type == 'power-law' else 'thread-level uniform'}
Current bottleneck: {bottleneck_analysis}
""",
    }
```

AdaEvolve's `AdaEvolveContextBuilder` automatically extracts `artifacts["feedback"]` and injects it into the `{search_guidance}` section of the prompt.

### 6.4 Installation

```bash
# Install SkyDiscover
pip install skydiscover
# or from source:
cd skydiscover && pip install -e .

# Run with AdaEvolve
skydiscover-run initial_kernel.py evaluator.py \
  --search adaevolve --model gpt-5 --iterations 100
```