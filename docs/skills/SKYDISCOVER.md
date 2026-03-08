<chatName="SkyDiscover SKILLS.md Technical Reference"/>

I'll create the comprehensive SKILLS.md document. Let me write it directly based on the extensive codebase context provided.

---

# SKILLS.md — SkyDiscover Technical Reference

Below is the complete SKILLS.md document. Given the scope, I'll structure it as a standalone reference covering architecture, all algorithms, configuration, evaluation, prompt construction, and extension patterns.

---

## Document Structure

```
 1. System Overview & Architecture
 2. Core Loop: sample → prompt → generate → evaluate → add
 3. Entry Points (CLI, Python API, Runner)
 4. Configuration System
 5. Search Algorithms
    5.1  TopK (baseline)
    5.2  Best-of-N
    5.3  Beam Search
    5.4  AdaEvolve (deep dive)
    5.5  EvoX (deep dive)
    5.6  GEPA Native (deep dive)
    5.7  OpenEvolve Native
 6. Program Database API
 7. Evaluation Pipeline
 8. Context Builder & Prompt Construction
 9. LLM Pool & Model Configuration
10. Extension Guide
11. Checkpointing & Resume
12. Live Monitor & Human Feedback
13. Key Formulas & Algorithms Reference
14. Configuration Reference (all YAML fields)
15. Troubleshooting & Anti-Patterns
```

---

Here is the full document content:

````markdown
# SKILLS.md — SkyDiscover Complete Technical Reference

> **Purpose**: A standalone reference that enables understanding, using, and extending SkyDiscover without reading source code.
>
> **Version**: Current as of March 2026.

---

## Table of Contents

1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Core Loop](#2-core-loop)
3. [Entry Points](#3-entry-points)
4. [Configuration System](#4-configuration-system)
5. [Search Algorithms](#5-search-algorithms)
6. [Program Database API](#6-program-database-api)
7. [Evaluation Pipeline](#7-evaluation-pipeline)
8. [Context Builder & Prompt Construction](#8-context-builder--prompt-construction)
9. [LLM Pool & Model Configuration](#9-llm-pool--model-configuration)
10. [Extension Guide](#10-extension-guide)
11. [Checkpointing & Resume](#11-checkpointing--resume)
12. [Live Monitor & Human Feedback](#12-live-monitor--human-feedback)
13. [Key Formulas & Algorithms Reference](#13-key-formulas--algorithms-reference)
14. [Configuration Reference](#14-configuration-reference)
15. [Troubleshooting & Anti-Patterns](#15-troubleshooting--anti-patterns)

---

## 1. System Overview & Architecture

SkyDiscover is an iterative LLM-driven discovery engine. It evolves solutions (code, prompts, or images) by repeatedly mutating the best candidates using large language models and scoring them with user-provided evaluators.

### Component Map

```
┌──────────────────────────────────────────────────────────────┐
│                        CLI / Python API                       │
│                  api.py  ·  cli.py  ·  runner.py              │
└──────────────┬───────────────────────────────────┬────────────┘
               │                                   │
    ┌──────────▼──────────┐             ┌──────────▼──────────┐
    │    Config System     │             │   Runner (runner.py) │
    │     config.py        │◄────────────│  loads config        │
    │  YAML + overrides    │             │  creates database    │
    └──────────────────────┘             │  creates controller  │
                                         │  runs loop           │
                                         │  saves checkpoints   │
                                         └──────────┬───────────┘
                                                    │
                              ┌──────────────────────▼────────────────────────┐
                              │        DiscoveryController                    │
                              │  default_discovery_controller.py              │
                              │                                               │
                              │  Orchestrates the per-iteration cycle:        │
                              │  sample → prompt → generate → evaluate → add  │
                              └──┬──────┬──────────┬───────────┬──────────────┘
                                 │      │          │           │
                    ┌────────────▼┐  ┌──▼────────┐ │  ┌────────▼────────┐
                    │ ProgramDB   │  │ Context   │ │  │   Evaluator     │
                    │ (search/)   │  │ Builder   │ │  │ evaluation/     │
                    │ add/sample  │  │ prompts   │ │  │ evaluate()      │
                    └─────────────┘  └───────────┘ │  └─────────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │    LLM Pool     │
                                          │   llm/          │
                                          │ weighted sample │
                                          └─────────────────┘
```

### Key Source Files

| File | Role |
|------|------|
| `api.py` | Public API: `run_discovery()`, `discover_solution()` |
| `cli.py` | CLI entry point: `skydiscover-run` |
| `runner.py` | Top-level orchestrator: config → database → controller → loop |
| `config.py` | All configuration dataclasses + YAML loading |
| `search/base_database.py` | `Program` dataclass + `ProgramDatabase` ABC |
| `search/default_discovery_controller.py` | Default iteration loop + shared primitives |
| `search/registry.py` | Factory functions + registration |
| `search/route.py` | Search type → class routing |
| `context_builder/base.py` | `ContextBuilder` ABC |
| `context_builder/default/builder.py` | Default prompt construction |
| `evaluation/evaluator.py` | Program evaluation with cascade + LLM judge |
| `llm/llm_pool.py` | Weighted LLM backend sampling |
| `llm/openai.py` | OpenAI-compatible API client |

### Data Types

```python
@dataclass
class Program:
    id: str                          # UUID
    solution: str                    # Source code, prompt text, or image description
    language: str = "python"         # "python", "text", "prompt", "image", "cpp"
    metrics: Dict[str, Any]          # Evaluation results, must include "combined_score"
    iteration_found: int = 0         # When this program was created
    parent_id: Optional[str] = None  # Parent program ID
    parent_info: Optional[Tuple[str, str]] = None   # (label, parent_id)
    context_info: Optional[List[Tuple[str, str]]]    # [(label, context_id), ...]
    other_context_ids: Optional[List[str]] = None
    metadata: Dict[str, Any]         # Arbitrary (changes summary, parent metrics, etc.)
    artifacts: Dict[str, Any]        # Evaluator-produced artifacts (feedback, diagnostics)
    generation: int = 0              # Depth in the mutation tree
    prompts: Optional[Dict] = None   # Logged prompts (if log_prompts=True)
    timestamp: float                 # time.time() at creation
```

---

## 2. Core Loop

Every iteration executes five steps:

```
sample → prompt → generate → evaluate → add
  ↑                                       │
  └───────────────────────────────────────┘
```

### Step-by-Step

1. **Sample** — The database picks a parent program and context programs.
   - Returns: `(parent_dict, context_programs_dict)`
   - `parent_dict`: `{label_string: Program}` — label carries search guidance
   - `context_programs_dict`: `{label_string: [Program, ...]}` — other examples for the LLM

2. **Prompt** — The context builder assembles system + user messages.
   - Injects: parent solution, context programs, metrics, search guidance, failed attempts
   - Modes: diff-based (SEARCH/REPLACE blocks), full rewrite, prompt optimization, image generation

3. **Generate** — The LLM produces a candidate.
   - Text mode: parse code from response (diff or full rewrite)
   - Image mode: VLM generates an image
   - Agentic mode: multi-turn tool-calling agent loop

4. **Evaluate** — User's `evaluate(program_path)` scores the candidate.
   - Returns metrics dict (must include `combined_score` for best tracking)
   - Optional: cascade evaluation (stage1 → threshold → stage2)
   - Optional: LLM-as-judge appends `llm_*` metrics

5. **Add** — Scored candidate stored back in database.
   - Database handles population management (eviction, island assignment, etc.)
   - Tracks best program globally
   - Persists to disk if `db_path` is set

### Retry Logic

The default controller retries up to `retry_times` (default 3) per iteration:
- Parse failures → rebuild prompt with error context
- Evaluation failures (validity=0, timeout) → retry with failed attempt context
- LLM errors → immediate return (no retry)

---

## 3. Entry Points

### CLI

```bash
# Basic usage
uv run skydiscover-run initial_program.py evaluator.py \
  --config config.yaml --model gpt-5 --iterations 100

# With search algorithm
uv run skydiscover-run initial_program.py evaluator.py \
  --search adaevolve --model gpt-5 --iterations 200

# From scratch (no initial program)
uv run skydiscover-run evaluator.py --model gpt-5 --iterations 50

# Resume from checkpoint
uv run skydiscover-run initial_program.py evaluator.py \
  --checkpoint outputs/adaevolve/problem_0305_1200/checkpoints/checkpoint_50
```

### Python API

```python
from skydiscover import run_discovery, discover_solution

# File-based
result = run_discovery(
    evaluator="evaluator.py",
    initial_program="initial_program.py",
    model="gpt-5",
    iterations=100,
    search="adaevolve",
    system_prompt="Optimize a circle packing algorithm.",
)
print(result.best_score, result.best_solution)

# Inline evaluator + solution
result = discover_solution(
    evaluator=lambda path: {"combined_score": score_function(path)},
    initial_solution="def solve(): return 42",
    model="gpt-5",
    iterations=50,
)

# Multi-model
result = run_discovery(
    evaluator="eval.py",
    initial_program="init.py",
    model="gpt-5,gemini/gemini-3-pro",
    search="evox",
    iterations=200,
)
```

### DiscoveryResult

```python
@dataclass
class DiscoveryResult:
    best_program: Optional[Program]   # Full program object
    best_score: float                 # combined_score of best
    best_solution: str                # Source code / prompt text
    metrics: Dict[str, Any]           # All metrics of best program
    output_dir: Optional[str]         # Where results are saved
    initial_score: Optional[float]    # Score of the seed program
```

---

## 4. Configuration System

### Hierarchy

```
Config
├── llm: LLMConfig
│   ├── models: List[LLMModelConfig]           # Solution generation
│   ├── evaluator_models: List[LLMModelConfig]  # LLM-as-judge
│   └── guide_models: List[LLMModelConfig]      # Paradigm generation, labels
├── context_builder: ContextBuilderConfig
├── search: SearchConfig
│   ├── type: str                               # "topk", "adaevolve", "evox", etc.
│   └── database: DatabaseConfig (subclass)     # Algorithm-specific config
├── evaluator: EvaluatorConfig
├── agentic: AgenticConfig
└── monitor: MonitorConfig
```

### Provider Auto-Detection

Model strings are parsed to determine provider, API base, and API key env vars:

| Model string | Provider | API base |
|-------------|----------|----------|
| `gpt-5` | openai | `https://api.openai.com/v1` |
| `gemini/gemini-3-pro` | gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `claude-4-opus` | anthropic | `https://api.anthropic.com/v1/` |
| `deepseek-coder` | deepseek | `https://api.deepseek.com/v1` |
| `ollama/llama3` | ollama | (requires explicit `api_base`) |

API keys are resolved from environment variables automatically (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`).

### YAML Config Example

```yaml
# General
max_iterations: 200
checkpoint_interval: 10
language: python
diff_based_generation: true
max_parallel_iterations: 1

# LLM
llm:
  models:
    - name: gpt-5
      temperature: 0.7
      max_tokens: 32000
  guide_models:
    - name: gpt-5
      temperature: 0.9
  evaluator_models:
    - name: gpt-5-mini

# Prompt
prompt:
  system_message: |-
    You are an expert algorithm researcher...
  template: default

# Search
search:
  type: adaevolve
  num_context_programs: 4
  database:
    num_islands: 2
    decay: 0.9
    intensity_min: 0.15
    intensity_max: 0.5
    use_paradigm_breakthrough: true

# Evaluator
evaluator:
  timeout: 360
  cascade_evaluation: true
  cascade_thresholds: [0.3, 0.6]

# Monitor
monitor:
  enabled: true
  port: 8765
```

### Runtime Overrides

`apply_overrides()` lets CLI/API arguments take priority over YAML:

```python
apply_overrides(
    config,
    model="gpt-5",           # Overwrites config.llm.models
    api_base="http://...",    # Overwrites all model api_base
    search="adaevolve",       # Overwrites config.search.type + database config class
    system_prompt="...",      # Overwrites config.context_builder.system_message
    agentic=True,             # Enables agentic mode
)
```

---

## 5. Search Algorithms

### 5.0 Algorithm Comparison

| Algorithm | Type | Key Idea | Best For |
|-----------|------|----------|----------|
| `topk` | Database only | Top-1 parent + next-K context | Simple baselines |
| `best_of_n` | Database only | Keep best N programs | Quick experiments |
| `beam_search` | Database only | Beam-width selection with diversity | Structured search spaces |
| `adaevolve` | DB + Controller | Adaptive multi-island search with UCB + paradigm breakthroughs | Production use, long runs |
| `evox` | DB + Controller | Co-evolves the search algorithm itself | Meta-optimization |
| `gepa_native` | DB + Controller | Acceptance gating + reflective prompting + merge | Quality-focused search |
| `openevolve_native` | Database only | MAP-Elites + island-based | Feature-space coverage |

### 5.1 TopK (Baseline)

**File**: `search/topk/database.py` (56 lines)

Simplest algorithm. Always uses the best program as parent and the next K programs as context.

```
Parent:  rank 1 (best program)
Context: ranks 2..K+1
```

No population management, no eviction. All programs are kept forever.

### 5.2 Best-of-N

**File**: `search/best_of_n/database.py`

Maintains a fixed-size pool of the best N programs. Parent selection is random from the pool.

### 5.3 Beam Search

**File**: `search/beam_search/database.py`

Maintains a beam of width W. Selection uses diversity-weighted scoring to balance quality and coverage.

---

### 5.4 AdaEvolve — Adaptive Multi-Island Search (Deep Dive)

**Paper**: [AdaEvolve](https://arxiv.org/abs/2602.20133)

**Files**:
```
search/adaevolve/
├── controller.py        # AdaEvolveController
├── database.py          # AdaEvolveDatabase
├── adaptation.py        # AdaptiveState + MultiDimensionalAdapter
├── archive/
│   ├── unified_archive.py  # UnifiedArchive (quality-diversity)
│   └── diversity.py        # DiversityStrategy implementations
└── paradigm/
    ├── generator.py     # ParadigmGenerator (LLM breakthrough ideas)
    └── tracker.py       # ParadigmTracker (stagnation detection)
```

#### Three Levels of Adaptation

**Level 1 — Global: Island Scheduling (UCB)**

Multiple subpopulations (islands) evolve in parallel. A UCB bandit with decayed rewards selects which island gets compute next.

```python
# UCB selection (MultiDimensionalAdapter.select_dimension_ucb)
for each island i:
    reward_avg = decayed_rewards[i] / decayed_visits[i]
    exploration_bonus = C * sqrt(ln(total_iterations) / raw_visits[i])
    ucb_score = reward_avg + exploration_bonus

select island with highest ucb_score
```

**Key design**: Two separate normalizations prevent "Poor Island Bias":
- **Search intensity** uses LOCAL best → scale-invariant per-island adaptation
- **UCB rewards** use GLOBAL best → fair cross-island comparison

Without this, a low-quality island with a large local percentage improvement would dominate UCB over globally productive islands.

**Level 2 — Local: Exploration vs. Exploitation**

Each island tracks an accumulated improvement signal G (exponential moving average of squared normalized deltas):

```
G_t = ρ · G_{t-1} + (1 - ρ) · δ²

where δ = min(raw_delta / (|best_score| + ε), 1.0)  # normalized, capped
```

Search intensity is computed as:

```
intensity = I_min + (I_max - I_min) / (1 + √(G + ε))
```

| G value | Meaning | Intensity | Behavior |
|---------|---------|-----------|----------|
| High | Island is productive | Low (→ I_min) | Exploit: sample from top fitness |
| Low | Island is stagnating | High (→ I_max) | Explore: sample by novelty |

Sampling mode selection per iteration:
```
rand = random()
if rand < intensity:       mode = "exploration"
elif rand < intensity + (1-intensity)*0.7:  mode = "exploitation"
else:                      mode = "balanced"
```

**Level 3 — Meta: Paradigm Breakthroughs**

When global improvement rate drops below a threshold (e.g., < 5% of iterations improve over a window of 30), the system generates high-level strategy shifts:

1. **ParadigmTracker** monitors improvement rate across all islands
2. When stagnating: **ParadigmGenerator** calls the LLM with a 6-step analysis framework
3. LLM produces structured paradigm ideas (JSON array):
   ```json
   [
     {
       "idea": "Use scipy.optimize.minimize with SLSQP",
       "description": "Apply direct optimization to all variables...",
       "what_to_optimize": "combined_score",
       "cautions": "Use multiple starting points",
       "approach_type": "scipy.optimize.minimize"
     }
   ]
   ```
4. Paradigms are injected into prompts for the next N iterations
5. Each paradigm is used `max_paradigm_uses` times, then archived with outcome data
6. Previously tried paradigms (with SUCCESS/FAILED labels) are fed back to the generator

#### AdaEvolve Database Architecture

```
AdaEvolveDatabase
├── num_islands × UnifiedArchive  (quality-diversity per island)
├── MultiDimensionalAdapter       (UCB + per-island AdaptiveState)
├── ParadigmTracker               (paradigm lifecycle)
└── Migration ring topology       (periodic top-program transfer)
```

**UnifiedArchive** (per island):
- Fixed-size flat list with quality-diversity balance
- **Elite Score** = `fitness_weight × fitness_percentile + novelty_weight × novelty_percentile`
  - Optional: `+ pareto_objectives_weight × pareto_percentile` (NSGA-II ranking)
- **Novelty**: Average k-NN distance using pluggable DiversityStrategy
- **Eviction**: Deterministic crowding — new program competes with its most similar non-protected neighbor
- **Protection**: Top programs by elite score + best by raw fitness + Pareto front members
- **Genealogy**: Parent-child tracking for sibling context

**DiversityStrategy** options:
| Strategy | Based On | Best For |
|----------|----------|----------|
| `code` | Token Jaccard + structural features + length | Code optimization |
| `text` | Token Jaccard + length (no structure) | Prompt optimization |
| `metric` | Normalized Euclidean in metric space | Multi-objective problems |
| `hybrid` | Weighted combination | Balanced diversity |

**Migration** (ring topology):
- Every `migration_interval` iterations
- Island i → Island (i+1) % num_islands
- Top `migration_count` programs copied (deduplicated)
- Migrants update recipient's best_score + G but NOT UCB stats (island didn't earn it)

**Dynamic Island Spawning**:
- When global productivity < `spawn_productivity_threshold`
- Cooldown: `spawn_cooldown_iterations` between spawns
- New island uses underused configuration preset (balanced, quality, diversity, pareto, exploration)
- Seeded with top 5 programs from existing islands

#### AdaEvolve Controller Flow

```python
async def run_discovery(start, max_iterations):
    setup_iteration_stats_logging()
    ensure_all_islands_seeded()

    for iteration in range(start, start + max_iterations):
        # Check for paradigm stagnation
        if database.is_paradigm_stagnating():
            await generate_paradigms_if_needed()

        # Generate and evaluate
        result = await run_normal_step(iteration)  # with retry

        if result.error:
            log_failed_stats(iteration)
        else:
            process_result(result, iteration)
            log_success_stats(iteration)

        # CRITICAL: end-of-iteration housekeeping
        database.end_iteration(iteration)
        # → dynamic island spawning check
        # → UCB island selection for next iteration
        # → periodic migration
```

**Paradigm-aware generation**:
When a paradigm is active, the controller:
1. Uses the best program as parent (not the sampled parent)
2. Injects paradigm guidance into prompt via AdaEvolveContextBuilder
3. Records paradigm usage
4. Archives paradigm with outcome when exhausted

**Sibling context**:
Previous children of the same parent are shown to the LLM to avoid repeating failed mutations:
```
## Previous attempts on this parent
[Attempt 1] Score: 0.72 — used scipy.optimize.minimize
[Attempt 2] Score: 0.68 — used grid search
```

#### AdaEvolve Config

```yaml
search:
  type: adaevolve
  num_context_programs: 4
  database:
    # Island structure
    num_islands: 2
    population_size: 20

    # Adaptive search
    decay: 0.9                    # ρ — EMA decay factor
    intensity_min: 0.15           # Exploitation floor
    intensity_max: 0.5            # Exploration ceiling

    # Migration
    migration_interval: 15
    migration_count: 5

    # Archive
    use_unified_archive: true
    archive_elite_ratio: 0.2
    fitness_weight: 1.0
    novelty_weight: 0.0
    k_neighbors: 5
    diversity_strategy: code      # "code", "text", "metric", "hybrid"

    # Dynamic islands
    use_dynamic_islands: true
    max_islands: 5
    spawn_productivity_threshold: 0.015
    spawn_cooldown_iterations: 30

    # Paradigm breakthrough
    use_paradigm_breakthrough: true
    paradigm_window_size: 10
    paradigm_improvement_threshold: 0.12
    paradigm_max_uses: 2
    paradigm_num_to_generate: 3

    # Ablation flags (set false to disable)
    use_adaptive_search: true     # false → fixed_intensity
    use_ucb_selection: true       # false → round-robin
    use_migration: true
```

---

### 5.5 EvoX — Self-Evolving Search Strategies (Deep Dive)

**Paper**: [EvoX](https://arxiv.org/abs/2602.23413)

**Files**:
```
search/evox/
├── controller.py                    # CoEvolutionController
├── database/
│   ├── initial_search_strategy.py   # EvolvedProgramDatabase (seed)
│   ├── search_strategy_db.py        # SearchStrategyDatabase (stores strategies)
│   └── search_strategy_evaluator.py # Validates generated databases
├── config/
│   ├── search.yaml                  # Search-side config
│   └── evox_search_sys_prompt.txt   # System prompt for strategy generation
└── utils/
    ├── search_scorer.py             # LogWindowScorer
    ├── variation_operator_generator.py
    ├── template.py
    └── coevolve_logging.py
```

#### Core Idea: Two Levels of Co-Evolution

**Inner loop (solution discovery)**: A solution database stores candidates. The LLM generates new candidates; the evaluator scores them. The database implementation is not fixed — it gets hot-swapped by the outer loop.

**Outer loop (search strategy evolution)**: When solutions stagnate, the LLM generates an entirely new `EvolvedProgramDatabase` class (as Python code). The new strategy is validated, all programs are migrated, and solution discovery resumes.

#### Algorithm

```
for each iteration:
    1. Sample parent + context from current solution database
    2. Generate solution candidate via LLM, evaluate, store
    3. Track stagnation: if improvement < 0.01 for switch_interval iterations:
        a. Score current search algorithm by solution improvement
        b. Generate new EvolvedProgramDatabase class via LLM
        c. Validate new database (structural + functional checks)
        d. Migrate all programs to new database, resume
    4. If new database fails at runtime: restore previous database, preserve new programs
```

#### Search Algorithm Scoring

The `LogWindowScorer` evaluates how well a search strategy performed during its active window:

```
score = improvement × (1 + log(1 + start_score)) / √(horizon)
```

- **improvement**: `best_score_at_end - best_score_at_start`
- **start_score**: Score when the strategy was activated (log-weighted: higher start → harder to improve → more credit)
- **horizon**: Number of iterations the strategy ran (√ normalization: longer runs aren't unfairly penalized)

#### Stagnation Detection

```python
DEFAULT_SWITCH_RATIO = 0.10   # 10% of total iterations
DEFAULT_IMPROVEMENT_THRESHOLD = 0.01

switch_interval = max(1, int(max_iterations * 0.10))

# Each iteration:
if (current_best - last_tracked_best) > 0.01:
    stagnant_count = 0
else:
    stagnant_count += 1

if stagnant_count >= switch_interval:
    evolve_search()
```

#### Database Hot-Swap Process

1. **Score pending strategy** via `LogWindowScorer`
2. **Generate new strategy** via `search_controller._run_iteration()`
   - LLM receives: current strategy code, scoring metrics, database statistics
   - LLM outputs: complete Python class implementing `add()` + `sample()`
3. **Validate**: Structural checks (has required methods) + metric preservation
4. **Write to temp file** and dynamically import
5. **Migrate**: Copy all programs from old database to new, preserve prompts
6. **Assign variation operators** (diverge/refine labels)
7. **Install fallback**: Keep old database in case new one fails
8. **Wrap `add()`**: Ensure `_update_best_program()` is always called

#### Fallback/Restore

If the new database causes an error during `add()` or `sample()`:
1. Programs discovered during the failed strategy's successful runs are migrated back
2. Previous database is restored as active
3. Previous search code is restored
4. Failed attempt is counted (`_num_search_evolutions += 1`)

#### Variation Operators

Before the main loop, the controller generates problem-specific "diverge" and "refine" labels via LLM. These are injected into the solution database's sampling labels:

- **Diverge label**: Encourages fundamentally different approaches
- **Refine label**: Encourages incremental improvement on working solutions

Can be disabled with `auto_generate_variation_operators: false` to use defaults.

#### EvoX Config

```yaml
search:
  type: evox
  database:
    database_file_path: null          # Auto-detected (initial_search_strategy.py)
    evaluation_file: null             # Auto-detected (search_strategy_evaluator.py)
    config_path: null                 # Auto-detected (search.yaml)
    auto_generate_variation_operators: true
```

---

### 5.6 GEPA Native — Guided Evolution (Deep Dive)

**Files**:
```
search/gepa_native/
├── controller.py     # GEPANativeController
├── database.py       # GEPANativeDatabase
└── pareto_utils.py   # Pareto-front selection utilities
```

#### Three Core Ideas

**1. Reflective Prompting**

Recent rejected programs and their evaluator diagnostics are formatted into the LLM prompt as actionable feedback. The LLM sees:
- What was tried and failed
- Why it failed (error messages, low scores)
- The parent score that wasn't beaten

This is injected via the `{search_guidance}` placeholder in the prompt template.

**2. Acceptance Gating**

A child is accepted ONLY if its fitness strictly exceeds the parent's:

```python
if child_score <= parent_score:
    database.add_rejected(child)  # Store for reflective prompting
    return False  # Rejected
```

This prevents population pollution — bad mutations never enter the elite pool.

**3. LLM-Mediated Merge**

Two complementary programs are merged by the LLM:

- **Proactive**: After each accepted mutation, schedule a merge
- **Reactive**: After `merge_after_stagnation` iterations without improvement

Merge candidate selection:
1. **Preferred**: Two programs that each lead on a different metric
2. **Fallback**: Best program + random from top 5

Merge acceptance criterion: merged score ≥ max(score_a, score_b)

Guards:
- Deduplication (pair_key tracking)
- Budget (max_merge_attempts)
- Self-merge prevention

#### GEPA Database

**Elite Pool**: Fixed-size list sorted by fitness (descending).
- Eviction: Remove weakest members when pool exceeds `population_size`
- Pin: Best program + initial program are never evicted

**Parent Selection**:
| Strategy | Behavior |
|----------|----------|
| `epsilon_greedy` (default) | Best with prob (1-ε), random otherwise |
| `best` | Always pick highest score |
| `pareto` | Frequency-weighted from Pareto front across metrics |

**Per-Metric Best Tracking**: Tracks which program leads on each individual metric. Used for merge candidate selection.

**Rejection History**: Bounded deque (default 20). Programs are NOT added to `self.programs` — only stored for reflective prompting.

#### GEPA Config

```yaml
search:
  type: gepa_native
  database:
    population_size: 40
    candidate_selection_strategy: epsilon_greedy
    epsilon: 0.1
    acceptance_gating: true
    use_merge: true
    merge_after_stagnation: 15
    max_merge_attempts: 10
    max_recent_failures: 5
```

---

### 5.7 OpenEvolve Native

**File**: `search/openevolve_native/database.py`

MAP-Elites + island-based search. Maintains a feature grid where each cell holds the best program for that feature combination. Islands provide population diversity.

---

## 6. Program Database API

### Abstract Interface (ProgramDatabase)

| Method | Abstract? | Purpose |
|--------|-----------|---------|
| `add(program, iteration)` | **Yes** | Store a scored program |
| `sample(num_context_programs)` | **Yes** | Select parent + context |
| `save(path, iteration)` | No | Checkpoint to disk |
| `load(path)` | No | Restore from checkpoint |
| `_update_best_program(program)` | No | Track best (call from `add`) |
| `get_best_program(metric?)` | No | Return highest-scoring program |
| `get_top_programs(n, metric?)` | No | Return top N by score |
| `get(program_id)` | No | Retrieve by ID |
| `log_status()` | No | Log database summary |
| `get_statistics(...)` | No | Return stats dict for prompt context |
| `log_prompt(...)` | No | Store prompt/response for a program |

### `sample()` Return Format

Two valid return formats:

```python
# Plain (used by TopK, BestOfN)
return parent_program, [context_program_1, context_program_2]

# Dict-wrapped (used by AdaEvolve, GEPA, EvoX)
return {"exploration_label": parent}, {"": [context_1, context_2]}
```

The dict key is a label string injected into the prompt. The controller normalizes both formats.

### `get_statistics()` Output

Used by the context builder for prompt construction:

```python
{
    "population_size": int,
    "solution_score_summary": {
        "best": float, "q75": float, "q50": float, "q25": float, "worst": float,
        "score_tiers": {...},
        "unique_scores": int,
    },
    "top_solution_scores": [float, ...],
    "avg_solutions_per_parent": float,
    "recent_solution_stats": {
        "execution_trace": [...],
        "score_trajectory": [float, ...],
        "iterations_without_improvement": int,
        "most_reused_parent_ratio": float,
        ...
    },
    "previous_programs": [Program, ...],
}
```

---

## 7. Evaluation Pipeline

### User-Provided Evaluator

The evaluator is a Python file with an `evaluate(program_path)` function:

```python
def evaluate(program_path: str) -> dict:
    """Score a candidate program.

    Args:
        program_path: Path to temp file containing the candidate solution.

    Returns:
        Dict with at least 'combined_score' (float).
        Optional keys: any additional metrics, 'error', 'validity'.
    """
    # Import and run the candidate
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    score = run_tests(module)
    return {"combined_score": score, "accuracy": acc, "speed": spd}
```

### Cascade Evaluation

When `cascade_evaluation: true`, the evaluator can define staged evaluation:

```python
def evaluate_stage1(program_path):
    """Quick validation — runs fast, filters obvious failures."""
    return {"combined_score": quick_score}

def evaluate_stage2(program_path):
    """Full evaluation — runs slow, computes all metrics."""
    return {"combined_score": full_score, "accuracy": acc}
```

Flow: `stage1 → threshold check → stage2 → merge metrics`

Thresholds: `cascade_thresholds: [0.3, 0.6]` — stage1 score must exceed 0.3 to proceed.

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    metrics: Dict[str, Any]      # Scores (must include combined_score)
    artifacts: Dict[str, Any]    # Diagnostic info (feedback, stderr, etc.)
```

Artifacts are stored on the Program and can be used by:
- Paradigm generator (evaluator feedback)
- GEPA reflective prompting
- Merge prompt construction

### LLM-as-Judge

When `llm_as_judge: true`, an LLM scores programs alongside the evaluator:
- Uses `evaluator_system_message` template
- Appends `llm_*` prefixed metrics to the evaluation result
- Stored in artifacts for transparency

---

## 8. Context Builder & Prompt Construction

### Template System

Templates are `.txt` files with `{placeholder}` substitution:

| Template | When Used |
|----------|-----------|
| `system_message.txt` | Default system prompt |
| `diff_user_message.txt` | Diff-based generation (SEARCH/REPLACE) |
| `full_rewrite_user_message.txt` | Full rewrite mode |
| `full_rewrite_prompt_opt_user_message.txt` | Prompt optimization |
| `image_user_message.txt` | Image generation |
| `evaluator_system_message.txt` | LLM judge system |
| `evaluator_user_message.txt` | LLM judge user |

### Key Placeholders

| Placeholder | Source |
|-------------|--------|
| `{system_message}` | `config.context_builder.system_message` |
| `{current_program}` | Parent solution code |
| `{current_program_metrics}` | Parent metrics formatted |
| `{other_context_programs}` | Context programs with scores |
| `{search_guidance}` | Algorithm-specific guidance (explore/exploit labels, paradigm, siblings) |
| `{previous_attempt}` | Failed attempt context for retries |
| `{errors}` | Error messages from failed evaluations |

### Prompt Modes

1. **Diff-based** (default): LLM outputs SEARCH/REPLACE blocks
   ```
   <<<<<<< SEARCH
   old code
   =======
   new code
   >>>>>>> REPLACE
   ```

2. **Full rewrite**: LLM outputs complete solution in a code block

3. **Prompt optimization**: For `language: text` — LLM outputs new prompt text

4. **Image mode**: For `language: image` — VLM generates an image

### Custom Context Builders

Extend `DefaultContextBuilder` and inject via `{search_guidance}`:

```python
class MyContextBuilder(DefaultContextBuilder):
    def build_prompt(self, current_program, context=None, **kwargs):
        guidance = self._format_my_guidance(context.get("my_data"))
        return super().build_prompt(
            current_program, context,
            search_guidance=guidance,
            **kwargs,
        )
```

Algorithm-specific builders:
- `AdaEvolveContextBuilder`: paradigm guidance, sibling context, error context
- `GEPANativeContextBuilder`: rejection history, evaluator diagnostics
- `EvoxContextBuilder`: LLM-generated problem summaries, search statistics

---

## 9. LLM Pool & Model Configuration

### LLMPool

Weighted sampling over one or more LLM backends:

```python
pool = LLMPool(models_cfg=[
    LLMModelConfig(name="gpt-5", weight=0.7),
    LLMModelConfig(name="gemini-3-pro", weight=0.3),
])

# Each generate() call samples one model by weight
response = await pool.generate(system_message, messages)

# generate_all() calls all models concurrently
responses = await pool.generate_all(system_message, messages)
```

### Three Model Pools

| Pool | Config Key | Used By |
|------|-----------|---------|
| `llms` | `llm.models` | Solution generation |
| `evaluator_llms` | `llm.evaluator_models` | LLM-as-judge |
| `guide_llms` | `llm.guide_models` | Paradigm generation, variation operators |

If `evaluator_models` or `guide_models` are not specified, they default to `models`.

### LLMModelConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `name` | None | Model name (e.g., "gpt-5") |
| `api_base` | Auto-detected | API endpoint URL |
| `api_key` | From env | API key |
| `weight` | 1.0 | Sampling weight in pool |
| `temperature` | 0.7 | Generation temperature |
| `top_p` | 0.95 | Nucleus sampling |
| `max_tokens` | 32000 | Max output tokens |
| `timeout` | 600 | Request timeout (seconds) |
| `retries` | 3 | Number of retries |
| `retry_delay` | 5 | Delay between retries (seconds) |
| `reasoning_effort` | None | For reasoning models |

---

## 10. Extension Guide

### Level 1: Database Only

Implement `add()` + `sample()`. Default controller handles the loop.

```python
from skydiscover.search.base_database import Program, ProgramDatabase

class MyDatabase(ProgramDatabase):
    def __init__(self, name, config):
        super().__init__(name, config)

    def add(self, program, iteration=None, **kwargs):
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)  # REQUIRED
        return program.id

    def sample(self, num_context_programs=4, **kwargs):
        parent = ...   # selection logic
        context = ...  # context programs
        return parent, context
        # OR: return {"label": parent}, {"": context}
```

Register in `route.py`:
```python
register_database("my_algo", MyDatabase)
```

### Level 2: Database + Controller

Override `run_discovery()` for cross-iteration behavior.

```python
from skydiscover.search.default_discovery_controller import (
    DiscoveryController, DiscoveryControllerInput
)

class MyController(DiscoveryController):
    def __init__(self, controller_input):
        super().__init__(controller_input)

    async def run_discovery(self, start_iteration, max_iterations, **kwargs):
        for iteration in range(start_iteration, start_iteration + max_iterations):
            if self.shutdown_event.is_set():
                break

            result = await self._run_iteration(iteration, retry_times=3)

            if result.error:
                continue

            # Custom logic: acceptance gating, stagnation, etc.
            self._process_iteration_result(
                result, iteration, kwargs.get("checkpoint_callback")
            )

        return self.database.get_best_program()
```

Register both:
```python
register_database("my_algo", MyDatabase)
register_controller("my_algo", MyController)
```

### Adding Config

```python
# In config.py
@dataclass
class MyDatabaseConfig(DatabaseConfig):
    my_param: float = 1.0
    another_param: int = 10

# Add to _DB_CONFIG_BY_TYPE
_DB_CONFIG_BY_TYPE["my_algo"] = MyDatabaseConfig
```

```yaml
# In config.yaml
search:
  type: my_algo
  database:
    my_param: 2.0
    another_param: 20
```

### Controller Primitives

| Method | Purpose |
|--------|---------|
| `await self._run_iteration(iteration)` | Full sample→prompt→generate→evaluate cycle |
| `self._process_iteration_result(result, iteration, cb)` | Add to DB, log, checkpoint |
| `await self._run_from_scratch_iteration(iteration)` | Generate first program when DB is empty |
| `self._build_prompt(parent, context, failed)` | Build LLM prompt |
| `self._parse_llm_response(response, parent, ...)` | Extract solution from LLM output |
| `self.database` | The current database instance |
| `self.shutdown_event.is_set()` | Graceful shutdown flag |
| `self.llms` / `self.guide_llms` | LLM pools |
| `self.evaluator` | Evaluator instance |
| `self.context_builder` | Context builder instance |

---

## 11. Checkpointing & Resume

### Checkpoint Structure

```
outputs/<search_type>/<problem>_<MMDD_HHMM>/
├── checkpoints/
│   └── checkpoint_<iteration>/
│       ├── programs/
│       │   └── <program_id>.json    # Program data + metrics
│       ├── best_program.py          # Best solution code
│       ├── best_program_info.json   # Best program metadata
│       ├── metadata.json            # Base checkpoint metadata
│       └── adaevolve_metadata.json  # Algorithm-specific state (if AdaEvolve)
│       └── gepa_metadata.json       # Algorithm-specific state (if GEPA)
├── best/
│   ├── best_program.py
│   └── best_program_info.json
└── logs/
    └── <search_type>.log
```

### Resume

```bash
uv run skydiscover-run init.py eval.py \
  --checkpoint outputs/adaevolve/problem_0305/checkpoints/checkpoint_50
```

The database's `load()` restores:
- All programs
- Best program tracking
- Last iteration number
- Algorithm-specific state (islands, archives, adaptive state, paradigm tracker, etc.)

**Important**: Ablation flags (e.g., `use_adaptive_search`, `use_ucb_selection`) are NOT restored from checkpoint — current config takes priority. This allows running ablation experiments from existing checkpoints.

---

## 12. Live Monitor & Human Feedback

### Monitor Dashboard

```yaml
monitor:
  enabled: true
  port: 8765
  host: 127.0.0.1
  max_solution_length: 10000
  summary_model: gpt-5-mini
  summary_interval: 0  # 0 = manual only
```

Provides a web UI at `http://localhost:8765/` with:
- Real-time score progression chart
- Current best solution display
- Iteration details
- AI-generated summaries of top solutions

### Human Feedback

```yaml
human_feedback_enabled: true
human_feedback_file: outputs/feedback.md
human_feedback_mode: append  # or "replace"
```

Write guidance to the feedback file while the run is active:
```markdown
Focus on reducing memory usage.
Try using numpy vectorized operations instead of loops.
```

Feedback is injected into the next iteration's prompt.

---

## 13. Key Formulas & Algorithms Reference

### AdaEvolve: Search Intensity

```
intensity = I_min + (I_max - I_min) / (1 + √(G + ε))

where:
  G_t = ρ · G_{t-1} + (1 - ρ) · δ²
  δ = min(raw_delta / (|best_score| + ε), 1.0)
  raw_delta = fitness - best_score  (only when fitness > best_score)
```

### AdaEvolve: UCB Island Selection

```
UCB_score(i) = reward_avg(i) + C · √(ln(N) / visits(i))

where:
  reward_avg(i) = decayed_rewards(i) / decayed_visits(i)
  decayed_rewards_t = ρ · decayed_rewards_{t-1} + global_normalized_delta
  decayed_visits_t = ρ · decayed_visits_{t-1} + 1
  C = √2 ≈ 1.41  (exploration constant)
  N = total iterations
```

### UnifiedArchive: Elite Score

```
elite_score = fitness_weight · fitness_percentile
            + novelty_weight · novelty_percentile
            [+ pareto_weight · pareto_percentile]   (if objectives configured)

where:
  fitness_percentile = 1 - (fitness_rank / (n-1))
  novelty = avg(k-NN distances using DiversityStrategy)
  novelty_percentile = count(n < novelty for all n in archive) / len(archive)
```

### EvoX: Search Algorithm Scoring

```
combined_score = improvement · log_weight / √horizon

where:
  improvement = running_best - start_score
  log_weight = 1 + ln(1 + max(0, start_score))
  horizon = number of iterations the strategy ran
```

### AdaEvolve: Sampling Mode

```
rand = random()
if rand < intensity:                          → exploration
elif rand < intensity + (1-intensity) · 0.7:  → exploitation
else:                                         → balanced
```

### GEPA: Acceptance Gate

```
accepted = child_score > parent_score  (strict inequality)
```

---

## 14. Configuration Reference

### Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iterations` | int | 100 | Maximum iterations |
| `checkpoint_interval` | int | 10 | Checkpoint every N iterations |
| `log_level` | str | "INFO" | Logging level |
| `language` | str | None | "python", "text", "image", "cpp" (auto-detected) |
| `diff_based_generation` | bool | True | Use SEARCH/REPLACE diffs |
| `max_solution_length` | int | 60000 | Max chars for generated solution |
| `max_parallel_iterations` | int | 1 | Concurrent iterations (>1 for parallelism) |

### AdaEvolveDatabaseConfig

| Field | Default | Description |
|-------|---------|-------------|
| `num_islands` | 2 | Number of parallel subpopulations |
| `population_size` | 20 | Max programs per island |
| `decay` | 0.9 | EMA decay factor (ρ) |
| `intensity_min` | 0.15 | Min search intensity (exploitation floor) |
| `intensity_max` | 0.5 | Max search intensity (exploration ceiling) |
| `migration_interval` | 15 | Iterations between migrations |
| `migration_count` | 5 | Programs migrated per cycle |
| `use_unified_archive` | True | Use quality-diversity archives |
| `diversity_strategy` | "code" | "code", "text", "metric", "hybrid" |
| `fitness_weight` | 1.0 | Weight for fitness in elite score |
| `novelty_weight` | 0.0 | Weight for novelty in elite score |
| `k_neighbors` | 5 | k for k-NN novelty |
| `use_paradigm_breakthrough` | True | Enable paradigm breakthroughs |
| `paradigm_window_size` | 10 | Window for improvement rate |
| `paradigm_improvement_threshold` | 0.12 | Rate below which triggers paradigm |
| `paradigm_max_uses` | 2 | Uses per paradigm before rotation |
| `use_dynamic_islands` | True | Enable dynamic island spawning |
| `max_islands` | 5 | Maximum islands |
| `use_adaptive_search` | True | Ablation: disable G-based intensity |
| `use_ucb_selection` | True | Ablation: disable UCB (use round-robin) |
| `use_migration` | True | Ablation: disable migration |
| `fixed_intensity` | 0.4 | Intensity when `use_adaptive_search=false` |
| `higher_is_better` | {} | Per-metric direction: `{"error": false}` |
| `fitness_key` | None | Primary metric key (auto-detected if None) |
| `pareto_objectives` | [] | Multi-objective keys for NSGA-II |
| `pareto_objectives_weight` | 0.0 | Weight in elite score |

### EvoXDatabaseConfig

| Field | Default | Description |
|-------|---------|-------------|
| `database_file_path` | Auto | Path to initial search strategy Python file |
| `evaluation_file` | Auto | Path to search strategy evaluator |
| `config_path` | Auto | Path to search-side YAML config |
| `auto_generate_variation_operators` | True | LLM-generate diverge/refine labels |

### GEPANativeDatabaseConfig

| Field | Default | Description |
|-------|---------|-------------|
| `population_size` | 40 | Elite pool size |
| `candidate_selection_strategy` | "epsilon_greedy" | "epsilon_greedy", "best", "pareto" |
| `epsilon` | 0.1 | Exploration probability |
| `acceptance_gating` | True | Reject children ≤ parent |
| `use_merge` | True | Enable LLM-mediated merge |
| `merge_after_stagnation` | 15 | Iterations before reactive merge |
| `max_merge_attempts` | 10 | Total merge budget |
| `max_recent_failures` | 5 | Rejected programs shown in prompt |
| `max_rejection_history` | 20 | Bounded deque size |

---

## 15. Troubleshooting & Anti-Patterns

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `combined_score` always 0 | Evaluator doesn't return `combined_score` | Add `combined_score` key to evaluator return dict |
| No improvement after many iterations | System prompt too vague | Add specific problem description to `prompt.system_message` |
| Solutions getting longer each iteration | No length control | Set `max_solution_length` or add length penalty to evaluator |
| All programs identical | `num_context_programs=0` or no diversity | Increase context programs, use `diversity_strategy: code` |
| AdaEvolve stuck on one island | UCB exploration too low | Increase `intensity_max` or check if `use_ucb_selection: true` |
| EvoX crashes after strategy switch | Generated database has bugs | Check logs for validation errors; system auto-restores fallback |
| GEPA rejects everything | Acceptance gate too strict | Lower initial program quality bar or reduce `epsilon` |

### Evaluator Best Practices

1. **Always return `combined_score`** — this is the primary optimization target
2. **Return multiple metrics** — enables Pareto optimization and richer prompt context
3. **Include error information** — `"error": "message"` helps the LLM fix issues
4. **Use cascade evaluation** — fast stage1 filters obvious failures cheaply
5. **Return artifacts** — `"feedback": "analysis..."` enables paradigm-informed generation

### Performance Tuning

| Parameter | Effect of Increasing |
|-----------|---------------------|
| `max_parallel_iterations` | More concurrent LLM calls, faster wall-clock time |
| `num_islands` (AdaEvolve) | More diversity, but each island gets less compute |
| `population_size` | More history retained, slower sampling |
| `migration_interval` | Less cross-pollination, more island independence |
| `paradigm_window_size` | Slower paradigm trigger, more data for decision |
| `num_context_programs` | Richer prompt context, higher token cost |
````

---

This document covers the complete SkyDiscover architecture at a depth sufficient for understanding, using, and extending the system. The AdaEvolve and EvoX sections provide implementation-level detail including exact formulas, data flow, configuration, and design rationale. The extension guide provides copy-pasteable patterns for both database-only and database+controller customization.