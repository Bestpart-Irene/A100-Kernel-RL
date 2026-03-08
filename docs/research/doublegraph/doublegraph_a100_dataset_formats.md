# DoubleGraph A100 Dataset Formats for KernelForge RL

## Key Files

- `docs/research/doublegraph/doublegraph_a100_manifest.jsonl` — canonical manifest (192 kernels)
- `datasets/combined_kernelforge.jsonl` — unified RL training dataset
- `datasets/doublegraph_sft.jsonl` — SFT format for Stage 2

Legacy files (`RL_DATASET_INDEX.md`, `SKILLS.md`, `doublegraph_a100_manifest.csv`, `doublegraph_a100_dataset_full_details.md`) archived to `archive/research_legacy/`.

## 1. Scope and Ground Truth

This document defines the canonical dataset formats for importing DoubleGraph A100 CUDA kernels into another project (RL environment, AdaEvolve islands, EvoX strategies, and evaluation pipeline).

Source root in this repo:
- `cpp/src/aai/impl/a100/`

Observed inventory in this repo snapshot:
- Total kernel files (`.cu`): `192`
- Total sidecar flag files (`.cu.flags`): `172`
- Kernels without sidecar flags: `20`

A100 categories and counts:

| category | cu_count | flags_count |
|---|---:|---:|
| centrality | 32 | 29 |
| community | 42 | 40 |
| components | 6 | 6 |
| cores | 8 | 8 |
| link_analysis | 32 | 24 |
| link_prediction | 48 | 44 |
| traversal | 20 | 17 |
| tree | 4 | 4 |

Variant distribution by filename suffix:
- `base`: `70`
- `seg`: `70`
- `mask`: `26`
- `seg_mask`: `26`

## 2. Canonical Naming and IDs

### 2.1 File conventions

Each kernel file follows:
- `a100/<category>/<algorithm_variant>.cu`

Optional sidecar flags file:
- `a100/<category>/<algorithm_variant>.cu.flags`

### 2.2 Stable IDs

`kernel_id` is derived as:
- `a100/<category>/<stem>`
- where `stem` is filename without `.cu`

Examples:
- `cpp/src/aai/impl/a100/traversal/bfs.cu` -> `kernel_id = "a100/traversal/bfs"`
- `cpp/src/aai/impl/a100/community/triangle_count_seg_mask.cu` -> `kernel_id = "a100/community/triangle_count_seg_mask"`

### 2.3 Variant parsing rule

Suffix precedence (must be applied in this order):
1. `_seg_mask` -> `variant = "seg_mask"`
2. `_seg` -> `variant = "seg"`
3. `_mask` -> `variant = "mask"`
4. otherwise -> `variant = "base"`

`algorithm_name` is stem with the detected suffix removed.

## 3. Dataset Artifacts

Recommended export as newline-delimited JSON (`.jsonl`) with one record per line:
- `kernel_source_records.jsonl`
- `kernel_flags_records.jsonl`
- `kernel_task_records.jsonl`
- `evolution_seed_records.jsonl`
- `curriculum_items.jsonl`

RL runtime payloads can be generated per-episode and stored as:
- `rl_episode_observations.jsonl`

## 4. Schemas

## 4.1 KernelSourceRecord (raw extraction)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "KernelSourceRecord",
  "type": "object",
  "required": [
    "kernel_id",
    "gpu_target",
    "category",
    "algorithm_name",
    "variant",
    "source_path",
    "source_code",
    "source_sha256",
    "line_count",
    "includes",
    "has_cache_struct",
    "uses_cooperative_groups",
    "uses_atomic_ops",
    "uses_cusparse"
  ],
  "properties": {
    "kernel_id": {"type": "string", "pattern": "^a100/[a-z_]+/[a-z0-9_]+$"},
    "gpu_target": {"type": "string", "const": "a100"},
    "category": {
      "type": "string",
      "enum": [
        "centrality",
        "community",
        "components",
        "cores",
        "link_analysis",
        "link_prediction",
        "traversal",
        "tree"
      ]
    },
    "algorithm_name": {"type": "string", "minLength": 1},
    "variant": {"type": "string", "enum": ["base", "seg", "mask", "seg_mask"]},
    "source_path": {"type": "string"},
    "source_code": {"type": "string", "minLength": 1},
    "source_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
    "line_count": {"type": "integer", "minimum": 1},
    "includes": {"type": "array", "items": {"type": "string"}},
    "has_cache_struct": {"type": "boolean"},
    "uses_cooperative_groups": {"type": "boolean"},
    "uses_atomic_ops": {"type": "boolean"},
    "uses_cusparse": {"type": "boolean"}
  },
  "additionalProperties": false
}
```

Feature extraction guidance:
- `includes`: collect from `#include <...>` and `#include "..."` lines.
- `has_cache_struct`: `true` if code contains `struct Cache` and `Cacheable` usage.
- `uses_cooperative_groups`: `true` if code includes `cooperative_groups` or `cg::` references.
- `uses_atomic_ops`: `true` if code contains `atomic` primitives (for example `atomicAdd`, `atomicCAS`, `atomicOr`).
- `uses_cusparse`: `true` if code contains `cusparse` tokens.

## 4.2 KernelFlagsRecord (sidecar mapping)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "KernelFlagsRecord",
  "type": "object",
  "required": [
    "kernel_id",
    "flags_path",
    "flags",
    "missing_flags_policy",
    "requires_rdc"
  ],
  "properties": {
    "kernel_id": {"type": "string", "pattern": "^a100/[a-z_]+/[a-z0-9_]+$"},
    "flags_path": {"type": ["string", "null"]},
    "flags": {"type": "array", "items": {"type": "string"}},
    "missing_flags_policy": {
      "type": "string",
      "enum": ["inherit_default", "none"]
    },
    "requires_rdc": {"type": "boolean"}
  },
  "additionalProperties": false
}
```

Rules:
- If `.cu.flags` exists, `flags_path` is non-null and `flags` is each non-empty line in that file.
- If `.cu.flags` is missing, `flags_path = null` and `flags = []`.
- Default policy:
  - `missing_flags_policy = "inherit_default"`
  - default compiler baseline must inject `--use_fast_math` unless downstream policy overrides.
- `requires_rdc = true` if either:
  - `flags` contains `--rdc=true`, or
  - `KernelSourceRecord.uses_cooperative_groups = true`

Observed sidecar flag tokens in this repo:
- `--use_fast_math`
- `--rdc=true`
- `--extra-device-vectorization`
- `--maxrregcount=40`
- `--maxrregcount=48`
- `--maxrregcount=56`
- `--maxrregcount=64`
- `--expt-relaxed-constexpr`
- `-Xptxas`
- `-dlcm=ca`

## 4.3 KernelTaskRecord (RL-facing normalized task)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "KernelTaskRecord",
  "type": "object",
  "required": [
    "task_id",
    "kernel_id",
    "category",
    "topology_tags",
    "compile_profile",
    "header_context",
    "optimization_targets",
    "constraints"
  ],
  "properties": {
    "task_id": {"type": "string"},
    "kernel_id": {"type": "string", "pattern": "^a100/[a-z_]+/[a-z0-9_]+$"},
    "category": {
      "type": "string",
      "enum": [
        "centrality",
        "community",
        "components",
        "cores",
        "link_analysis",
        "link_prediction",
        "traversal",
        "tree"
      ]
    },
    "topology_tags": {"type": "array", "items": {"type": "string"}},
    "compile_profile": {
      "type": "object",
      "required": ["arch", "opt_level", "flags"],
      "properties": {
        "arch": {"type": "string", "const": "sm_80"},
        "opt_level": {"type": "string", "enum": ["-O2", "-O3"]},
        "flags": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": false
    },
    "header_context": {
      "type": "object",
      "required": ["compact_graph", "cache_pool", "types"],
      "properties": {
        "compact_graph": {"type": "string"},
        "cache_pool": {"type": "string"},
        "types": {"type": "string"}
      },
      "additionalProperties": false
    },
    "optimization_targets": {"type": "array", "items": {"type": "string"}},
    "constraints": {"type": "array", "items": {"type": "string"}}
  },
  "additionalProperties": false
}
```

Normalization defaults:
- `task_id = "task/" + kernel_id`
- `compile_profile.arch = "sm_80"`
- `compile_profile.opt_level = "-O3"`

## 4.4 RLEpisodeObservationRecord (runtime prompt payload)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RLEpisodeObservationRecord",
  "type": "object",
  "required": [
    "episode_id",
    "step_id",
    "task_id",
    "problem_prompt",
    "topology_context",
    "kernel_context",
    "reward_context"
  ],
  "properties": {
    "episode_id": {"type": "string"},
    "step_id": {"type": "integer", "minimum": 0},
    "task_id": {"type": "string"},
    "problem_prompt": {"type": "string"},
    "topology_context": {
      "type": "object",
      "required": ["topology_type", "graph_properties"],
      "properties": {
        "topology_type": {"type": "string"},
        "graph_properties": {"type": "object"}
      },
      "additionalProperties": true
    },
    "kernel_context": {
      "type": "object",
      "required": ["kernel_id", "source_code", "header_context"],
      "properties": {
        "kernel_id": {"type": "string"},
        "source_code": {"type": "string"},
        "header_context": {"type": "object"}
      },
      "additionalProperties": true
    },
    "reward_context": {
      "type": "object",
      "required": ["metrics"],
      "properties": {
        "metrics": {"type": "object"},
        "penalties": {"type": "object"}
      },
      "additionalProperties": true
    }
  },
  "additionalProperties": false
}
```

## 4.5 EvolutionSeedRecord (AdaEvolve/EvoX seed payload)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "EvolutionSeedRecord",
  "type": "object",
  "required": [
    "seed_id",
    "kernel_id",
    "island_category",
    "genome_payload",
    "strategy_profile"
  ],
  "properties": {
    "seed_id": {"type": "string"},
    "kernel_id": {"type": "string", "pattern": "^a100/[a-z_]+/[a-z0-9_]+$"},
    "island_category": {
      "type": "string",
      "enum": [
        "centrality",
        "community",
        "components",
        "cores",
        "link_analysis",
        "link_prediction",
        "traversal",
        "tree"
      ]
    },
    "genome_payload": {
      "oneOf": [
        {"type": "string"},
        {"type": "object"}
      ]
    },
    "strategy_profile": {"type": "object"},
    "fitness_baseline": {"type": "object"}
  },
  "additionalProperties": false
}
```

## 4.6 CurriculumItemRecord

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CurriculumItemRecord",
  "type": "object",
  "required": [
    "curriculum_id",
    "level",
    "task_ids",
    "gating_metrics",
    "promotion_rule"
  ],
  "properties": {
    "curriculum_id": {"type": "string"},
    "level": {"type": "integer", "minimum": 1},
    "task_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    "gating_metrics": {"type": "object"},
    "promotion_rule": {"type": "object"}
  },
  "additionalProperties": false
}
```

## 5. Validation Rules

A data export is valid only if all checks pass.

1. Inventory consistency
- Exactly one `KernelSourceRecord` per discovered `.cu` file.
- `KernelSourceRecord` count must equal discovered kernel file count (`192` for this snapshot).

2. Foreign key integrity
- Every `KernelFlagsRecord.kernel_id` must exist in `KernelSourceRecord`.
- Every `KernelTaskRecord.kernel_id` must exist in `KernelSourceRecord`.
- Every `EvolutionSeedRecord.kernel_id` must exist in `KernelSourceRecord`.

3. Category integrity
- `category` must match the directory under `a100/`.

4. Variant integrity
- `variant` must be derived only by the suffix precedence rule in section 2.3.

5. Sidecar handling integrity
- If sidecar exists and is non-empty, `flags` must preserve line order.
- If sidecar missing, `flags_path = null`, `flags = []`, and `missing_flags_policy` must be explicitly set.

6. Compile integrity
- A100 compile profile must include `arch = sm_80`.
- If `requires_rdc = true`, compile command must include `-rdc=true`.

7. Deterministic hashing
- `source_sha256` must be computed over exact file bytes used for `source_code`.

## 6. Reference Compile Command Template

Baseline template for candidate evaluation:

```bash
nvcc -arch=sm_80 -O3 -shared <flags...> -o <out.so> <kernel.cu>
```

Policy defaults:
- If `KernelFlagsRecord.flags` is non-empty, append as-is.
- If missing sidecar and policy is `inherit_default`, inject `--use_fast_math`.
- If `requires_rdc` is true, append `-rdc=true` (if not already present).

## 7. Example Records

## 7.1 Example with sidecar flags (`traversal/bfs`)

```json
{
  "kernel_id": "a100/traversal/bfs",
  "gpu_target": "a100",
  "category": "traversal",
  "algorithm_name": "bfs",
  "variant": "base",
  "source_path": "cpp/src/aai/impl/a100/traversal/bfs.cu",
  "source_code": "/* full bfs.cu content here */",
  "source_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "line_count": 459,
  "includes": [
    "cugraph/aai/algorithms.hpp",
    "cuda_runtime.h",
    "cstdint",
    "limits"
  ],
  "has_cache_struct": true,
  "uses_cooperative_groups": false,
  "uses_atomic_ops": true,
  "uses_cusparse": false
}
```

```json
{
  "kernel_id": "a100/traversal/bfs",
  "flags_path": "cpp/src/aai/impl/a100/traversal/bfs.cu.flags",
  "flags": ["--use_fast_math"],
  "missing_flags_policy": "inherit_default",
  "requires_rdc": false
}
```

## 7.2 Example without sidecar flags (`traversal/sssp_f64`)

```json
{
  "kernel_id": "a100/traversal/sssp_f64",
  "gpu_target": "a100",
  "category": "traversal",
  "algorithm_name": "sssp_f64",
  "variant": "base",
  "source_path": "cpp/src/aai/impl/a100/traversal/sssp_f64.cu",
  "source_code": "/* full sssp_f64.cu content here */",
  "source_sha256": "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd",
  "line_count": 210,
  "includes": ["cugraph/aai/algorithms.hpp", "cuda_runtime.h"],
  "has_cache_struct": true,
  "uses_cooperative_groups": false,
  "uses_atomic_ops": true,
  "uses_cusparse": false
}
```

```json
{
  "kernel_id": "a100/traversal/sssp_f64",
  "flags_path": null,
  "flags": [],
  "missing_flags_policy": "inherit_default",
  "requires_rdc": false
}
```

## 7.3 Example RL task + observation

```json
{
  "task_id": "task/a100/traversal/bfs",
  "kernel_id": "a100/traversal/bfs",
  "category": "traversal",
  "topology_tags": ["power_law", "sparse", "diameter_high"],
  "compile_profile": {
    "arch": "sm_80",
    "opt_level": "-O3",
    "flags": ["--use_fast_math"]
  },
  "header_context": {
    "compact_graph": "<stripped compact_graph.hpp excerpt>",
    "cache_pool": "<stripped cache_pool.hpp excerpt>",
    "types": "<stripped types.hpp excerpt>"
  },
  "optimization_targets": ["runtime_ms", "warp_efficiency", "occupancy"],
  "constraints": ["preserve_correctness", "no_api_changes", "sm80_only"]
}
```

```json
{
  "episode_id": "ep_000123",
  "step_id": 0,
  "task_id": "task/a100/traversal/bfs",
  "problem_prompt": "Optimize BFS kernel for a sparse power-law graph on A100.",
  "topology_context": {
    "topology_type": "power_law",
    "graph_properties": {
      "num_vertices": 1000000,
      "num_edges": 12000000,
      "density": 0.000012,
      "directed": true
    }
  },
  "kernel_context": {
    "kernel_id": "a100/traversal/bfs",
    "source_code": "/* bfs code */",
    "header_context": {
      "compact_graph": "...",
      "cache_pool": "...",
      "types": "..."
    }
  },
  "reward_context": {
    "metrics": {
      "runtime_ms": 2.9,
      "warp_efficiency": 0.78,
      "occupancy": 0.64
    },
    "penalties": {
      "compile_fail": 0,
      "wrong_answer": 0
    }
  }
}
```

## 7.4 Example Evo seed

```json
{
  "seed_id": "seed/a100/traversal/bfs",
  "kernel_id": "a100/traversal/bfs",
  "island_category": "traversal",
  "genome_payload": {
    "representation": "source_patch",
    "base_kernel_id": "a100/traversal/bfs"
  },
  "strategy_profile": {
    "mutation_family": "block_and_memory",
    "crossover_family": "intra_category"
  },
  "fitness_baseline": {
    "runtime_ms": 2.9,
    "correctness_passed": true
  }
}
```

## 8. Parser Acceptance Tests

1. Parse all `.cu` files under `cpp/src/aai/impl/a100` and assert record count equals discovery count.
2. Parse sidecar flags and ensure at least one known token appears (`--use_fast_math` expected in all sidecars in this snapshot).
3. Validate variant extraction against known suffix forms.
4. Validate missing sidecar behavior using one known missing file (`traversal/sssp_f64.cu`).
5. Validate schema conformance for all six record types.
6. Validate foreign keys (`kernel_id`) across all exported datasets.
7. Validate compile profile generation for `requires_rdc` and non-`requires_rdc` cases.

## 9. Known Missing Sidecar Files in This Snapshot

The following 20 A100 kernels do not currently have `.cu.flags` sidecars:

- `centrality/eigenvector_centrality_f64.cu`
- `centrality/katz_centrality_f64_mask.cu`
- `centrality/katz_centrality_f64_seg_mask.cu`
- `community/louvain_f64.cu`
- `community/louvain_f64_seg.cu`
- `link_analysis/hits_f64_mask.cu`
- `link_analysis/pagerank_f64_mask.cu`
- `link_analysis/pagerank_f64_p64.cu`
- `link_analysis/pagerank_f64_p64_mask.cu`
- `link_analysis/pagerank_f64_p64_seg.cu`
- `link_analysis/pagerank_f64_seg.cu`
- `link_analysis/pagerank_mask.cu`
- `link_analysis/pagerank_seg_mask.cu`
- `link_prediction/cosine_f64.cu`
- `link_prediction/jaccard_all_pairs_f64.cu`
- `link_prediction/overlap_all_pairs_f64.cu`
- `link_prediction/overlap_f64.cu`
- `traversal/sssp_f64.cu`
- `traversal/sssp_f64_mask.cu`
- `traversal/sssp_f64_seg_mask.cu`

## 10. Defaults and Assumptions

- Scope is A100 only.
- Dataset contract is authoritative for downstream parser implementation.
- Full kernel source is preserved (not only device functions).
- Header context should be injected from:
  - `cpp/include/cugraph/aai/compact_graph.hpp`
  - `cpp/include/cugraph/aai/cache_pool.hpp`
  - `cpp/include/cugraph/aai/types.hpp`
- RL prompt builders may strip comments from injected header snippets to reduce token load.
