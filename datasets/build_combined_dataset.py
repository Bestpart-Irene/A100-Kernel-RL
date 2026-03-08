"""Build a unified KernelForge dataset from doubleGraph manifest + CUDA-Agent Ops-6K.

Output schema (one JSON object per line):
{
    "prompt": str,
    "ops": list[str],
    "difficulty": int,
    "data_source": "doublegraph_a100" | "ops_6k",
    "task_code": str | None,
    "evaluation_backend": "wcc" | "ops6k" | "unsupported",
    "support_reason": str,
    "topology": str | None,
    "graph_properties": dict | None,
    "kernel_id": str | None,
    "expert_code": str | None,
    "compile_flags": list[str] | None,
}
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.curriculum import CurriculumManager
from training.task_support import infer_evaluation_backend, parse_ops, support_reason

DATASETS_DIR = Path(__file__).resolve().parent
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from extract_doublegraph_a100 import ALGO_DISPLAY_NAMES, CATEGORY_META

DEFAULT_DG_MANIFEST = ROOT / "docs" / "research" / "doublegraph" / "doublegraph_a100_manifest.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "datasets" / "combined_kernelforge.jsonl"
DEFAULT_LOCAL_OPS_PATH = ROOT / "archive" / "datasets_legacy" / "curated_200.jsonl"
CUDA_AGENT_DATASET_ID = "BytedTsinghua-SIA/CUDA-Agent-Ops-6K"


def _get_cuda_agent_helpers():
    """Import CUDA-Agent prompt helpers lazily.

    This avoids importing Hugging Face datasets unless Ops-6K loading is requested.
    """
    from training.cuda_agent_integration import _build_cuda_prompt, _parse_ops

    return _build_cuda_prompt, _parse_ops


CATEGORY_TO_GRAPH_PROPS: dict[str, dict[str, Any]] = {
    "traversal": {
        "type": "power-law",
        "num_vertices": 10000,
        "num_edges": 62000,
        "avg_degree": 12.4,
        "max_degree": 4891,
        "density": 0.0012,
        "diameter": 8,
        "degree_distribution": "heavy-tail",
    },
    "components": {
        "type": "sparse-islands",
        "num_vertices": 1589,
        "num_edges": 2742,
        "avg_degree": 3.45,
        "max_degree": 34,
        "density": 0.0022,
        "diameter": 17,
        "degree_distribution": "sparse with disconnected islands",
    },
    "link_analysis": {
        "type": "dense-regular",
        "num_vertices": 1005,
        "num_edges": 25571,
        "avg_degree": 25.4,
        "max_degree": 71,
        "density": 0.0253,
        "diameter": 5,
        "degree_distribution": "near-uniform",
    },
    "community": {
        "type": "dense-community",
        "num_vertices": 4039,
        "num_edges": 88234,
        "avg_degree": 43.7,
        "max_degree": 1045,
        "density": 0.0108,
        "diameter": 8,
        "degree_distribution": "community-clustered",
    },
    "centrality": {
        "type": "dense-regular",
        "num_vertices": 1005,
        "num_edges": 25571,
        "avg_degree": 25.4,
        "max_degree": 71,
        "density": 0.0253,
        "diameter": 5,
        "degree_distribution": "near-uniform",
    },
    "link_prediction": {
        "type": "bipartite",
        "num_vertices": 3500,
        "num_edges": 30000,
        "avg_degree": 17.1,
        "max_degree": 450,
        "density": 0.0049,
        "diameter": 6,
        "degree_distribution": "bimodal",
    },
}


def _load_hf_ops6k_split():
    """Load HF dataset without shadowing local `datasets/` package."""
    cwd = os.getcwd()
    orig_sys_path = list(sys.path)
    try:
        shadow_paths = {"", ".", cwd, str(ROOT), str(DATASETS_DIR)}
        sys.path = [p for p in sys.path if p not in shadow_paths]
        import datasets as hf_datasets  # noqa: WPS433

        return hf_datasets.load_dataset(CUDA_AGENT_DATASET_ID, split="train")
    finally:
        sys.path = orig_sys_path


def _load_local_ops_split(path: str | Path = DEFAULT_LOCAL_OPS_PATH) -> list[dict[str, Any]]:
    """Fallback local prompt corpus for offline development."""
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line in f:
            raw = line.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _difficulty_from_variant(variant: str) -> int:
    if variant == "base":
        return 2
    if variant in {"seg", "mask"}:
        return 3
    return 4


def _difficulty_from_ops(ops: list[str]) -> int:
    if len(ops) <= 1:
        return 1
    if len(ops) == 2:
        return 2
    return 3


def _graph_properties_for_category(category: str, feature_hints: list[str]) -> dict[str, Any]:
    base = dict(CATEGORY_TO_GRAPH_PROPS.get(category, {}))
    meta = CATEGORY_META.get(category, {})
    base.setdefault("type", meta.get("topology", "unknown"))
    base.setdefault("optimization_hints", [])

    hints = list(base["optimization_hints"])
    if meta.get("pattern"):
        hints.append(str(meta["pattern"]))
    if feature_hints:
        hints.append("Kernel feature markers: " + ", ".join(sorted(feature_hints)))
    base["optimization_hints"] = hints

    if meta.get("datasets"):
        base["example_datasets"] = list(meta["datasets"])

    return base


def _doublegraph_prompt(record: dict[str, Any]) -> str:
    kernel_id = str(record.get("kernel_id", ""))
    category = str(record.get("category", "unknown"))
    algo_name = str(record.get("algorithm_name", "unknown"))
    display_name = ALGO_DISPLAY_NAMES.get(algo_name, algo_name.replace("_", " ").title())
    variant = str(record.get("variant", "base"))
    flags = record.get("flags") or []

    meta = CATEGORY_META.get(category, {})
    topo_desc = meta.get("topology_desc", "Graph topology metadata provided in context.")
    pattern = meta.get("pattern", "Optimize for A100 memory hierarchy and warp behavior.")

    feature_tags = []
    if record.get("uses_cooperative_groups"):
        feature_tags.append("cooperative_groups")
    if record.get("uses_atomic_ops"):
        feature_tags.append("atomic_ops")
    if record.get("uses_cache_pool"):
        feature_tags.append("cache_pool")
    if record.get("uses_cusparse"):
        feature_tags.append("cusparse")

    lines = [
        (
            "Write a high-performance CUDA kernel for NVIDIA A100 (sm_80) implementing "
            f"{display_name}."
        ),
        f"Kernel ID: {kernel_id}",
        f"Category: {category} | Variant: {variant}",
        f"Topology context: {topo_desc}",
        f"Optimization pattern: {pattern}",
    ]

    if flags:
        lines.append(f"Recommended compile flags: {' '.join(str(x) for x in flags)}")
    if feature_tags:
        lines.append("Observed expert-kernel features: " + ", ".join(feature_tags))

    lines.append("Return CUDA/C++ code only.")
    return "\n".join(lines)


def load_doublegraph_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Convert doubleGraph manifest records to unified problem dicts."""
    path = Path(manifest_path)
    records: list[dict[str, Any]] = []

    with path.open() as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            item = json.loads(raw)

            feature_hints = []
            if item.get("uses_cooperative_groups"):
                feature_hints.append("cooperative_groups")
            if item.get("uses_atomic_ops"):
                feature_hints.append("atomic_ops")
            if item.get("uses_cusparse"):
                feature_hints.append("cusparse")

            category = str(item.get("category", ""))
            graph_props = _graph_properties_for_category(category, feature_hints)
            records.append(
                {
                    "prompt": _doublegraph_prompt(item),
                    "ops": [str(item.get("algorithm_name", "unknown"))],
                    "difficulty": _difficulty_from_variant(str(item.get("variant", "base"))),
                    "data_source": "doublegraph_a100",
                    "task_code": None,
                    "evaluation_backend": infer_evaluation_backend(
                        {
                            "prompt": _doublegraph_prompt(item),
                            "ops": [str(item.get("algorithm_name", "unknown"))],
                            "kernel_id": item.get("kernel_id"),
                        }
                    ),
                    "support_reason": support_reason(
                        {
                            "prompt": _doublegraph_prompt(item),
                            "ops": [str(item.get("algorithm_name", "unknown"))],
                            "kernel_id": item.get("kernel_id"),
                        }
                    ),
                    "topology": graph_props.get("type"),
                    "graph_properties": graph_props,
                    "kernel_id": item.get("kernel_id"),
                    "expert_code": None,
                    "compile_flags": list(item.get("flags", [])),
                }
            )

    return records


def load_ops6k(max_samples: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
    """Convert Ops-6K rows to unified problem dicts."""
    if max_samples is not None and int(max_samples) <= 0:
        return []

    _build_cuda_prompt, _parse_ops = _get_cuda_agent_helpers()
    data_source = "ops_6k"
    try:
        ds = _load_hf_ops6k_split()
    except Exception as exc:
        print(f"Falling back to local curated ops dataset: {exc}")
        ds = _load_local_ops_split()
        data_source = "ops_local_fallback"

    rows: list[dict[str, Any]] = []
    for item in ds:
        prompt = _build_cuda_prompt(item)
        if not prompt:
            continue
        ops = _parse_ops(item.get("ops"))
        row = {
            "prompt": prompt,
            "ops": parse_ops(ops),
            "difficulty": _difficulty_from_ops(ops),
            "data_source": data_source,
            "task_code": str(item.get("code", "")),
            "topology": None,
            "graph_properties": None,
            "kernel_id": None,
            "expert_code": None,
            "compile_flags": None,
        }
        row["evaluation_backend"] = infer_evaluation_backend(row)
        row["support_reason"] = support_reason(row)
        rows.append(
            row
        )

    if max_samples is not None:
        max_samples = max(0, int(max_samples))
        rng = random.Random(seed)
        supported_rows = [row for row in rows if row.get("evaluation_backend") == "ops6k"]
        unsupported_rows = [row for row in rows if row.get("evaluation_backend") != "ops6k"]
        rng.shuffle(supported_rows)
        rng.shuffle(unsupported_rows)
        rows = supported_rows[:max_samples]
        if len(rows) < max_samples:
            rows.extend(unsupported_rows[: max_samples - len(rows)])

    return rows


def build_combined_dataset(
    dg_path: str | Path = DEFAULT_DG_MANIFEST,
    ops6k_max: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build and shuffle combined dataset rows."""
    dg_rows = load_doublegraph_manifest(dg_path)
    ops_rows = load_ops6k(max_samples=ops6k_max, seed=seed)

    merged = dg_rows + ops_rows
    rng = random.Random(seed)
    rng.shuffle(merged)
    return merged


def inject_into_curriculum(cm: CurriculumManager, dataset: list[dict[str, Any]]) -> dict[str, int]:
    """Route problems to curriculum phases by difficulty and add them in bulk."""
    phase_map = {
        1: "single_ops",
        2: "fusion_2op",
        3: "arch_specific",
        4: "advanced",
    }
    grouped: dict[str, list[dict[str, Any]]] = {
        "single_ops": [],
        "fusion_2op": [],
        "arch_specific": [],
        "advanced": [],
    }

    for row in dataset:
        difficulty = int(row.get("difficulty", 1))
        phase_name = phase_map.get(difficulty, "single_ops")
        grouped[phase_name].append(row)

    for phase_name, problems in grouped.items():
        if problems:
            cm.add_problems(phase_name, problems)

    return {name: len(items) for name, items in grouped.items()}


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build combined KernelForge dataset")
    parser.add_argument(
        "--dg-manifest",
        type=str,
        default=str(DEFAULT_DG_MANIFEST),
        help="Path to doubleGraph manifest JSONL",
    )
    parser.add_argument(
        "--ops6k-max",
        type=int,
        default=None,
        help="Optional cap for Ops-6K examples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    rows = build_combined_dataset(
        dg_path=args.dg_manifest,
        ops6k_max=args.ops6k_max,
        seed=args.seed,
    )
    write_jsonl(rows, args.output)

    by_source: dict[str, int] = {}
    by_difficulty: dict[int, int] = {}
    for row in rows:
        src = str(row.get("data_source", "unknown"))
        by_source[src] = by_source.get(src, 0) + 1
        diff = int(row.get("difficulty", 1))
        by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

    print(f"Wrote {len(rows)} rows -> {args.output}")
    print(f"By source: {by_source}")
    print(f"By difficulty: {dict(sorted(by_difficulty.items()))}")


if __name__ == "__main__":
    main()
