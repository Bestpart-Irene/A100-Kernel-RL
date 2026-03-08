#!/usr/bin/env python3
"""Compare before/after benchmark results for KernelForge.

Usage:
    python scripts/compare_results.py results/before.json results/after.json
    python scripts/compare_results.py results/before.json results/after.json --output results/comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    """Load benchmark results JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(before: dict, after: dict) -> dict:
    """Compare two benchmark result sets and produce a comparison report."""
    bm = before.get("metrics", {})
    am = after.get("metrics", {})

    metric_keys = [
        "fast_0", "fast_1", "fast_1.05", "fast_compile",
        "geomean_vs_eager", "geomean_vs_compile",
        "correctness_rate", "compile_rate",
    ]

    deltas = {}
    for k in metric_keys:
        bv = bm.get(k, 0.0)
        av = am.get(k, 0.0)
        deltas[k] = {
            "before": bv,
            "after": av,
            "delta": av - bv,
            "relative": (av - bv) / bv if bv else None,
        }

    # Per-task comparison
    before_tasks = {r["task_id"]: r for r in before.get("results", [])}
    after_tasks = {r["task_id"]: r for r in after.get("results", [])}
    all_task_ids = sorted(set(before_tasks) | set(after_tasks))

    task_diffs = []
    for tid in all_task_ids:
        br = before_tasks.get(tid, {})
        ar = after_tasks.get(tid, {})
        task_diffs.append({
            "task_id": tid,
            "before_correct": br.get("correct"),
            "after_correct": ar.get("correct"),
            "before_speedup_eager": br.get("speedup_vs_eager"),
            "after_speedup_eager": ar.get("speedup_vs_eager"),
            "before_speedup_compile": br.get("speedup_vs_compile"),
            "after_speedup_compile": ar.get("speedup_vs_compile"),
            "improved": (
                (ar.get("correct") and not br.get("correct"))
                or (
                    ar.get("correct") and br.get("correct")
                    and (ar.get("speedup_vs_eager", 0) or 0) > (br.get("speedup_vs_eager", 0) or 0)
                )
            ),
            "regressed": (
                (br.get("correct") and not ar.get("correct"))
                or (
                    ar.get("correct") and br.get("correct")
                    and (ar.get("speedup_vs_eager", 0) or 0) < (br.get("speedup_vs_eager", 0) or 0)
                )
            ),
        })

    improved = sum(1 for t in task_diffs if t["improved"])
    regressed = sum(1 for t in task_diffs if t["regressed"])
    unchanged = len(task_diffs) - improved - regressed

    return {
        "before_model": before.get("model", "unknown"),
        "after_model": after.get("model", "unknown"),
        "before_timestamp": before.get("timestamp"),
        "after_timestamp": after.get("timestamp"),
        "metrics": deltas,
        "summary": {
            "total_tasks": len(all_task_ids),
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        },
        "task_diffs": task_diffs,
    }


def print_comparison(comp: dict) -> None:
    """Print a human-readable comparison report."""
    print("=" * 70)
    print("KERNELFORGE BEFORE/AFTER COMPARISON")
    print("=" * 70)
    print(f"Before: {comp['before_model']} ({comp.get('before_timestamp', '?')})")
    print(f"After:  {comp['after_model']} ({comp.get('after_timestamp', '?')})")
    print()

    print("METRICS:")
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for k, v in comp["metrics"].items():
        bv = v["before"]
        av = v["after"]
        dv = v["delta"]
        sign = "+" if dv > 0 else ""
        if isinstance(bv, float):
            print(f"  {k:<25} {bv:>10.3f} {av:>10.3f} {sign}{dv:>9.3f}")
        else:
            print(f"  {k:<25} {str(bv):>10} {str(av):>10} {sign}{dv:>9}")

    print()
    s = comp["summary"]
    print(f"TASKS: {s['total_tasks']} total — "
          f"{s['improved']} improved, {s['regressed']} regressed, "
          f"{s['unchanged']} unchanged")

    # Show per-task details for improved/regressed
    print()
    for td in comp["task_diffs"]:
        if td["improved"]:
            marker = "[+]"
        elif td["regressed"]:
            marker = "[-]"
        else:
            marker = "[ ]"

        bc = "Y" if td["before_correct"] else "N" if td["before_correct"] is not None else "?"
        ac = "Y" if td["after_correct"] else "N" if td["after_correct"] is not None else "?"
        bs = td["before_speedup_eager"]
        as_ = td["after_speedup_eager"]
        bs_str = f"{bs:.2f}x" if bs is not None else "  n/a"
        as_str = f"{as_:.2f}x" if as_ is not None else "  n/a"

        print(f"  {marker} {td['task_id']:<30} correct: {bc}->{ac}  "
              f"speedup: {bs_str}->{as_str}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare KernelForge benchmark results")
    parser.add_argument("before", type=str, help="Path to before-training results JSON")
    parser.add_argument("after", type=str, help="Path to after-training results JSON")
    parser.add_argument("--output", type=str, default=None, help="Save comparison JSON")
    args = parser.parse_args()

    before = load_results(args.before)
    after = load_results(args.after)
    comp = compare(before, after)

    print_comparison(comp)

    if args.output:
        out_dir = Path(args.output).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2, ensure_ascii=False)
        print(f"\nComparison saved to {args.output}")


if __name__ == "__main__":
    main()
