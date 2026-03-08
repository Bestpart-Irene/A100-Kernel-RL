"""Dataset integrity verification — SHA-256 hashing and structural validation.

Run before training to catch data corruption or unexpected schema changes.
"""
from __future__ import annotations

import hashlib
import json
import os


def hash_file(path: str) -> str | None:
    """Compute SHA-256 hash of a file. Returns None if file doesn't exist."""
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_jsonl(path: str, required_keys: list[str] | None = None) -> dict:
    """Verify each line is valid JSON with optional required keys.

    Returns:
        Dict with line_count, errors list, and valid bool.
    """
    if not os.path.exists(path):
        return {"path": path, "exists": False, "line_count": 0, "errors": ["File not found"], "valid": False}

    errors = []
    line_count = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if required_keys:
                    missing = [k for k in required_keys if k not in obj]
                    if missing:
                        errors.append(f"Line {i}: missing keys {missing}")
                line_count += 1
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON — {e}")

    return {
        "path": path,
        "exists": True,
        "line_count": line_count,
        "errors": errors[:20],
        "total_errors": len(errors),
        "valid": len(errors) == 0,
    }


def verify_combined_dataset(path: str = "datasets/combined_kernelforge.jsonl") -> dict:
    """Validate the combined training dataset."""
    result = verify_jsonl(path, required_keys=["prompt", "ops", "difficulty", "data_source"])
    if result["valid"] and result["line_count"] > 0:
        difficulties = {}
        sources = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                d = row.get("difficulty", 0)
                difficulties[d] = difficulties.get(d, 0) + 1
                s = row.get("data_source", "unknown")
                sources[s] = sources.get(s, 0) + 1
        result["difficulty_distribution"] = dict(sorted(difficulties.items()))
        result["source_distribution"] = dict(sorted(sources.items()))
    result["sha256"] = hash_file(path)
    return result


def verify_manifest(path: str = "docs/research/doublegraph/doublegraph_a100_manifest.jsonl") -> dict:
    """Validate the doubleGraph A100 manifest."""
    result = verify_jsonl(path, required_keys=["kernel_id", "category", "algorithm_name", "variant"])
    result["sha256"] = hash_file(path)
    return result


def verify_sft_dataset(path: str = "datasets/doublegraph_sft.jsonl") -> dict:
    """Validate the SFT dataset."""
    result = verify_jsonl(path, required_keys=["messages"])
    result["sha256"] = hash_file(path)
    return result


def main():
    """Run all integrity checks and print report."""
    print("Dataset Integrity Report")
    print("=" * 50)

    for name, check_fn in [
        ("Combined Dataset", verify_combined_dataset),
        ("doubleGraph Manifest", verify_manifest),
        ("SFT Dataset", verify_sft_dataset),
    ]:
        result = check_fn()
        status = "PASS" if result.get("valid") else "FAIL"
        print(f"\n{name}: {status}")
        print(f"  Path: {result.get('path', 'N/A')}")
        print(f"  Lines: {result.get('line_count', 0)}")
        print(f"  SHA-256: {result.get('sha256', 'N/A')}")
        if result.get("errors"):
            for e in result["errors"][:5]:
                print(f"  ERROR: {e}")
        if result.get("difficulty_distribution"):
            print(f"  Difficulties: {result['difficulty_distribution']}")
        if result.get("source_distribution"):
            print(f"  Sources: {result['source_distribution']}")


if __name__ == "__main__":
    main()
