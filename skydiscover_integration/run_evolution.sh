#!/bin/bash
# AdaEvolve evolutionary search for CUDA kernel optimization.
# Uses doubleGraph A100 kernels as seeds and the configured remote A100 backend for evaluation.
#
# Implements AdaEvolve (multi-island UCB) + EvoX (self-evolving strategies)
# from the SkyDiscover framework — no external dependency required.
#
# Seeds from initial_kernels/ (adapted doubleGraph production code)
# Evaluator bridges to the shared KernelForge evaluator contract via skydiscover_integration/evaluator.py
#
# Usage:
#   ./run_evolution.sh              # Full run (50 iterations)
#   ./run_evolution.sh --dry-run    # Validate seeds only
#   ITERATIONS=10 ./run_evolution.sh  # Custom iteration count
#   N_ISLANDS=6 ./run_evolution.sh    # Custom island count

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SEED_DIR="$SCRIPT_DIR/initial_kernels"
OUTPUT_DIR="$PROJECT_ROOT/outputs/adaevolve"
ITERATIONS="${ITERATIONS:-50}"
N_ISLANDS="${N_ISLANDS:-4}"
EVAL_BACKEND="${KERNELFORGE_EVAL_BACKEND:-coreweave}"
EVAL_URL="${KERNELFORGE_EVAL_URL:-}"
MODAL_APP="${KERNELFORGE_MODAL_APP:-kernelforge-a100}"

# Parse flags
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

echo "=== AdaEvolve Evolutionary Search ==="
echo "Seeds:      $SEED_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Iterations: $ITERATIONS"
echo "Islands:    $N_ISLANDS"
echo "Dry run:    $DRY_RUN"
echo "Backend:    $EVAL_BACKEND"

mkdir -p "$OUTPUT_DIR"

if [[ "$EVAL_BACKEND" == "modal" ]]; then
    if ! modal token verify 2>/dev/null; then
        echo "WARNING: Modal token not configured. Stage 2 evaluation will fail."
        echo "  Run: modal token set"
    fi
elif [[ "$EVAL_BACKEND" == "coreweave" && -z "$EVAL_URL" ]]; then
    echo "WARNING: KERNELFORGE_EVAL_URL is not set. Stage 2 evaluation will fail on the CoreWeave path."
fi

# Run AdaEvolve
python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')

from skydiscover_integration.evaluator import KernelForgeEvaluator
from skydiscover_integration.adaevolve import AdaEvolve

evaluator = KernelForgeEvaluator(
    target_arch='sm_80',
    eval_mode='wcc',
)

# Load seed kernels
import os
seed_codes = []
seed_dir = '$SEED_DIR'
for f in sorted(os.listdir(seed_dir)):
    if f.endswith('.cu'):
        path = os.path.join(seed_dir, f)
        with open(path) as fh:
            seed_codes.append(fh.read())
        print(f'  Loaded seed: {f}')

print(f'Found {len(seed_codes)} seed kernels.')

# Validate seeds via stage1
for i, code in enumerate(seed_codes):
    result = evaluator.evaluate_stage1(code)
    status = 'PASS' if result.metrics.get('compiles') else 'FAIL'
    print(f'  Stage1 seed {i}: {status} (score={result.combined_score})')

dry_run = $([[ "$DRY_RUN" == "true" ]] && echo "True" || echo "False")
if dry_run:
    print('Dry run — seeds validated, skipping evolution.')
    sys.exit(0)

# Run AdaEvolve
evo = AdaEvolve(
    evaluator=evaluator,
    seeds=seed_codes,
    n_islands=$N_ISLANDS,
    budget=$ITERATIONS,
    output_dir='$OUTPUT_DIR',
)
results = evo.run()

print(f'\nTop 5 results:')
for i, r in enumerate(results[:5]):
    print(f'  {i+1}. score={r[\"score\"]:.3f}, strategy={r[\"strategy\"]}, gen={r[\"generation\"]}')
"

echo "=== Done ==="
