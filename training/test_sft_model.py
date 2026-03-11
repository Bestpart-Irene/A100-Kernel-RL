#!/usr/bin/env python3
"""
Test the SFT model's CUDA generation capabilities.

This script evaluates how well the model learned CUDA patterns
from supervised fine-tuning.
"""

import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import modal

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Modal setup
app = modal.App("kernelforge-sft-test")
cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
    add_python="3.12",
).uv_pip_install(
    "transformers>=4.40.0",
    "torch>=2.4",
    "numpy>=1.26",
)

# Model path
MODEL_PATH = os.getenv("SFT_MODEL_PATH", str(Path(__file__).parent.parent / "models" / "qwen9b-cuda-sft"))

# Test prompts
TEST_PROMPTS = [
    {
        "name": "Vector Add",
        "prompt": """Write a CUDA kernel for A100 (sm_80) to add two vectors element-wise.

GPU: A100 (sm_80), 40MB L2 cache, 108 SMs
Architecture: Ampere, Tensor Cores available  
Memory: HBM2e, bandwidth 1.55 TB/s

A100 Optimization Guidelines:
- Exploit 40MB L2 for bitmaps < 128KB
- Use degree-aware scheduling for >1M vertices
- Consider warp-per-vertex for high-degree nodes
- Use __shfl_sync for warp-level reductions
- Leverage cooperative groups for complex synchronization

Requirements:
- Output ONLY CUDA code, no explanations
- Include extern "C" __global__ signature
- Keep kernel under 800 tokens

Algorithm: vector_add
Category: ops
Variant: base

Output format:
extern "C" __global__ void kernel_name(...) {
    // Implementation
}""",
        "expected_patterns": ["extern \"C\"", "__global__", "threadIdx", "blockIdx"]
    },
    {
        "name": "WCC Initialize",
        "prompt": """Write CUDA kernels for Weakly Connected Components using Union-Find algorithm on A100.

GPU: A100 (sm_80), 40MB L2 cache, 108 SMs
Architecture: Ampere, Tensor Cores available  
Memory: HBM2e, bandwidth 1.55 TB/s

A100 Optimization Guidelines:
- Exploit 40MB L2 for bitmaps < 128KB
- Use degree-aware scheduling for >1M vertices
- Consider warp-per-vertex for high-degree nodes
- Use __shfl_sync for warp-level reductions
- Leverage cooperative groups for complex synchronization

Requirements:
- Output ONLY CUDA code, no explanations
- Include extern "C" __global__ signature
- Keep kernel under 800 tokens

Algorithm: weakly_connected_components
Category: components
Variant: base

Output format:
extern "C" __global__ void kernel_name(...) {
    // Implementation
}""",
        "expected_patterns": ["extern \"C\"", "__global__", "atomic", "parent"]
    },
    {
        "name": "BFS Frontier",
        "prompt": """Write a BFS kernel for A100 using frontier-based approach.

GPU: A100 (sm_80), 40MB L2 cache, 108 SMs
Architecture: Ampere, Tensor Cores available  
Memory: HBM2e, bandwidth 1.55 TB/s

A100 Optimization Guidelines:
- Exploit 40MB L2 for bitmaps < 128KB
- Use degree-aware scheduling for >1M vertices
- Consider warp-per-vertex for high-degree nodes
- Use __shfl_sync for warp-level reductions
- Leverage cooperative groups for complex synchronization

Requirements:
- Output ONLY CUDA code, no explanations
- Include extern "C" __global__ signature
- Keep kernel under 800 tokens

Algorithm: bfs
Category: traversal
Variant: base

Output format:
extern "C" __global__ void kernel_name(...) {
    // Implementation
}""",
        "expected_patterns": ["extern \"C\"", "__global__", "frontier", "distances"]
    }
]

def load_model_and_tokenizer(model_path: str):
    """Load fine-tuned model and tokenizer."""
    print(f"Loading SFT model from {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    
    print(f"Model loaded successfully")
    return model, tokenizer

def generate_cuda_code(model, tokenizer, prompt: str, max_tokens: int = 1024):
    """Generate CUDA code from prompt."""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the completion
    completion = generated_text[len(prompt):].strip()
    
    return completion

def evaluate_cuda_generation(completion: str, expected_patterns: list) -> dict:
    """Evaluate generated CUDA code."""
    results = {
        "has_extern_c": "extern \"C\"" in completion,
        "has_global": "__global__" in completion,
        "has_braces": "{" in completion and "}" in completion,
        "token_count": len(completion.split()),
        "pattern_matches": [],
        "compilation_ready": False
    }
    
    # Check expected patterns
    for pattern in expected_patterns:
        found = pattern in completion
        results["pattern_matches"].append({
            "pattern": pattern,
            "found": found
        })
    
    # Basic compilation readiness check
    if results["has_extern_c"] and results["has_global"] and results["has_braces"]:
        results["compilation_ready"] = True
    
    return results

@app.function(
    image=cuda_image,
    gpu="A100",
    timeout=1800,
)
def test_sft_model():
    """Test SFT model on CUDA generation tasks."""
    print("=== Testing SFT Model CUDA Generation ===")
    
    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    except FileNotFoundError:
        print(f"SFT model not found at {MODEL_PATH}")
        print("Please run stage0_sft.py first to train the model")
        return {"status": "error", "message": "Model not found"}
    
    # Test each prompt
    results = []
    
    for test_case in TEST_PROMPTS:
        print(f"\n--- Testing: {test_case['name']} ---")
        
        # Generate code
        completion = generate_cuda_code(model, tokenizer, test_case['prompt'])
        
        # Evaluate
        eval_results = evaluate_cuda_generation(completion, test_case['expected_patterns'])
        
        # Store results
        result = {
            "name": test_case['name'],
            "prompt_length": len(test_case['prompt']),
            "completion_length": len(completion),
            "completion_preview": completion[:200] + "..." if len(completion) > 200 else completion,
            "evaluation": eval_results
        }
        results.append(result)
        
        # Print results
        print(f"Generated {len(completion)} tokens")
        print(f"Has extern \"C\": {eval_results['has_extern_c']}")
        print(f"Has __global__: {eval_results['has_global']}")
        print(f"Compilation ready: {eval_results['compilation_ready']}")
        print(f"Pattern matches: {sum(1 for p in eval_results['pattern_matches'] if p['found'])}/{len(eval_results['pattern_matches'])}")
        
        # Show completion
        print("\nGenerated code:")
        print(completion)
        print("\n" + "="*50)
    
    # Summary
    print("\n=== Summary ===")
    total_tests = len(results)
    compilation_ready = sum(1 for r in results if r['evaluation']['compilation_ready'])
    extern_c_count = sum(1 for r in results if r['evaluation']['has_extern_c'])
    global_count = sum(1 for r in results if r['evaluation']['has_global'])
    
    print(f"Total tests: {total_tests}")
    print(f"Compilation ready: {compilation_ready}/{total_tests} ({compilation_ready/total_tests*100:.1f}%)")
    print(f"Has extern \"C\": {extern_c_count}/{total_tests} ({extern_c_count/total_tests*100:.1f}%)")
    print(f"Has __global__: {global_count}/{total_tests} ({global_count/total_tests*100:.1f}%)")
    
    # Overall assessment
    if compilation_ready >= total_tests * 0.6:
        print("✅ SFT training successful - model ready for Stage 1 GRPO")
    elif compilation_ready >= total_tests * 0.3:
        print("⚠️  SFT training partially successful - consider more epochs")
    else:
        print("❌ SFT training needs improvement - model not ready for RL")
    
    return {
        "status": "completed",
        "results": results,
        "summary": {
            "total_tests": total_tests,
            "compilation_ready": compilation_ready,
            "success_rate": compilation_ready / total_tests
        }
    }

@app.local_entrypoint()
def main():
    """Run SFT model testing."""
    print("=== Testing SFT Model CUDA Generation ===")
    
    # Run on Modal
    result = test_sft_model.remote()
    
    if result['status'] == 'completed':
        print(f"\nTest completed with {result['summary']['success_rate']*100:.1f}% success rate")
        
        if result['summary']['success_rate'] >= 0.6:
            print("\n🚀 Ready to proceed to Stage 1 GRPO training!")
        else:
            print("\n⚠️  Consider improving SFT training before RL")
    else:
        print(f"Test failed: {result['message']}")

if __name__ == "__main__":
    main()
