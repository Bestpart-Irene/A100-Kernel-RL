#!/usr/bin/env python3
"""
Stage 1: Enhanced GRPO training with SFT foundation.

This stage uses the SFT-trained model and applies contrastive RL
techniques for CUDA optimization.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import modal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import logging
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our custom components
from training.custom_grpo_trainer import TRLOOGRPOTrainer
from training.task_support import build_modal_payload, build_reward_contract
from openenv_env.eval_backend import evaluate_code_remote

# Modal setup
app = modal.App("kernelforge-enhanced-grpo")
cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
    add_python="3.12",
).uv_pip_install(
    "transformers>=4.40.0",
    "datasets>=2.18.0",
    "accelerate>=0.29.0",
    "peft>=0.10.0",
    "bitsandbytes>=0.43.0",
    "trl>=0.11.0",
    "wandb>=0.16.0",
    "torch>=2.4",
    "numpy>=1.26",
    "openenv-core[core]>=0.2.1",
).add_local_python_source("training", "openenv_env", "eval_service")

# Model configuration
SFT_MODEL_PATH = os.getenv("SFT_MODEL_PATH", str(Path(__file__).parent.parent / "models" / "qwen9b-cuda-sft"))
DATASET_PATH = os.getenv("GRPO_DATASET_PATH", str(Path(__file__).parent.parent / "datasets/combined_kernelforge.jsonl"))

# Enhanced GRPO configuration
GRPO_CONFIG = {
    "output_dir": os.getenv("GRPO_OUTPUT_DIR", str(Path(__file__).parent.parent / "models" / "qwen9b-cuda-grpo-enhanced")),
    "num_train_epochs": int(os.getenv("GRPO_NUM_EPOCHS", "1")),
    "per_device_train_batch_size": int(os.getenv("GRPO_BATCH_SIZE", "4")),
    "gradient_accumulation_steps": int(os.getenv("GRPO_GRADIENT_ACCUMULATION", "1")),
    "learning_rate": float(os.getenv("GRPO_LEARNING_RATE", "1e-6")),
    "lr_scheduler_type": "cosine",
    "warmup_ratio": float(os.getenv("GRPO_WARMUP_RATIO", "0.1")),
    "logging_steps": int(os.getenv("GRPO_LOGGING_STEPS", "1")),
    "save_steps": int(os.getenv("GRPO_SAVE_STEPS", "10")),
    "max_steps": int(os.getenv("GRPO_MAX_STEPS", "50")),
    "max_completion_length": int(os.getenv("GRPO_MAX_COMPLETION_LENGTH", "1024")),
    "num_generations": int(os.getenv("GRPO_NUM_GENERATIONS", "4")),  # G for GRPO
    "beta": float(os.getenv("GRPO_BETA", "0.01")),  # KL penalty
    "use_trloo": os.getenv("GRPO_USE_TRLOO", "true").lower() == "true",
    "contrastive_learning": os.getenv("GRPO_CONTRASTIVE", "true").lower() == "true",
}

def load_rl_dataset(dataset_path: str) -> Dataset:
    """Load RL dataset with enhanced prompts."""
    print(f"Loading RL dataset from {dataset_path}")
    
    examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Enhance prompts with A100-specific guidance
    enhanced_examples = []
    
    for example in examples:
        enhanced_prompt = f"""{example['prompt']}

CRITICAL REQUIREMENTS:
- Output ONLY CUDA code, no explanations or comments
- Include extern "C" __global__ signature exactly
- Keep kernel under 800 tokens
- Use A100-specific optimizations:
  * Leverage 40MB L2 cache for small data structures
  * Use __shfl_sync for warp-level reductions
  * Consider cooperative groups for complex synchronization
  * Optimize for memory bandwidth

Expected signature pattern:
extern "C" __global__ void kernel_name(
    const int* row_ptr,
    const int* col_idx,
    int num_vertices,
    int* output
) {{
    // Implementation
}}"""
        
        enhanced_examples.append({
            **example,
            "prompt": enhanced_prompt
        })
    
    return Dataset.from_list(enhanced_examples)

def load_sft_model(model_path: str):
    """Load SFT-trained model for GRPO."""
    print(f"Loading SFT-trained model from {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"SFT model not found at {model_path}")
        print("Falling back to base model")
        model_path = "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled"
    
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
        trust_remote_code=True,
        use_cache=False
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model loaded successfully")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model, tokenizer

def contrastive_reward_function(generations: List[str], task_data: Dict, **kwargs) -> List[float]:
    """
    Enhanced reward function with contrastive learning.
    
    Compares multiple generations and rewards the best performing one.
    """
    rewards = []
    
    for i, generation in enumerate(generations):
        try:
            # Build evaluation payload
            fn_name, payload = build_modal_payload(generation, task_data)
            
            # Evaluate
            eval_result = evaluate_code_remote(fn_name, payload)
            
            # Extract reward using contract
            reward_info = build_reward_contract(eval_result)
            reward = reward_info["reward"]
            
            # Contrastive bonus: reward relative to other generations
            if GRPO_CONFIG["contrastive_learning"] and len(generations) > 1:
                # Simple contrastive: best gets bonus
                other_rewards = []
                for j, other_gen in enumerate(generations):
                    if i != j:
                        fn_name_j, payload_j = build_modal_payload(other_gen, task_data)
                        eval_result_j = evaluate_code_remote(fn_name_j, payload_j)
                        reward_info_j = build_reward_contract(eval_result_j)
                        other_rewards.append(reward_info_j["reward"])
                
                if other_rewards:
                    avg_other = np.mean(other_rewards)
                    # Bonus if better than average
                    if reward > avg_other:
                        contrastive_bonus = 0.1 * (reward - avg_other)
                        reward += contrastive_bonus
            
            rewards.append(reward)
            
        except Exception as e:
            print(f"Evaluation error for generation {i}: {e}")
            rewards.append(-1.0)  # Penalty for failed evaluation
    
    return rewards

def create_enhanced_prompts(generations: List[str], rewards: List[float], original_prompt: str) -> str:
    """
    Create contrastive learning prompt showing performance differences.
    """
    if not GRPO_CONFIG["contrastive_learning"] or len(generations) < 2:
        return original_prompt
    
    # Sort by reward
    sorted_indices = np.argsort(rewards)[::-1]  # Descending
    
    # Build contrastive prompt
    contrastive_prompt = f"""{original_prompt}

For reference, here are some previous implementations and their performance:

"""
    
    for i, idx in enumerate(sorted_indices[:3]):  # Show top 3
        contrastive_prompt += f"""
Variant {i+1} (Performance: {rewards[idx]:.2f}):
```cuda
{generations[idx][:500]}...
```
"""
    
    contrastive_prompt += """
Based on the above examples, write an improved CUDA kernel that achieves better performance.
Focus on the patterns that made higher-performing variants successful.
"""
    
    return contrastive_prompt

@app.function(
    image=cuda_image,
    gpu="A100",
    timeout=7200,
    volumes={"/cache": modal.Volume.from_name("kernelforge-cache", create_if_missing=True)},
)
def run_enhanced_grpo():
    """Run enhanced GRPO training."""
    import wandb
    
    # Initialize wandb
    wandb.init(
        project="kernelforge-enhanced-grpo",
        config=GRPO_CONFIG,
        name="qwen9b-cuda-grpo-enhanced"
    )
    
    print("=== Stage 1: Enhanced GRPO Training ===")
    print(f"SFT Model: {SFT_MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Using TRLOO: {GRPO_CONFIG['use_trloo']}")
    print(f"Contrastive Learning: {GRPO_CONFIG['contrastive_learning']}")
    
    # Load model and tokenizer
    model, tokenizer = load_sft_model(SFT_MODEL_PATH)
    
    # Load dataset
    dataset = load_rl_dataset(DATASET_PATH)
    
    print(f"Training examples: {len(dataset)}")
    
    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=GRPO_CONFIG["output_dir"],
        num_train_epochs=GRPO_CONFIG["num_train_epochs"],
        per_device_train_batch_size=GRPO_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=GRPO_CONFIG["gradient_accumulation_steps"],
        learning_rate=GRPO_CONFIG["learning_rate"],
        lr_scheduler_type=GRPO_CONFIG["lr_scheduler_type"],
        warmup_ratio=GRPO_CONFIG["warmup_ratio"],
        logging_steps=GRPO_CONFIG["logging_steps"],
        save_steps=GRPO_CONFIG["save_steps"],
        max_steps=GRPO_CONFIG["max_steps"],
        max_completion_length=GRPO_CONFIG["max_completion_length"],
        num_generations=GRPO_CONFIG["num_generations"],
        beta=GRPO_CONFIG["beta"],
        generation_kwargs={
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
    
    # Choose trainer
    if GRPO_CONFIG["use_trloo"]:
        trainer_class = TRLOOGRPOTrainer
        print("Using TRLOO-enhanced GRPO trainer")
    else:
        trainer_class = GRPOTrainer
        print("Using standard GRPO trainer")
    
    # Create trainer
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_function=contrastive_reward_function,
        # For contrastive learning
        **({"create_prompts": create_enhanced_prompts} if GRPO_CONFIG["contrastive_learning"] else {})
    )
    
    # Train
    print("Starting enhanced GRPO training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {GRPO_CONFIG['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(GRPO_CONFIG["output_dir"])
    
    # Final evaluation
    print("Running final evaluation...")
    # TODO: Add comprehensive evaluation
    
    wandb.finish()
    
    return {
        "status": "completed",
        "output_dir": GRPO_CONFIG["output_dir"],
        "config": GRPO_CONFIG
    }

@app.local_entrypoint()
def main():
    """Run enhanced GRPO training."""
    print("=== KernelForge Stage 1: Enhanced GRPO ===")
    print("Building on SFT foundation with contrastive RL...")
    
    # Check prerequisites
    if not os.path.exists(SFT_MODEL_PATH):
        print(f"⚠️  SFT model not found at {SFT_MODEL_PATH}")
        print("Run stage0_sft.py first, or the script will use base model")
    
    # Run on Modal
    result = run_enhanced_grpo.remote()
    print(f"Enhanced GRPO training completed: {result}")
    
    print("\n=== Expected Improvements ===")
    print("Compilation success rate: 0% → 60-80% (from SFT)")
    print("Positive reward rate: 0% → 20-40% (from GRPO)")
    print("Speedup achievement: 0% → 10-20% (from contrastive learning)")

if __name__ == "__main__":
    main()
