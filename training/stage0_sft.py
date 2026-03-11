#!/usr/bin/env python3
"""
Stage 0: Supervised Fine-Tuning on CUDA kernels.

This stage teaches the model basic CUDA syntax and patterns before
moving to reinforcement learning.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import modal

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Modal setup
app = modal.App("kernelforge-sft")
cuda_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
    add_python="3.12",
).uv_pip_install(
    "transformers>=4.40.0",
    "datasets>=2.18.0",
    "accelerate>=0.29.0",
    "peft>=0.10.0",
    "bitsandbytes>=0.43.0",
    "wandb>=0.16.0",
    "torch>=2.4",
    "numpy>=1.26",
)

# Model configuration
MODEL_NAME = os.getenv("SFT_MODEL_NAME", "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled")
SFT_DATA_PATH = os.getenv("SFT_DATA_PATH", str(Path(__file__).parent.parent / "datasets/basic_cuda_sft.jsonl"))
OUTPUT_DIR = os.getenv("SFT_OUTPUT_DIR", str(Path(__file__).parent.parent / "models" / "qwen9b-cuda-sft"))

# Training hyperparameters
SFT_CONFIG = {
    "batch_size": int(os.getenv("SFT_BATCH_SIZE", "4")),
    "gradient_accumulation_steps": int(os.getenv("SFT_GRADIENT_ACCUMULATION_STEPS", "2")),
    "learning_rate": float(os.getenv("SFT_LEARNING_RATE", "2e-5")),
    "num_epochs": int(os.getenv("SFT_NUM_EPOCHS", "3")),
    "max_length": int(os.getenv("SFT_MAX_LENGTH", "2048")),
    "warmup_steps": int(os.getenv("SFT_WARMUP_STEPS", "10")),
    "save_steps": int(os.getenv("SFT_SAVE_STEPS", "100")),
    "eval_steps": int(os.getenv("SFT_EVAL_STEPS", "50")),
    "logging_steps": int(os.getenv("SFT_LOGGING_STEPS", "10")),
}

def load_sft_dataset(data_path: str) -> Dataset:
    """Load and prepare SFT dataset."""
    print(f"Loading SFT dataset from {data_path}")
    
    examples = []
    with open(data_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Format for supervised fine-tuning
    texts = []
    for example in examples:
        prompt = example['prompt']
        completion = example['completion']
        
        # Combine prompt and completion
        full_text = f"{prompt}\n\n{completion}"
        texts.append(full_text)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Split train/eval
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    return train_test_split['train'], train_test_split['test']

def format_prompt(example: Dict[str, Any]) -> str:
    """Format a single example for training."""
    prompt = example['prompt']
    completion = example['completion']
    
    # Add special tokens if needed
    # For Qwen models, we can use the standard format
    return f"{prompt}\n\n{completion}"

def load_model_and_tokenizer(model_name: str):
    """Load model and tokenizer for SFT."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False  # Important for training
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model loaded successfully")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the dataset."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_overflowing_tokens=False,
    )

@app.function(
    image=cuda_image,
    gpu="A100",
    timeout=3600,
    volumes={"/cache": modal.Volume.from_name("kernelforge-cache", create_if_missing=True)},
)
def run_sft_training():
    """Run supervised fine-tuning."""
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    import wandb
    
    # Initialize wandb
    wandb.init(
        project="kernelforge-sft",
        config=SFT_CONFIG,
        name="qwen9b-cuda-sft"
    )
    
    print("=== Stage 0: CUDA SFT Training ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {SFT_DATA_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Load dataset
    train_dataset, eval_dataset = load_sft_dataset(SFT_DATA_PATH)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, SFT_CONFIG["max_length"]),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer, SFT_CONFIG["max_length"]),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=SFT_CONFIG["num_epochs"],
        per_device_train_batch_size=SFT_CONFIG["batch_size"],
        per_device_eval_batch_size=SFT_CONFIG["batch_size"],
        gradient_accumulation_steps=SFT_CONFIG["gradient_accumulation_steps"],
        learning_rate=SFT_CONFIG["learning_rate"],
        warmup_steps=SFT_CONFIG["warmup_steps"],
        logging_steps=SFT_CONFIG["logging_steps"],
        save_steps=SFT_CONFIG["save_steps"],
        eval_steps=SFT_CONFIG["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        bf16=True,  # Use bfloat16 for A100
        dataloader_num_workers=0,  # Modal compatibility
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate
    print("Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    
    wandb.finish()
    
    return {
        "status": "completed",
        "eval_loss": eval_results['eval_loss'],
        "output_dir": OUTPUT_DIR
    }

@app.local_entrypoint()
def main():
    """Run SFT training locally or on Modal."""
    print("=== KernelForge Stage 0: CUDA SFT ===")
    print("Teaching model CUDA fundamentals before RL...")
    
    # Run on Modal
    result = run_sft_training.remote()
    print(f"SFT Training completed: {result}")
    
    # Print next steps
    print("\n=== Next Steps ===")
    print("1. Model now understands basic CUDA patterns")
    print("2. Ready for Stage 1: GRPO training with CUDA kernels")
    print("3. Expected improvement in compilation success rate: 0% → 60-80%")

if __name__ == "__main__":
    main()
