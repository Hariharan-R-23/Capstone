"""
================================================================================
AI-Driven Lyric Generation: Top 5 Artists Fine-Tuning Pipeline
Trains LoRA adapters exclusively on the Top 5 artists dataset:
- Elvis Presley
- Willie Nelson
- Ella Fitzgerald
- Frank Sinatra
- Dean Martin

Hardware Tuning:
- GPU: Dedicated NVIDIA RTX 5050 Laptop GPU (max acceleration via 4-bit QLoRA)
- RAM: Ultra-low RAM footprint (<1.5 GB allocated) with zero worker-forking
- Output Folder: ./checkpoints/top5_artists_lyric_model
================================================================================
"""

import os
import sys
import gc
import json
import logging
import torch
import psutil
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_top5_model(
    dataset_path: str = "./top5_dataset",
    output_dir: str = "./checkpoints/top5_artists_lyric_model",
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    max_steps: int = 40,
    batch_size: int = 2,
    grad_accum: int = 1,
    learning_rate: float = 3e-4,
    max_seq_length: int = 256,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=" * 70)
    logger.info("STARTING RAPID TRAINING FOR TOP 5 ARTISTS LYRIC GENERATION MODEL")
    logger.info(f"Target Output Directory: {output_dir}")
    logger.info(f"Base Model: {base_model_name}")
    logger.info(f"Max Steps: {max_steps} | Batch Size: {batch_size} | Grad Accum: {grad_accum}")
    logger.info(f"Available System RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)")
    logger.info("=" * 70)

    # 1. Load Pre-Formatted Dataset
    logger.info(f"Loading memory-mapped dataset from '{dataset_path}'...")
    dataset_dict = load_from_disk(dataset_path)
    train_ds = dataset_dict["train"].select_columns(["text"])
    val_ds = dataset_dict["validation"].select_columns(["text"])
    logger.info(f"Loaded Train: {len(train_ds)} samples | Validation: {len(val_ds)} samples")

    # 2. Setup Tokenizer
    logger.info(f"Loading Tokenizer from '{base_model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    tokenizer.padding_side = "right"

    # Add custom conditioning tokens
    special_tokens = ["[Artist]", "[Mood]", "[Genre]", "[Theme]", "[Lyrics]"]
    new_tokens = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info(f"Registered special tokens: {new_tokens}")

    # 3. Setup 4-bit Quantized Base Model
    compute_dtype = torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading Base LLM '{base_model_name}' directly onto GPU in 4-bit NF4 with bfloat16 compute...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Fast training prep (gradient checkpointing disabled for 2x faster forward-backward)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # 4. Attach LoRA Adapter
    logger.info(f"Initializing LoRA config: r={lora_r}, alpha={lora_alpha}...")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Training Arguments configured for safe low RAM and high GPU compute
    import inspect
    raw_kwargs = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 5,
        "logging_steps": 5,
        "save_strategy": "no",
        "eval_strategy": "no",
        "gradient_checkpointing": False,
        "fp16": False,
        "bf16": True,
        "optim": "paged_adamw_8bit",
        "dataloader_num_workers": 0,
        "report_to": "none",
        "seed": 42,
        "max_length": max_seq_length,
        "dataset_text_field": "text",
    }
    sig = inspect.signature(SFTConfig.__init__)
    valid_kwargs = {k: v for k, v in raw_kwargs.items() if k in sig.parameters}
    training_args = SFTConfig(**valid_kwargs)

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    # Clean memory right before starting training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Starting training loop across all Top 5 artists songs...")
    train_output = trainer.train()
    logger.info(f"Training completed successfully! Train metrics: {train_output.metrics}")

    # 6. Save final adapter and tokenizer
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    logger.info(f"Saving final fine-tuned LoRA adapters to '{final_adapter_dir}'...")
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # 7. Evaluate on validation set
    logger.info("Running final evaluation on validation split...")
    eval_metrics = trainer.evaluate()
    logger.info(f"Validation metrics: {eval_metrics}")

    # Save summary metadata
    meta = {
        "base_model": base_model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "train_metrics": train_output.metrics,
        "eval_metrics": eval_metrics,
        "top_5_artists": [
            "Elvis Presley",
            "Willie Nelson",
            "Ella Fitzgerald",
            "Frank Sinatra",
            "Dean Martin"
        ]
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Training complete! Final model is ready in: {final_adapter_dir}")
    return final_adapter_dir


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    train_top5_model()
