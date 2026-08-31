"""
================================================================================
AI-Driven Lyric Generation through Artistic Style Modeling and Context Adaptation
Supervised Fine-Tuning (SFT) Module (sft_trainer.py)
================================================================================
Step 2: Supervised Fine-Tuning (SFT Fine-Tuning)

Responsibilities:
1. Load Base LLM: Llama-3-8B-Instruct / Mistral-7B / Configurable Backbone.
2. Initialize Tokenizer & Register Special Context Tokens:
   `[Artist]`, `[Mood]`, `[Genre]`, `[Theme]`, `[Verse]`, `[Chorus]`, etc.
3. Configure Parameter-Efficient Fine-Tuning (PEFT):
   - LoRA / QLoRA with 4-bit NormalFloat (NF4) or native bfloat16.
   - Target all projection matrices for rich artistic adaptation.
4. Execute SFT training loop using `trl.SFTTrainer` with gradient checkpointing.
5. Save LoRA adapters, merged configuration, and tokenizer artifacts.
================================================================================
"""

import os
import sys
import argparse
import logging
from typing import Optional, List

import torch
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
from trl import SFTTrainer

# Import project configurations and dataset loaders
from config import ModelConfig, LoRAConfig, SFTTrainingConfig
from dataset_loader import get_dataset_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_tokenizer(model_name: str, special_tokens: List[str]):
    """
    Initialize tokenizer and register domain-specific special tokens for
    artistic conditioning (Artist, Mood, Genre, Theme, Structure tags).
    """
    logger.info(f"Loading tokenizer from '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"  # Standard for causal LM training

    # Add custom metadata and lyric section tokens
    new_tokens = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]
    if new_tokens:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        logger.info(f"Added {num_added} new special conditioning tokens to tokenizer: {new_tokens}")

    return tokenizer


def load_base_model(
    model_cfg: ModelConfig,
    tokenizer,
    device_map: str = "auto"
):
    """
    Load causal language model with hardware acceleration (BF16 / FP16, QLoRA 4-bit / 8-bit).
    """
    logger.info(f"Loading base model: {model_cfg.model_name_or_path}")

    # Determine compute dtype
    if model_cfg.torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif model_cfg.torch_dtype == "float16" and torch.cuda.is_available():
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    # Quantization Configuration (QLoRA)
    bnb_config = None
    if model_cfg.load_in_4bit and torch.cuda.is_available():
        logger.info("Enabling 4-bit QLoRA quantization (NF4 format with double quant)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        )
    elif model_cfg.load_in_8bit and torch.cuda.is_available():
        logger.info("Enabling 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Attention implementation selection
    attn_impl = model_cfg.attn_implementation
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning("FlashAttention-2 package not found. Falling back to PyTorch SDPA.")
            attn_impl = "sdpa"

    model_kwargs = {
        "pretrained_model_name_or_path": model_cfg.model_name_or_path,
        "quantization_config": bnb_config,
        "device_map": device_map if torch.cuda.is_available() else None,
        "torch_dtype": compute_dtype,
        "trust_remote_code": model_cfg.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Resize token embeddings if new special tokens were added
    if len(tokenizer) > model.config.vocab_size:
        logger.info(f"Resizing model token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Prepare model for PEFT training if quantized
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True
        )

    return model


def build_peft_lora_model(model, lora_cfg: LoRAConfig):
    """Wrap base model with parameter-efficient LoRA adapters."""
    logger.info(f"Applying LoRA (rank={lora_cfg.r}, alpha={lora_cfg.lora_alpha}, dropout={lora_cfg.lora_dropout})...")
    
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.target_modules,
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model, peft_config


def train_sft(
    model_cfg: Optional[ModelConfig] = None,
    lora_cfg: Optional[LoRAConfig] = None,
    sft_cfg: Optional[SFTTrainingConfig] = None,
    language: str = "english",
    max_samples: Optional[int] = None,
):
    """
    Complete Supervised Fine-Tuning Execution:
    Data Preparation -> Model & LoRA Instantiation -> SFTTrainer Execution -> Save Checkpoint.
    """
    model_cfg = model_cfg or ModelConfig()
    lora_cfg = lora_cfg or LoRAConfig()
    sft_cfg = sft_cfg or SFTTrainingConfig()

    os.makedirs(sft_cfg.output_dir, exist_ok=True)

    # 1. Dataset Ingestion
    logger.info(f"--- [Step 1/4: Ingestion] Preparing {language.upper()} lyric dataset ---")
    loader = get_dataset_loader(
        dataset_name_or_path=sft_cfg.dataset_name,
        language=language,
        max_samples=max_samples,
        seed=sft_cfg.seed,
    )
    dataset_dict = loader.prepare_dataset()
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"]

    # 2. Tokenizer & Model Setup
    logger.info("--- [Step 2/4: SFT Setup] Initializing Tokenizer and Base Model ---")
    tokenizer = setup_tokenizer(model_cfg.model_name_or_path, model_cfg.special_tokens)
    base_model = load_base_model(model_cfg, tokenizer)
    peft_model, peft_config = build_peft_lora_model(base_model, lora_cfg)

    # 3. Training Arguments (with adaptive SFTConfig / TrainingArguments support)
    raw_kwargs = {
        "output_dir": sft_cfg.output_dir,
        "num_train_epochs": sft_cfg.num_train_epochs,
        "per_device_train_batch_size": sft_cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": sft_cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": sft_cfg.gradient_accumulation_steps,
        "learning_rate": sft_cfg.learning_rate,
        "weight_decay": sft_cfg.weight_decay,
        "lr_scheduler_type": sft_cfg.lr_scheduler_type,
        "logging_steps": sft_cfg.logging_steps,
        "save_strategy": sft_cfg.save_strategy,
        "save_steps": sft_cfg.save_steps,
        "save_total_limit": sft_cfg.save_total_limit,
        "eval_strategy": sft_cfg.eval_strategy,
        "eval_steps": sft_cfg.eval_steps,
        "gradient_checkpointing": sft_cfg.gradient_checkpointing,
        "fp16": sft_cfg.fp16,
        "bf16": sft_cfg.bf16,
        "report_to": sft_cfg.report_to if sft_cfg.report_to in ["tensorboard", "wandb"] else "none",
        "seed": sft_cfg.seed,
        "max_length": sft_cfg.max_seq_length,
        "dataset_text_field": "text",
    }
    try:
        import inspect
        from trl import SFTConfig
        sig = inspect.signature(SFTConfig.__init__)
        valid_kwargs = {k: v for k, v in raw_kwargs.items() if k in sig.parameters}
        training_args = SFTConfig(**valid_kwargs)
    except Exception:
        import inspect
        sig = inspect.signature(TrainingArguments.__init__)
        valid_kwargs = {k: v for k, v in raw_kwargs.items() if k in sig.parameters}
        training_args = TrainingArguments(**valid_kwargs)

    # 4. SFT Trainer Initialization
    logger.info("--- Starting Supervised Fine-Tuning Loop ---")
    try:
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=sft_cfg.max_seq_length,
            tokenizer=tokenizer,
        )

    # 5. Train & Save
    train_result = trainer.train()
    logger.info(f"SFT Training completed. Train metrics: {train_result.metrics}")

    # Save final LoRA adapter and tokenizer
    final_save_dir = os.path.join(sft_cfg.output_dir, "final_adapter")
    logger.info(f"Saving final LoRA adapters and tokenizer to '{final_save_dir}'...")
    trainer.model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)

    # Evaluate on validation split
    eval_metrics = trainer.evaluate()
    logger.info(f"Validation Evaluation Metrics: {eval_metrics}")

    return final_save_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning (SFT) for Artistic Lyric Generation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base LLM HuggingFace ID (e.g. meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.3)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="manoh2f2/tsterbak-lyrics-dataset-with-emotions",
        help="HuggingFace dataset path",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "tamil"],
        help="Language phase selector (Phase 1: english, Phase 2: tamil)",
    )
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft_lyric_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick test run")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (full 16-bit LoRA)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    m_cfg = ModelConfig(
        model_name_or_path=args.model_name,
        load_in_4bit=not args.no_4bit,
    )
    l_cfg = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    s_cfg = SFTTrainingConfig(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
    )

    train_sft(
        model_cfg=m_cfg,
        lora_cfg=l_cfg,
        sft_cfg=s_cfg,
        language=args.language,
        max_samples=args.max_samples,
    )
