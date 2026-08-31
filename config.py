"""
================================================================================
AI-Driven Lyric Generation through Artistic Style Modeling and Context Adaptation
Configuration Module (config.py)
================================================================================
Defines unified, strongly-typed configuration dataclasses for:
1. Base Model & Tokenizer Configuration
2. Parameter-Efficient Fine-Tuning (LoRA / QLoRA) Configuration
3. Supervised Fine-Tuning (SFT) Training Arguments
4. Multi-Objective Reward Engine Configuration
5. PPO Reinforcement Learning Training Configuration
6. Inference & Evaluation Configuration
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for base model loading, precision, and quantization."""
    # Pretrained model identifier on Hugging Face Hub
    # Suggested high-performance models:
    # - "meta-llama/Meta-Llama-3-8B-Instruct"
    # - "mistralai/Mistral-7B-Instruct-v0.3"
    # - "Qwen/Qwen2.5-7B-Instruct"
    # - "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (for rapid testing)
    model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Precision and hardware optimization (configured for high-end GPUs)
    torch_dtype: str = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    attn_implementation: str = "sdpa"  # "flash_attention_2" or "sdpa" (Scaled Dot Product Attention)
    use_fast_tokenizer: bool = True
    trust_remote_code: bool = True

    # Quantization Options (QLoRA vs Full Precision LoRA)
    # On high-end GPUs with ample VRAM (e.g., 24GB+ like RTX 3090/4090/A100),
    # load_in_4bit can be set to False for faster full-precision bfloat16 training.
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Special tokens used to structure context-conditioned song lyrics
    special_tokens: List[str] = field(default_factory=lambda: [
        "[Artist]",
        "[Mood]",
        "[Genre]",
        "[Theme]",
        "[Intro]",
        "[Verse]",
        "[Chorus]",
        "[Bridge]",
        "[Outro]",
        "[Lyrics]",
    ])


@dataclass
class LoRAConfig:
    """Configuration for Parameter-Efficient Fine-Tuning (LoRA/QLoRA)."""
    r: int = 32                      # LoRA rank (higher rank captures complex poetic styles)
    lora_alpha: int = 64             # Scaling factor (typically 2x rank)
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    # Target all linear projections for maximum style and thematic adaptability
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ])


@dataclass
class SFTTrainingConfig:
    """Configuration for Supervised Fine-Tuning (Stage 2)."""
    output_dir: str = "./checkpoints/sft_lyric_model"
    dataset_name: str = "manoh2f2/tsterbak-lyrics-dataset-with-emotions"
    max_seq_length: int = 512
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 100
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False
    report_to: str = "tensorboard"   # "wandb" or "tensorboard" or "none"
    seed: int = 42


@dataclass
class RewardConfig:
    """Configuration for Multi-Objective Reward Engine (Stage 3)."""
    # Sentence Transformer for Mood / Emotion Semantic Alignment
    mood_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Reward Component Weights (must sum to 1.0 ideally, or adjust relative influence)
    rhyme_weight: float = 0.45       # Phonetic Rhyme Density & Couplet Structure
    mood_weight: float = 0.40        # Semantic Mood / Context Alignment
    fluency_weight: float = 0.15     # Repetition Penalty & Stanza Coherence

    # Reward Scaling & Normalization
    rhyme_scale: float = 1.0
    mood_scale: float = 1.0
    repetition_penalty_factor: float = 0.5
    min_line_count: int = 4          # Minimum required lyric lines for valid stanza evaluation


@dataclass
class PPOConfig:
    """Configuration for PPO Reinforcement Learning Policy Optimization (Stage 4)."""
    sft_model_path: str = "./checkpoints/sft_lyric_model"
    output_dir: str = "./checkpoints/ppo_lyric_model"
    
    learning_rate: float = 1.41e-5
    batch_size: int = 16             # Effective PPO batch size
    mini_batch_size: int = 4         # Mini-batch for PPO gradient updates
    gradient_accumulation_steps: int = 2
    ppo_epochs: int = 4              # PPO optimization epochs per batch
    
    # KL Divergence and Reward Steering
    init_kl_coef: float = 0.1        # Initial KL penalty to prevent policy drift
    target_kl: float = 6.0           # Target KL divergence
    adap_kl_ctrl: bool = True        # Adaptive KL controller
    
    # Generation parameters during RL rollout
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_p: float = 0.92
    top_k: int = 50
    
    # Total PPO training steps
    total_steps: int = 500
    save_freq: int = 50
    log_with: str = "tensorboard"    # "wandb" or "tensorboard"


@dataclass
class InferenceConfig:
    """Configuration for controllable lyric generation at inference time."""
    model_path: str = "./checkpoints/sft_lyric_model"
    max_new_tokens: int = 384
    temperature: float = 0.75
    top_p: float = 0.90
    top_k: int = 50
    repetition_penalty: float = 1.15
    do_sample: bool = True
