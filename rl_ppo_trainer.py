"""
================================================================================
AI-Driven Lyric Generation through Artistic Style Modeling and Context Adaptation
PPO Reinforcement Learning Policy Optimization (rl_ppo_trainer.py)
================================================================================
Step 4: Policy Optimization via Multi-Objective Reinforcement Learning

Responsibilities:
1. Load Base Model + SFT LoRA adapter.
2. Configure training for policy optimization.
3. Connect MultiObjectiveRewardEngine (Phonetic Rhyme + Mood Alignment + Repetition Penalty).
4. Run PPO/REINFORCE Training Loop:
   - Rollout: Sample conditioning prompts and generate candidate lyric stanzas.
   - Reward: Score candidate lyrics against target emotion and rhyme scheme.
   - Optimize: Update policy parameters using reward signals.
5. Log real-time metrics and export aligned lyric adapters.
================================================================================
"""

import os
import sys
import re
import argparse
import logging
from typing import Optional, List, Dict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

# Project imports
from config import ModelConfig, PPOConfig, RewardConfig
from dataset_loader import get_dataset_loader
from reward_engine import MultiObjectiveRewardEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_ppo_dataset(
    dataset_name: str,
    language: str = "english",
    max_samples: Optional[int] = 500,
    seed: int = 42
):
    """
    Load dataset and extract prompt conditioning queries for PPO rollouts.
    Each query is formatted as:
    `[Artist]: {artist} | [Mood]: {emotion} | [Genre]: {genre} | [Theme]: {theme}\n[Lyrics]:\n`
    """
    loader = get_dataset_loader(
        dataset_name_or_path=dataset_name,
        language=language,
        max_samples=max_samples,
        seed=seed
    )
    dataset_dict = loader.prepare_dataset()
    raw_train = dataset_dict["train"]

    def extract_ppo_prompt(example):
        text = example["text"]
        # Split at "[Lyrics]:" to get the prompt header
        if "[Lyrics]:" in text:
            prompt_header = text.split("[Lyrics]:")[0].strip()
        else:
            prompt_header = text.split("\n")[0].strip()
        
        prompt_query = f"{prompt_header}\n[Lyrics]:\n"
        
        # Extract mood from the prompt header: "[Artist]: X | [Mood]: Y | ..."
        mood = "Expressive"
        if "[Mood]:" in prompt_header:
            try:
                mood_part = prompt_header.split("[Mood]:")[1]
                mood = mood_part.split("|")[0].strip()
            except (IndexError, ValueError):
                pass
        
        # Extract artist
        artist = "Artist"
        if "[Artist]:" in prompt_header:
            try:
                artist_part = prompt_header.split("[Artist]:")[1]
                artist = artist_part.split("|")[0].strip()
            except (IndexError, ValueError):
                pass
        
        return {
            "query": prompt_query,
            "mood": mood,
            "artist": artist,
        }

    ppo_dataset = raw_train.map(extract_ppo_prompt, desc="Extracting PPO prompt queries")
    return ppo_dataset


def train_ppo(
    sft_model_path: str = "./checkpoints/sft_lyric_model/final_adapter",
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir: str = "./checkpoints/ppo_lyric_model",
    language: str = "english",
    total_steps: int = 200,
    batch_size: int = 8,
    mini_batch_size: int = 2,
    lr: float = 1.41e-5,
    reward_weights: Optional[Dict[str, float]] = None,
    log_with: str = "tensorboard",
    max_samples: Optional[int] = 500,
):
    """
    Executes RL-based policy optimization using a manual REINFORCE-style loop
    with the Multi-Objective Reward Engine (Rhyme + Mood + Fluency).
    
    Compatible with TRL 1.10+ where the PPO API requires neural reward/value models.
    Since our reward is rule-based (CMUdict rhyme + cosine mood + fluency), we use
    a manual optimization loop that applies the same gradient signal.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"--- [Step 4/4: PPO Training] Starting RL Alignment on device: {device} ---")

    # 1. Tokenizer Setup
    logger.info(f"Loading Tokenizer from: {sft_model_path if os.path.exists(sft_model_path) else base_model_name}")
    tokenizer_path = sft_model_path if os.path.exists(sft_model_path) else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    tokenizer.padding_side = "left"  # Generation rollouts require left padding

    # 2. Model Initialization with SFT LoRA adapters
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=compute_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Resize embeddings if tokenizer has extra tokens
    if len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    if os.path.exists(sft_model_path):
        logger.info(f"Attaching SFT LoRA adapters from '{sft_model_path}'...")
        model = PeftModel.from_pretrained(base_model, sft_model_path, is_trainable=True)
    else:
        logger.warning(f"SFT path '{sft_model_path}' not found. Using base model directly.")
        model = base_model

    model.train()

    # 3. Multi-Objective Reward Engine Setup
    logger.info("Initializing Multi-Objective Reward Engine...")
    r_cfg = RewardConfig()
    if reward_weights:
        r_cfg.rhyme_weight = reward_weights.get("rhyme", r_cfg.rhyme_weight)
        r_cfg.mood_weight = reward_weights.get("mood", r_cfg.mood_weight)
        r_cfg.fluency_weight = reward_weights.get("fluency", r_cfg.fluency_weight)

    reward_engine = MultiObjectiveRewardEngine(
        config=r_cfg,
        language=language,
        device=device
    )

    # 4. Prepare PPO Dataset
    logger.info("Preparing PPO rollout dataset...")
    ppo_dataset = build_ppo_dataset(
        dataset_name="manoh2f2/tsterbak-lyrics-dataset-with-emotions",
        language=language,
        max_samples=max_samples,
    )

    # Collect all prompts and moods
    prompts = ppo_dataset["query"]
    moods = ppo_dataset["mood"]

    # 5. Setup optimizer (only trainable params for LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 6. Generation parameters (kept small for 8GB VRAM)
    generation_kwargs = {
        "max_new_tokens": 128,
        "top_k": 50,
        "top_p": 0.92,
        "do_sample": True,
        "temperature": 0.8,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    # 7. REINFORCE-style RL Optimization Loop
    logger.info(f"--- Starting REINFORCE RL Loop for {total_steps} steps (batch_size={batch_size}) ---")
    
    all_rewards = []
    best_mean_reward = -float("inf")
    progress_bar = tqdm(total=total_steps, desc="RL Optimization Steps")

    for step in range(total_steps):
        # Sample a mini-batch of prompts
        indices = np.random.choice(len(prompts), size=min(batch_size, len(prompts)), replace=False)
        batch_prompts = [prompts[i] for i in indices]
        batch_moods = [moods[i] for i in indices]

        # Process one sample at a time to avoid OOM on 8GB VRAM,
        # then accumulate gradients across the mini-batch.
        all_sample_rewards = []
        all_sample_metrics = []
        optimizer.zero_grad()

        for sample_idx in range(len(batch_prompts)):
            single_prompt = batch_prompts[sample_idx]
            single_mood = batch_moods[sample_idx]

            # Tokenize single prompt
            inputs = tokenizer(
                single_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)
            prompt_length = inputs.input_ids.shape[1]

            # Generate lyrics
            model.eval()
            with torch.no_grad():
                gen_output = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            model.train()

            # Decode generated text
            gen_tokens = gen_output[0][prompt_length:]
            generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Compute multi-objective reward
            reward_val, metrics = reward_engine.evaluate_sample(single_mood, generated_text)
            all_sample_rewards.append(reward_val)
            all_sample_metrics.append(metrics)

            # Compute REINFORCE loss for this sample
            with torch.amp.autocast('cuda', dtype=compute_dtype):
                forward_outputs = model(
                    input_ids=gen_output,
                    attention_mask=(gen_output != tokenizer.pad_token_id).long(),
                )
                logits = forward_outputs.logits

                # Get log probs for generated tokens (after prompt)
                gen_logits = logits[:, prompt_length - 1:-1, :]
                gen_targets = gen_output[:, prompt_length:]

                log_probs = torch.nn.functional.log_softmax(gen_logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs, dim=-1, index=gen_targets.unsqueeze(-1)
                ).squeeze(-1)

                gen_mask = (gen_targets != tokenizer.pad_token_id).float()
                mean_log_prob = (token_log_probs * gen_mask).sum() / gen_mask.sum().clamp(min=1)

                # Scale loss by 1/batch_size for gradient accumulation
                sample_loss = -(reward_val * mean_log_prob) / len(batch_prompts)

            sample_loss.backward()

            # Free memory after each sample
            del forward_outputs, logits, gen_logits, log_probs, gen_output, inputs
            torch.cuda.empty_cache()

        # Apply accumulated gradients
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        rewards_list = all_sample_rewards
        metrics_list = all_sample_metrics
        mean_reward = float(np.mean(rewards_list))
        mean_rhyme = float(np.mean([m["rhyme_score"] for m in metrics_list]))
        mean_mood = float(np.mean([m["mood_alignment_score"] for m in metrics_list]))
        mean_fluency = float(np.mean([m["fluency_score"] for m in metrics_list]))
        all_rewards.append(mean_reward)

        progress_bar.update(1)
        progress_bar.set_postfix({
            "Reward": f"{mean_reward:.4f}",
            "Rhyme": f"{mean_rhyme:.4f}",
            "Mood": f"{mean_mood:.4f}",
        })

        if (step + 1) % 10 == 0:
            logger.info(
                f"Step {step + 1}/{total_steps} | "
                f"Reward: {mean_reward:.4f} | Rhyme: {mean_rhyme:.4f} | "
                f"Mood: {mean_mood:.4f} | Fluency: {mean_fluency:.4f}"
            )

        # Checkpoint intermediate models
        if (step + 1) % 50 == 0:
            save_ckpt_dir = os.path.join(output_dir, f"checkpoint_step_{step + 1}")
            logger.info(f"Saving intermediate checkpoint to '{save_ckpt_dir}'...")
            model.save_pretrained(save_ckpt_dir)
            tokenizer.save_pretrained(save_ckpt_dir)

        # Track best model
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward

    progress_bar.close()

    # 8. Save Final Model Artifacts
    final_ppo_path = os.path.join(output_dir, "final_ppo_model")
    logger.info(f"Saving final RL-aligned model to '{final_ppo_path}'...")
    model.save_pretrained(final_ppo_path)
    tokenizer.save_pretrained(final_ppo_path)

    logger.info(f"RL Training Complete! Best Mean Reward: {best_mean_reward:.4f}")
    logger.info(f"Average Reward (last 10 steps): {np.mean(all_rewards[-10:]):.4f}")
    return final_ppo_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Objective RL Policy Optimization for Controllable Lyric Generation"
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="./checkpoints/sft_lyric_model/final_adapter",
        help="Path to SFT LoRA checkpoint directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base LLM identifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/ppo_lyric_model",
        help="Output directory for RL checkpoints",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        choices=["english", "tamil"],
        help="Language phase (english or tamil)",
    )
    parser.add_argument("--steps", type=int, default=200, help="Total RL optimization steps")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.41e-5)
    parser.add_argument("--rhyme_weight", type=float, default=0.45)
    parser.add_argument("--mood_weight", type=float, default=0.40)
    parser.add_argument("--fluency_weight", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=500, help="Limit PPO dataset samples")
    parser.add_argument("--log_with", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_ppo(
        sft_model_path=args.sft_model_path,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        language=args.language,
        total_steps=args.steps,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        lr=args.lr,
        reward_weights={
            "rhyme": args.rhyme_weight,
            "mood": args.mood_weight,
            "fluency": args.fluency_weight,
        },
        max_samples=args.max_samples,
        log_with=args.log_with,
    )
