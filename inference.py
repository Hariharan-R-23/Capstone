"""
================================================================================
AI-Driven Lyric Generation through Artistic Style Modeling and Context Adaptation
Inference & Evaluation Module (inference.py)
================================================================================
Provides interactive and batch lyric generation with style conditioning:
- Conditioning inputs: [Artist], [Mood], [Genre], [Theme]
- Model Support: Base Model, SFT LoRA Adapters, or PPO Fine-Tuned Checkpoints
- Automated live evaluation via MultiObjectiveRewardEngine (Rhyme & Mood score)
================================================================================
"""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from config import InferenceConfig, ModelConfig, RewardConfig
from reward_engine import MultiObjectiveRewardEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LyricGenerator:
    """
    Inference generator supporting Base, SFT, and PPO models with live multi-objective reward scoring.
    """

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        adapter_path: Optional[str] = None,
        language: str = "english",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        logger.info(f"Loading generator on device: {self.device}")

        # 1. Load Tokenizer
        tokenizer_src = adapter_path if (adapter_path and os.path.exists(adapter_path)) else model_name_or_path
        logger.info(f"Loading Tokenizer from: {tokenizer_src}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|pad|>"

        # 2. Load Model
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Loading base model: {model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # 3. Attach LoRA Adapter if specified
        if adapter_path and os.path.exists(adapter_path):
            logger.info(f"Attaching LoRA fine-tuned adapters from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

        # 4. Initialize Multi-Objective Reward Engine for instant quality inspection
        self.reward_engine = MultiObjectiveRewardEngine(
            language=self.language,
            device=self.device
        )

    def generate_lyrics(
        self,
        artist: str = "Taylor Swift",
        mood: str = "Nostalgic Melancholy",
        genre: str = "Indie Pop",
        theme: str = "Lost Love and Autumn Memories",
        max_new_tokens: int = 384,
        temperature: float = 0.75,
        top_p: float = 0.90,
        top_k: int = 50,
        repetition_penalty: float = 1.15,
        evaluate_quality: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate song lyrics conditioned on artistic attributes.
        """
        # Construct standardized conditioning prompt
        prompt = (
            f"[Artist]: {artist.strip()} | [Mood]: {mood.strip()} | "
            f"[Genre]: {genre.strip()} | [Theme]: {theme.strip()}\n[Lyrics]:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated completion
        generated_tokens = outputs[0][input_length:]
        lyrics_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        result = {
            "prompt": prompt.strip(),
            "artist": artist,
            "mood": mood,
            "genre": genre,
            "theme": theme,
            "lyrics": lyrics_text,
        }

        # Calculate reward scores if requested
        if evaluate_quality:
            reward_score, metrics = self.reward_engine.evaluate_sample(mood, lyrics_text)
            result["reward_score"] = reward_score
            result["metrics"] = metrics

        return result


def interactive_cli(generator: LyricGenerator):
    """Interactive command-line interface for human prompt input and instant generation."""
    print("\n" + "=" * 80)
    print("AI-Driven Artistic Lyric Generator (Interactive Studio)")
    print("Type 'exit' or 'quit' to terminate session.")
    print("=" * 80 + "\n")

    while True:
        try:
            print("\nEnter Song Context Details (or press Enter for default):")
            artist = input("  [Artist] (default: Freddie Mercury / Queen): ").strip() or "Freddie Mercury"
            if artist.lower() in ["exit", "quit"]:
                break

            mood = input("  [Mood]   (default: Operatic Passion & Triumph): ").strip() or "Operatic Passion & Triumph"
            genre = input("  [Genre]  (default: Classic Rock): ").strip() or "Classic Rock"
            theme = input("  [Theme]  (default: Defying Destiny & Freedom): ").strip() or "Defying Destiny & Freedom"

            print("\nGenerating song lyrics, please wait...")
            output = generator.generate_lyrics(
                artist=artist,
                mood=mood,
                genre=genre,
                theme=theme,
            )

            print("\n" + "-" * 60)
            print("GENERATED SONG LYRICS:")
            print("-" * 60)
            print(output["lyrics"])
            print("-" * 60)

            if "metrics" in output:
                print("LIVE QUALITY SCORES (Reward Engine):")
                m = output["metrics"]
                print(f"  Composite Reward Score : {output['reward_score']:.4f}")
                print(f"  Phonetic Rhyme Score  : {m.get('rhyme_score', 0.0):.4f}")
                print(f"  Mood Alignment Score  : {m.get('mood_alignment_score', 0.0):.4f}")
                print(f"  Fluency & Flow Score  : {m.get('fluency_score', 0.0):.4f}")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting session...")
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AI-driven song lyrics with style conditioning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA SFT or PPO checkpoint")
    parser.add_argument("--language", type=str, default="english", choices=["english", "tamil"])
    parser.add_argument("--artist", type=str, default="Taylor Swift")
    parser.add_argument("--mood", type=str, default="Nostalgic Melancholy")
    parser.add_argument("--genre", type=str, default="Indie Folk")
    parser.add_argument("--theme", type=str, default="Midnight Reflections and Autumn Rain")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen = LyricGenerator(
        model_name_or_path=args.model_name,
        adapter_path=args.adapter_path,
        language=args.language,
    )

    if args.interactive:
        interactive_cli(gen)
    else:
        out = gen.generate_lyrics(
            artist=args.artist,
            mood=args.mood,
            genre=args.genre,
            theme=args.theme,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n--- GENERATED LYRICS ---")
        print(f"Conditioning: {out['prompt']}\n")
        print(out["lyrics"])
        if "metrics" in out:
            print("\n--- EVALUATION METRICS ---")
            for k, v in out["metrics"].items():
                print(f"{k}: {v}")
