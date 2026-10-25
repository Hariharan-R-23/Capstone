"""
================================================================================
AI-Driven Lyric Generation: Top 5 Artists Inference & Multi-Objective Evaluation
================================================================================
Provides high-performance inference and live multi-objective reward scoring:
- Top 5 Artists: Elvis Presley, Willie Nelson, Ella Fitzgerald, Frank Sinatra, Dean Martin
- Instruction-Conditioned Generation: Artist, Mood, Genre, Theme, Section, Syllables, Rhyme
- Multi-Dimensional Evaluation:
  * Phonetic End-Rhyme Score (CMUdict)
  * Semantic Mood Alignment (Sentence-Transformers MiniLM)
  * Fluency & Repetition Index
  * Syllable Regularity & Meter Consistency
  * Composite Quality Index
- Audio Preview synthesis via gTTS
================================================================================
"""

import os
import re
import gc
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Multi-objective reward engine
try:
    import pronouncing
except ImportError:
    pronouncing = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TOP_5_ARTISTS_INFO = {
    "Elvis Presley": {
        "genre": "Rock & Roll / Rockabilly",
        "desc": "The King of Rock 'n' Roll. Charismatic rhythm, expressive baritone vibrato, dynamic tempo, passionate love and raw energy.",
        "era": "1950s - 1970s",
        "icon": "🎸",
        "sample_themes": [
            "Jailhouse Rock & High Energy Rebellion",
            "Can't Help Falling in Love (Tender Romance)",
            "Heartbreak Hotel (Midnight Melancholy)",
            "Suspicious Minds (Doubt and Passion)",
            "Burning Love (Fiery Euphoria)",
        ]
    },
    "Willie Nelson": {
        "genre": "Country & Outlaw Folk",
        "desc": "Outlaw country pioneer. Conversational storytelling, acoustic melancholy, laid-back cadence, weathered wisdom and rambler spirit.",
        "era": "1960s - Present",
        "icon": "🤠",
        "sample_themes": [
            "On the Road Again (The Rambling Spirit)",
            "Always on My Mind (Poignant Regret)",
            "Blue Eyes Crying in the Rain (Quiet Solitude)",
            "Whiskey River (Numbing the Sorrow)",
            "Crazy (Heartache and Longing)",
        ]
    },
    "Ella Fitzgerald": {
        "genre": "Vocal Jazz & Swing",
        "desc": "The First Lady of Song. Crystal-clear diction, swinging rhythm, playful and sophisticated phrasing, intricate melodic phrasing.",
        "era": "1930s - 1980s",
        "icon": "🎷",
        "sample_themes": [
            "Stardust Memories under the Manhattan Moon",
            "Summertime (Lush Ballad & Sultry Dreams)",
            "Cheek to Cheek (Effortless Swing Euphoria)",
            "Cry Me a River (Smoky Bittersweet Sarcasm)",
            "Autumn in New York (Sophisticated Melancholy)",
        ]
    },
    "Frank Sinatra": {
        "genre": "Traditional Pop & Big Band",
        "desc": "Ol' Blue Eyes. Master of narrative phrasing, brassy swagger, velvety dramatic baritone, urbane charisma, triumph and late-night sorrow.",
        "era": "1940s - 1990s",
        "icon": "🎙️",
        "sample_themes": [
            "My Way (Unapologetic Defiance & Legacy)",
            "Fly Me to the Moon (Romantic Big Band Euphoria)",
            "In the Wee Small Hours of the Morning (Solitary Grief)",
            "Theme from New York, New York (Triumphant Ambition)",
            "One for My Baby (Late-Night Barstool Reflection)",
        ]
    },
    "Dean Martin": {
        "genre": "Easy Listening & Crooner Pop",
        "desc": "The King of Cool. Warm crooning tone, effortless nonchalance, Mediterranean romantic charm, smooth relaxation and witty charm.",
        "era": "1940s - 1980s",
        "icon": "🍸",
        "sample_themes": [
            "That's Amore (Sunny Italian Romance)",
            "Everybody Loves Somebody (Warm Serendipity)",
            "Ain't That a Kick in the Head (Playful Fascination)",
            "Return to Me (Tender Mediterranean Longing)",
            "Memories Are Made of This (Nostalgic Comfort)",
        ]
    }
}


def count_syllables_word(word: str) -> int:
    """Estimate syllables in a word using CMUdict or vowel heuristics."""
    clean = re.sub(r"[^a-zA-Z]", "", word.lower())
    if not clean:
        return 0

    if pronouncing is not None:
        phones = pronouncing.phones_for_word(clean)
        if phones:
            return pronouncing.syllable_count(phones[0])

    # Fallback vowel cluster heuristic
    clean = re.sub(r"e$", "", clean)
    vowels = re.findall(r"[aeiouy]+", clean)
    return max(1, len(vowels))


def analyze_rhymes_and_meter(lyrics_text: str) -> Dict[str, Any]:
    """Analyze phonetic rhyme schemes and syllable distributions."""
    raw_lines = [l.strip() for l in lyrics_text.split("\n") if l.strip() and not l.startswith("[")]
    num_lines = len(raw_lines)
    if num_lines < 2:
        return {
            "rhyme_score": 0.0,
            "rhyme_density": 0.0,
            "identified_pairs": [],
            "avg_syllables": 0.0,
            "syllable_variance": 0.0,
            "num_lines": num_lines,
        }

    # Extract last words and count syllables per line
    end_words = []
    line_syllables = []
    for line in raw_lines:
        words = re.findall(r"\b[A-Za-z']+\b", line)
        if words:
            end_words.append(words[-1].lower())
            line_syllables.append(sum(count_syllables_word(w) for w in words))
        else:
            end_words.append("")
            line_syllables.append(0)

    rhyme_points = 0.0
    identified_pairs = []

    for i in range(len(end_words)):
        for offset, scheme in [(1, "Couplet (AABB)"), (2, "Alternating (ABAB)")]:
            j = i + offset
            if j < len(end_words):
                w1, w2 = end_words[i], end_words[j]
                if w1 and w2 and w1 != w2:
                    is_rhyme = False
                    if pronouncing is not None:
                        rhymes_w1 = set(pronouncing.rhymes(w1))
                        if w2 in rhymes_w1:
                            is_rhyme = True
                        else:
                            p1 = pronouncing.phones_for_word(w1)
                            p2 = pronouncing.phones_for_word(w2)
                            if p1 and p2:
                                r1 = {pronouncing.rhyming_part(p) for p in p1}
                                r2 = {pronouncing.rhyming_part(p) for p in p2}
                                if r1.intersection(r2):
                                    is_rhyme = True
                    # Heuristic fallback
                    if not is_rhyme and len(w1) >= 3 and len(w2) >= 3:
                        if w1[-3:] == w2[-3:] or w1[-2:] == w2[-2:]:
                            is_rhyme = True

                    if is_rhyme:
                        weight = 1.0 if offset == 1 else 0.85
                        rhyme_points += weight
                        pair_str = f"'{w1}' & '{w2}' (Line {i+1} & {j+1} - {scheme})"
                        if pair_str not in identified_pairs:
                            identified_pairs.append(pair_str)

    max_expected = max(1.0, (num_lines - 1) * 0.9)
    rhyme_score = min(1.0, rhyme_points / max_expected)

    avg_syllables = float(sum(line_syllables) / max(1, len(line_syllables)))
    syllable_var = float(sum((s - avg_syllables) ** 2 for s in line_syllables) / max(1, len(line_syllables)))

    return {
        "rhyme_score": float(rhyme_score),
        "rhyme_density": float(rhyme_points / max(1, num_lines)),
        "identified_pairs": identified_pairs,
        "avg_syllables": round(avg_syllables, 1),
        "syllable_variance": round(syllable_var, 2),
        "line_syllables": line_syllables,
        "num_lines": num_lines,
    }


class Top5LyricEngine:
    """
    Singleton lyric generation and evaluation engine designed for Streamlit.
    Loads Qwen 2.5 3B with Top 5 Artists LoRA adapter onto GPU in 4-bit NF4.
    """

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: Optional[str] = "./checkpoints/top5_artists_lyric_model/final_adapter",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.has_lora = False

        logger.info(f"Initializing Top5LyricEngine on device: {self.device}")

        # 1. Load Tokenizer
        tokenizer_src = adapter_path if (adapter_path and os.path.exists(adapter_path)) else base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"

        # 2. Load Base Model in 4-bit
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        logger.info(f"Loading Base LLM: {base_model_name} in 4-bit NF4...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map={"": 0} if torch.cuda.is_available() else None,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # 3. Attach LoRA Adapter if checkpoint exists
        if adapter_path and os.path.exists(adapter_path):
            logger.info(f"Attaching Top 5 Artists LoRA fine-tuned adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.has_lora = True
        else:
            logger.warning(f"Adapter not yet found at '{adapter_path}'. Operating in base model zero-shot mode.")

        self.model.eval()

        # 4. Load Mood Semantic Embedder
        if SentenceTransformer is not None:
            logger.info("Loading sentence-transformers/all-MiniLM-L6-v2 for mood scoring...")
            self.mood_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        else:
            self.mood_embedder = None

    def generate(
        self,
        artist: str = "Frank Sinatra",
        mood: str = "Midnight Melancholy & Resilient Hope",
        genre: str = "Traditional Pop & Big Band",
        theme: str = "In the Wee Small Hours of the Morning",
        section: str = "Full Song",
        rhyme_scheme: str = "AABB (Couplets)",
        max_new_tokens: int = 256,
        temperature: float = 0.75,
        top_p: float = 0.90,
        repetition_penalty: float = 1.15,
    ) -> Dict[str, Any]:
        """Generate artist-conditioned song lyrics and evaluate all dimensions."""
        # Standardized prompt structure
        prompt_header = (
            f"[Artist]: {artist.strip()} | [Mood]: {mood.strip()} | "
            f"[Genre]: {genre.strip()} | [Theme]: {theme.strip()}"
        )
        if section and section != "Full Song":
            prompt_header += f" | [Section]: {section}"
        
        full_prompt = f"{prompt_header}\n[Lyrics]:\n"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = output_tokens[0][input_len:]
        raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean and format generated lyrics
        clean_lines = []
        for line in raw_text.split("\n"):
            line_str = line.strip()
            # Stop if the model repeats conditioning prompt tags
            if line_str.startswith("[Artist]:") or line_str.startswith("[Genre]:"):
                break
            clean_lines.append(line_str)
        cleaned_lyrics = "\n".join(clean_lines).strip()

        # Multi-Objective Quality Evaluation
        # 1. Phonetic Rhyme & Meter
        rhyme_meter_metrics = analyze_rhymes_and_meter(cleaned_lyrics)
        rhyme_score = rhyme_meter_metrics["rhyme_score"]

        # 2. Mood Alignment Score
        mood_score = 0.5
        if self.mood_embedder is not None and util is not None and cleaned_lyrics:
            clean_lyric_text = re.sub(r"\[.*?\]", "", cleaned_lyrics).strip()
            if clean_lyric_text:
                target_vector = self.mood_embedder.encode(f"A song with {mood} mood", convert_to_tensor=True)
                lyrics_vector = self.mood_embedder.encode(clean_lyric_text, convert_to_tensor=True)
                cos_sim = util.cos_sim(target_vector, lyrics_vector).item()
                mood_score = float(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))

        # 3. Fluency & Repetition Score
        lines = [l for l in cleaned_lyrics.split("\n") if l.strip() and not l.startswith("[")]
        if lines:
            unique_ratio = len(set(lines)) / max(1, len(lines))
            fluency_score = min(1.0, max(0.0, unique_ratio))
        else:
            fluency_score = 0.0

        # 4. Composite Quality Score (0 - 100%)
        # 40% Rhyme, 40% Mood Alignment, 20% Fluency
        composite_score = (rhyme_score * 0.40) + (mood_score * 0.40) + (fluency_score * 0.20)

        return {
            "prompt": prompt_header,
            "lyrics": cleaned_lyrics,
            "artist": artist,
            "mood": mood,
            "genre": genre,
            "theme": theme,
            "section": section,
            "has_lora_adapter": self.has_lora,
            "metrics": {
                "composite_score": round(composite_score * 100, 1),
                "rhyme_score": round(rhyme_score * 100, 1),
                "mood_alignment": round(mood_score * 100, 1),
                "fluency_score": round(fluency_score * 100, 1),
                "num_lines": rhyme_meter_metrics["num_lines"],
                "avg_syllables": rhyme_meter_metrics["avg_syllables"],
                "syllable_variance": rhyme_meter_metrics["syllable_variance"],
                "identified_pairs": rhyme_meter_metrics["identified_pairs"],
            }
        }
