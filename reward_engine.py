"""
================================================================================
AI-Driven Lyric Generation through Artistic Style Modeling and Context Adaptation
Multi-Objective Reward Engine (reward_engine.py)
================================================================================
Step 3: Multi-Objective Reward Modeling

Responsibilities:
1. BaseRhymeEvaluator: Abstract interface for phonetic rhyme analysis.
2. EnglishRhymeEvaluator: Uses CMU pronouncing dictionary to evaluate end-of-line
   rhyme scheme (AABB, ABAB, AAAA), slant rhymes, and rhyme density.
3. TamilRhymeEvaluator: Phase 2 extensible evaluator modeling Tamil poetic meter:
   - Monai (மோனை): Initial consonant/syllable matching.
   - Etukai (எதுகை): Second letter/phoneme harmony.
   - Iyalbu (இயைபு): End-line vowel/syllable resonance.
4. MoodAlignmentReward: Computes cosine similarity between generated lyrics and
   target emotion vector using `sentence-transformers` (`all-MiniLM-L6-v2`).
5. FluencyAndRepetitionPenalty: Penalizes degenerate loops and unnatural line lengths.
6. MultiObjectiveRewardEngine: Combines multi-objective signals into a calibrated
   scalar reward for TRL PPOTrainer.
================================================================================
"""

import abc
import re
import string
import logging
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import torch

try:
    import pronouncing
except ImportError:
    pronouncing = None

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

from config import RewardConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# Base Rhyme Evaluator (Abstract Interface)
# ==============================================================================
class BaseRhymeEvaluator(abc.ABC):
    """Abstract interface defining the contract for phonetic rhyme and poetic meter evaluation."""

    @abc.abstractmethod
    def evaluate_rhyme(self, lyrics_text: str) -> Dict[str, float]:
        """
        Evaluate rhyme structure of the lyrics.
        Returns a dictionary containing:
        - "rhyme_score": float in [0.0, 1.0]
        - "rhyme_density": float in [0.0, 1.0]
        - "num_rhyming_pairs": int
        - "num_lines": int
        """
        pass


# ==============================================================================
# Phase 1: English Phonetic Rhyme Evaluator (CMU Pronouncing Dictionary)
# ==============================================================================
class EnglishRhymeEvaluator(BaseRhymeEvaluator):
    """
    Evaluates English phonetic end-rhymes using CMU Pronouncing dictionary (`pronouncing`).
    Analyzes:
    - Couplet rhymes (Line i with Line i+1 -> AABB)
    - Cross / Alternating rhymes (Line i with Line i+2 -> ABAB)
    - Envelope rhymes (Line i with Line i+3 -> ABBA)
    - Penalizes identity rhymes (same word rhymed with itself).
    """

    def __init__(self):
        if pronouncing is None:
            logger.warning("`pronouncing` package not installed. Rhyme evaluation will use fallback heuristic.")

    @staticmethod
    def extract_line_end_words(lyrics_text: str) -> List[str]:
        """Extract the last alphanumeric word of each non-empty lyric line."""
        lines = lyrics_text.strip().split("\n")
        end_words = []
        for line in lines:
            line_clean = line.strip()
            # Ignore section headers like [Verse], [Chorus]
            if line_clean.startswith("[") and line_clean.endswith("]"):
                continue
            # Remove punctuation and extract trailing word
            words = re.findall(r"\b[A-Za-z']+\b", line_clean)
            if words:
                end_words.append(words[-1].lower())
        return end_words

    def check_words_rhyme(self, word_a: str, word_b: str) -> Tuple[bool, bool]:
        """
        Check if two words rhyme.
        Returns: (is_rhyme, is_identity)
        """
        if not word_a or not word_b:
            return False, False

        # Identity check (same word is not a true poetic rhyme)
        if word_a == word_b:
            return True, True

        if pronouncing is not None:
            phones_a = pronouncing.phones_for_word(word_a)
            phones_b = pronouncing.phones_for_word(word_b)

            if phones_a and phones_b:
                rhymes_a = set(pronouncing.rhymes(word_a))
                if word_b in rhymes_a:
                    return True, False

                # Slant rhyme / matching last phone check
                rhyming_parts_a = {pronouncing.rhyming_part(p) for p in phones_a}
                rhyming_parts_b = {pronouncing.rhyming_part(p) for p in phones_b}
                if rhyming_parts_a.intersection(rhyming_parts_b):
                    return True, False

        # Fallback suffix heuristic if pronouncing dictionary misses slang/contractions
        if len(word_a) >= 3 and len(word_b) >= 3:
            if word_a[-3:] == word_b[-3:] or word_a[-2:] == word_b[-2:]:
                return True, False

        return False, False

    def evaluate_rhyme(self, lyrics_text: str) -> Dict[str, float]:
        end_words = self.extract_line_end_words(lyrics_text)
        num_lines = len(end_words)

        if num_lines < 2:
            return {
                "rhyme_score": 0.0,
                "rhyme_density": 0.0,
                "num_rhyming_pairs": 0,
                "num_lines": num_lines,
            }

        rhyme_points = 0.0
        potential_pairs = 0
        identity_penalty = 0.0

        # Check Couplet (i, i+1), Alternating (i, i+2), and Triplet/Envelope (i, i+3)
        for i in range(num_lines):
            for offset, weight in [(1, 1.0), (2, 0.8), (3, 0.5)]:
                j = i + offset
                if j < num_lines:
                    potential_pairs += 1
                    is_rhyme, is_identity = self.check_words_rhyme(end_words[i], end_words[j])
                    if is_rhyme:
                        if is_identity:
                            # Diminished reward for identical repeated words
                            rhyme_points += 0.2 * weight
                            identity_penalty += 0.1
                        else:
                            rhyme_points += 1.0 * weight

        # Normalize score
        max_expected_points = max(1.0, (num_lines - 1) * 1.2)
        raw_score = min(1.0, rhyme_points / max_expected_points)
        final_score = max(0.0, raw_score - (identity_penalty * 0.2))

        return {
            "rhyme_score": float(np.clip(final_score, 0.0, 1.0)),
            "rhyme_density": float(rhyme_points / max(1, potential_pairs)),
            "num_rhyming_pairs": int(rhyme_points),
            "num_lines": num_lines,
        }


# ==============================================================================
# Phase 2: Tamil Poetic Meter & Rhyme Evaluator (Monai / Etukai / Iyalbu)
# ==============================================================================
class TamilRhymeEvaluator(BaseRhymeEvaluator):
    """
    Phase 2 Tamil Poetic Analyzer:
    Evaluates classical Tamil poetic prosody (யாப்பிலக்கணம்):
    1. மோனை (Monai): Initial letter / consonant harmony across lines.
    2. எதுகை (Etukai): Second letter / consonant harmony across lines.
    3. இயைபு (Iyalbu): End-of-line vowel and syllable resonance.
    """

    def __init__(self):
        # Tamil vowel harmony classes for slant matching
        self.uyir_groups = {
            "a": ["அ", "ஆ", "ஐ", "ஔ"],
            "i": ["இ", "ஈ", "எ", "ஏ"],
            "u": ["உ", "ஊ", "ஒ", "ஓ"],
        }

    @staticmethod
    def get_first_letter(word: str) -> str:
        return word[:1] if word else ""

    @staticmethod
    def get_second_letter(word: str) -> str:
        return word[1:2] if len(word) > 1 else ""

    def evaluate_rhyme(self, lyrics_text: str) -> Dict[str, float]:
        lines = [l.strip() for l in lyrics_text.split("\n") if l.strip() and not l.startswith("[")]
        num_lines = len(lines)

        if num_lines < 2:
            return {
                "rhyme_score": 0.0,
                "monai_score": 0.0,
                "etukai_score": 0.0,
                "num_lines": num_lines,
            }

        monai_matches = 0
        etukai_matches = 0
        total_couplets = num_lines - 1

        for i in range(total_couplets):
            words_curr = lines[i].split()
            words_next = lines[i + 1].split()

            if words_curr and words_next:
                # Monai (1st letter match)
                if self.get_first_letter(words_curr[0]) == self.get_first_letter(words_next[0]):
                    monai_matches += 1
                # Etukai (2nd letter match - hallmark of Tamil poetic rhyme)
                if self.get_second_letter(words_curr[0]) == self.get_second_letter(words_next[0]):
                    etukai_matches += 1

        monai_score = monai_matches / max(1, total_couplets)
        etukai_score = etukai_matches / max(1, total_couplets)
        
        # Etukai holds 60% weight, Monai holds 40%
        composite_score = (etukai_score * 0.6) + (monai_score * 0.4)

        return {
            "rhyme_score": float(np.clip(composite_score, 0.0, 1.0)),
            "monai_score": float(monai_score),
            "etukai_score": float(etukai_score),
            "num_lines": num_lines,
        }


# ==============================================================================
# Mood / Emotion Alignment Reward Engine
# ==============================================================================
class MoodAlignmentReward:
    """
    Computes semantic similarity between target mood/emotion prompt and generated lyrics.
    Uses sentence-transformers to encode semantic intent into high-dimensional vector space.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if SentenceTransformer is not None:
            logger.info(f"Loading Mood Semantic Alignment model: {model_name} on {self.device}")
            self.model = SentenceTransformer(model_name, device=self.device)
        else:
            logger.warning("`sentence-transformers` not available. Mood alignment using fallback keyword matching.")
            self.model = None

    def compute_mood_similarity(self, target_mood: str, lyrics_text: str) -> float:
        """
        Calculates cosine similarity between target emotion prompt and generated song text.
        Returns float in range [0.0, 1.0].
        """
        if not target_mood or not lyrics_text.strip():
            return 0.0

        if self.model is not None and util is not None:
            # Clean section markers for embedding calculation
            clean_lyrics = re.sub(r"\[.*?\]", "", lyrics_text).strip()
            if not clean_lyrics:
                return 0.0

            # Construct contextual mood descriptor
            mood_prompt = f"Song with emotion and mood of {target_mood}"
            
            with torch.no_grad():
                emb_mood = self.model.encode(mood_prompt, convert_to_tensor=True)
                emb_lyrics = self.model.encode(clean_lyrics, convert_to_tensor=True)
                cosine_sim = util.cos_sim(emb_mood, emb_lyrics).item()

            # Rescale cosine similarity from [-1, 1] to [0, 1]
            normalized_score = (cosine_sim + 1.0) / 2.0
            return float(np.clip(normalized_score, 0.0, 1.0))
        else:
            # Fallback heuristic
            mood_keywords = set(target_mood.lower().split())
            lyrics_words = set(re.findall(r"\w+", lyrics_text.lower()))
            overlap = len(mood_keywords.intersection(lyrics_words))
            return min(1.0, overlap / max(1, len(mood_keywords)))


# ==============================================================================
# Fluency & Repetition Penalty Engine
# ==============================================================================
class FluencyAndRepetitionPenalty:
    """
    Penalizes common degenerate language model failure modes:
    1. Repetitive line loops (e.g. repeating the same sentence 4 times).
    2. High 3-gram / 4-gram repetition.
    3. Abnormal line lengths (too short < 2 words or run-on sentences > 20 words).
    """

    @staticmethod
    def compute_penalty(lyrics_text: str) -> Dict[str, float]:
        lines = [l.strip() for l in lyrics_text.split("\n") if l.strip() and not l.startswith("[")]
        if not lines:
            return {"repetition_penalty": 1.0, "fluency_score": 0.0}

        # 1. Line-level repetition
        unique_lines = set(lines)
        line_rep_ratio = 1.0 - (len(unique_lines) / max(1, len(lines)))

        # 2. N-gram repetition (3-grams)
        words = re.findall(r"\w+", lyrics_text.lower())
        if len(words) >= 6:
            tri_grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
            unique_trigrams = set(tri_grams)
            trigram_rep_ratio = 1.0 - (len(unique_trigrams) / max(1, len(tri_grams)))
        else:
            trigram_rep_ratio = 0.0

        # 3. Line length variance penalty
        line_lengths = [len(l.split()) for l in lines]
        too_short = sum(1 for l in line_lengths if l < 2) / max(1, len(lines))
        too_long = sum(1 for l in line_lengths if l > 20) / max(1, len(lines))

        total_penalty = min(
            1.0,
            (line_rep_ratio * 0.5) + (trigram_rep_ratio * 0.3) + (too_short * 0.1) + (too_long * 0.1)
        )
        fluency_score = 1.0 - total_penalty

        return {
            "repetition_penalty": float(total_penalty),
            "fluency_score": float(max(0.0, fluency_score)),
        }


# ==============================================================================
# Multi-Objective Reward Engine (Composite Coordinator)
# ==============================================================================
class MultiObjectiveRewardEngine:
    """
    Coordinates and computes composite scalar reward for TRL PPOTrainer:
    R = w_rhyme * R_rhyme + w_mood * R_mood + w_fluency * R_fluency
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        language: str = "english",
        device: Optional[str] = None
    ):
        self.cfg = config or RewardConfig()
        self.language = language.lower()

        # Initialize Rhyme Evaluator (Language-Aware)
        if self.language in ["tamil", "ta"]:
            logger.info("Initializing Tamil Poetic Prosody (Monai/Etukai) Evaluator for Phase 2...")
            self.rhyme_evaluator: BaseRhymeEvaluator = TamilRhymeEvaluator()
        else:
            logger.info("Initializing English Phonetic CMUdict Rhyme Evaluator for Phase 1...")
            self.rhyme_evaluator: BaseRhymeEvaluator = EnglishRhymeEvaluator()

        # Initialize Mood Evaluator
        self.mood_evaluator = MoodAlignmentReward(
            model_name=self.cfg.mood_model_name,
            device=device
        )

        # Initialize Fluency & Repetition Evaluator
        self.fluency_evaluator = FluencyAndRepetitionPenalty()

    def evaluate_sample(self, target_mood: str, lyrics_text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate a single generated song text against target emotional prompt.
        Returns:
        - composite_reward: scalar float for PPO optimization.
        - metrics_dict: detailed breakdown for logging and visualization.
        """
        # 1. Phonetic Rhyme Evaluation
        rhyme_metrics = self.rhyme_evaluator.evaluate_rhyme(lyrics_text)
        rhyme_score = rhyme_metrics["rhyme_score"]

        # 2. Mood Alignment Evaluation
        mood_score = self.mood_evaluator.compute_mood_similarity(target_mood, lyrics_text)

        # 3. Fluency & Repetition Evaluation
        fluency_metrics = self.fluency_evaluator.compute_penalty(lyrics_text)
        fluency_score = fluency_metrics["fluency_score"]

        # 4. Composite Weighted Score
        composite_reward = (
            (self.cfg.rhyme_weight * rhyme_score * self.cfg.rhyme_scale)
            + (self.cfg.mood_weight * mood_score * self.cfg.mood_scale)
            + (self.cfg.fluency_weight * fluency_score)
        )

        metrics = {
            "composite_reward": float(composite_reward),
            "rhyme_score": float(rhyme_score),
            "mood_alignment_score": float(mood_score),
            "fluency_score": float(fluency_score),
            "repetition_penalty": float(fluency_metrics["repetition_penalty"]),
            "num_lines": rhyme_metrics.get("num_lines", 0),
        }

        return float(composite_reward), metrics

    def batch_evaluate(
        self,
        target_moods: List[str],
        lyrics_texts: List[str]
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Evaluate a batch of rollout samples during PPO training loop."""
        rewards = []
        metrics_list = []
        for mood, lyrics in zip(target_moods, lyrics_texts):
            r, m = self.evaluate_sample(mood, lyrics)
            rewards.append(r)
            metrics_list.append(m)
        return rewards, metrics_list


if __name__ == "__main__":
    print("--- Multi-Objective Reward Engine Verification ---")
    engine = MultiObjectiveRewardEngine(language="english")

    test_prompt_mood = "Melancholic Grief and Longing"
    sample_lyrics_rhyming = (
        "[Verse 1]\n"
        "The shadows fall across the empty floor\n"
        "I hear your footsteps walking out the door\n"
        "The winter wind is blowing cold and deep\n"
        "These broken promises I couldn't keep\n\n"
        "[Chorus]\n"
        "And now the night is turning into gray\n"
        "I watch the memories slowly fade away"
    )

    sample_lyrics_poor = (
        "Hello hello hello hello\n"
        "The sky is blue today.\n"
        "I eat an apple on the table.\n"
        "Computer machine running."
    )

    reward_good, metrics_good = engine.evaluate_sample(test_prompt_mood, sample_lyrics_rhyming)
    reward_bad, metrics_bad = engine.evaluate_sample(test_prompt_mood, sample_lyrics_poor)

    print("\n--- High Quality Rhyming Lyric Evaluation ---")
    print(f"Composite Reward: {reward_good:.4f}")
    for k, v in metrics_good.items():
        print(f"  {k}: {v}")

    print("\n--- Poor / Repetitive Lyric Evaluation ---")
    print(f"Composite Reward: {reward_bad:.4f}")
    for k, v in metrics_bad.items():
        print(f"  {k}: {v}")
