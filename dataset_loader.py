"""
================================================================================
AI-Driven Lyric Generation through Artistic Style Modeling and Context Adaptation
Dataset Loader Module (dataset_loader.py)
================================================================================
Step 1: Data Ingestion & Prompt Conditioning

Responsibilities:
1. Ingest baseline English lyric dataset with multi-label emotional annotations:
   HF Dataset: `manoh2f2/tsterbak-lyrics-dataset-with-emotions`
2. Construct standardized conditioning prompt format:
   `[Artist]: {artist} | [Mood]: {emotion} | [Genre]: {genre} | [Theme]: {theme}\n[Lyrics]:\n{lyrics}`
3. Provide modular BaseLyricDatasetLoader interface.
4. Provide EnglishLyricDatasetLoader (Phase 1 baseline).
5. Provide TamilLyricDatasetLoader (Phase 2 regional dataset placeholder e.g., `abishekmahi/tamil-kavithai`).
================================================================================
"""

import abc
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaseLyricDatasetLoader(abc.ABC):
    """
    Abstract Base Class for multi-lingual and multi-artist lyric dataset loaders.
    Enforces a standardized contract across Phase 1 (English) and Phase 2 (Tamil/Regional).
    """

    def __init__(
        self,
        dataset_name_or_path: str,
        split_ratio: float = 0.9,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        self.dataset_name_or_path = dataset_name_or_path
        self.split_ratio = split_ratio
        self.max_samples = max_samples
        self.seed = seed
        self.raw_dataset = None
        self.processed_dataset = None

    @abc.abstractmethod
    def load_raw_data(self) -> DatasetDict:
        """Load the raw dataset from Hugging Face Hub, local disk, or API."""
        pass

    @abc.abstractmethod
    def extract_metadata(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract standardized metadata dictionary:
        {
            "artist": str,
            "mood": str,
            "genre": str,
            "theme": str,
            "lyrics": str
        }
        """
        pass

    def clean_lyrics(self, lyrics_text: str) -> str:
        """
        Clean and normalize raw lyric strings:
        - Removes website watermarks, embed tags, and excessive blank lines.
        - Standardizes section tags (e.g., [Verse], [Chorus]).
        """
        if not isinstance(lyrics_text, str) or not lyrics_text.strip():
            return ""

        # Normalize line breaks and XML/Pandas export artifacts like _x000D_
        text = lyrics_text.replace("_x000D_", "").replace("\r", "")

        # Remove header noise like '123 Contributors', 'Translations', 'Embed', etc.
        text = re.sub(r"^\d+\s+Contributors?.*?\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\d*Embed$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[.*?See .*? Live.*?\]", "", text, flags=re.IGNORECASE)
        
        # Normalize whitespace while preserving stanza line breaks
        lines = [line.strip() for line in text.split("\n")]
        # Collapse multiple empty lines into a single blank line (stanza separator)
        cleaned_lines = []
        consecutive_empty = 0
        for line in lines:
            if not line:
                consecutive_empty += 1
                if consecutive_empty <= 1 and cleaned_lines:
                    cleaned_lines.append("")
            else:
                consecutive_empty = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def format_prompt(artist: str, mood: str, genre: str, theme: str) -> str:
        """
        Format standardized conditioning prompt header:
        `[Artist]: {artist} | [Mood]: {emotion} | [Genre]: {genre} | [Theme]: {theme}`
        """
        artist_clean = artist.strip() if artist else "Unknown Artist"
        mood_clean = mood.strip() if mood else "Expressive"
        genre_clean = genre.strip() if genre else "Contemporary"
        theme_clean = theme.strip() if theme else "Life & Passion"

        return f"[Artist]: {artist_clean} | [Mood]: {mood_clean} | [Genre]: {genre_clean} | [Theme]: {theme_clean}"

    def build_sft_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Construct prompt and full formatted text for Supervised Fine-Tuning."""
        meta = self.extract_metadata(example)
        cleaned_lyrics = self.clean_lyrics(meta["lyrics"])
        prompt_header = self.format_prompt(
            artist=meta["artist"],
            mood=meta["mood"],
            genre=meta["genre"],
            theme=meta["theme"]
        )

        # Full text formatted with special tokens
        completion_text = f"\n[Lyrics]:\n{cleaned_lyrics}"
        full_text = f"{prompt_header}{completion_text}"
        
        return {
            "prompt": prompt_header,
            "completion": completion_text,
            "lyrics": cleaned_lyrics,
            "text": full_text,
            "artist": meta["artist"],
            "mood": meta["mood"],
            "genre": meta["genre"],
            "theme": meta["theme"],
        }

    def prepare_dataset(self) -> DatasetDict:
        """Pipeline execution: Load -> Extract -> Clean -> Format -> Split."""
        logger.info(f"Loading dataset from '{self.dataset_name_or_path}'...")
        raw = self.load_raw_data()

        # Handle split structure
        if isinstance(raw, DatasetDict):
            if "train" in raw:
                dataset_to_process = raw["train"]
            else:
                dataset_to_process = next(iter(raw.values()))
        else:
            dataset_to_process = raw

        if self.max_samples and len(dataset_to_process) > self.max_samples:
            dataset_to_process = dataset_to_process.select(range(self.max_samples))

        logger.info(f"Processing and structuring {len(dataset_to_process)} raw examples...")
        
        # Apply transformation
        processed = dataset_to_process.map(
            self.build_sft_example,
            remove_columns=dataset_to_process.column_names,
            load_from_cache_file=False,
            desc="Formatting lyric prompts",
        )

        # Filter out invalid or overly short samples (less than 4 lines or < 30 chars)
        def is_valid_sample(ex):
            lyrics = ex.get("lyrics", "")
            return len(lyrics) >= 30 and len(lyrics.split("\n")) >= 4

        processed = processed.filter(is_valid_sample, desc="Filtering invalid lyrics")
        logger.info(f"Retained {len(processed)} high-quality lyric examples after filtering.")

        # Ensure only the 'text' column is kept for SFTTrainer causal language modeling
        processed = processed.select_columns(["text"])

        # Train / Validation Split
        split_dict = processed.train_test_split(
            train_size=self.split_ratio,
            seed=self.seed,
            shuffle=True
        )

        self.processed_dataset = DatasetDict({
            "train": split_dict["train"],
            "validation": split_dict["test"]
        })

        return self.processed_dataset


# ==============================================================================
# PHASE 1: English Lyric Dataset Loader
# ==============================================================================
class EnglishLyricDatasetLoader(BaseLyricDatasetLoader):
    """
    Concrete loader for Phase 1 English Lyric Dataset:
    `manoh2f2/tsterbak-lyrics-dataset-with-emotions`
    
    Expected raw columns:
    - artist (or Artist / author)
    - lyrics (or Lyrics / text)
    - emotion (or mood / feelings / multi-label emotions)
    - genre (or style)
    - theme (or topic / tag)
    """

    def load_raw_data(self) -> Any:
        try:
            return load_dataset(self.dataset_name_or_path)
        except Exception as e:
            logger.warning(
                f"Could not download directly from Hugging Face Hub ({e}). "
                f"Generating high-quality synthetic fallback samples for offline dry-run."
            )
            return self._create_synthetic_fallback()

    def extract_metadata(self, example: Dict[str, Any]) -> Dict[str, str]:
        # Artist extraction with fallback keys
        artist = (
            example.get("artist")
            or example.get("Artist")
            or example.get("singer")
            or example.get("author")
            or "Unknown Artist"
        )

        # Emotion / Mood extraction
        mood = (
            example.get("emotions")
            or example.get("emotion")
            or example.get("Emotion")
            or example.get("mood")
            or example.get("feelings")
            or "Nostalgic Melancholy"
        )

        # Genre extraction
        genre = (
            example.get("genre")
            or example.get("Genre")
            or example.get("style")
            or "Contemporary"
        )

        # Theme / Song extraction
        theme = (
            example.get("theme")
            or example.get("Theme")
            or example.get("topic")
            or example.get("song")
            or "Life & Passion"
        )

        # Lyrics extraction (supports 'seq', 'lyrics', 'Lyrics', 'text', 'content')
        lyrics = (
            example.get("seq")
            or example.get("lyrics")
            or example.get("Lyrics")
            or example.get("text")
            or example.get("content")
            or ""
        )

        return {
            "artist": str(artist),
            "mood": str(mood),
            "genre": str(genre),
            "theme": str(theme),
            "lyrics": str(lyrics),
        }

    def _create_synthetic_fallback(self) -> DatasetDict:
        """Generates synthetic multi-artist, multi-emotion lyric examples for dry-runs."""
        sample_records = [
            {
                "artist": "Adele",
                "emotion": "Melancholic Grief",
                "genre": "Soul Pop",
                "theme": "Lost Love and Reflection",
                "lyrics": (
                    "[Verse 1]\n"
                    "I heard that you're settled down\n"
                    "That you found a girl and you're married now\n"
                    "I heard that your dreams came true\n"
                    "Guess she gave you things I didn't give to you\n\n"
                    "[Chorus]\n"
                    "Never mind, I'll find someone like you\n"
                    "I wish nothing but the best for you too\n"
                    "Don't forget me, I beg, I remember you said\n"
                    "Sometimes it lasts in love, but sometimes it hurts instead"
                ),
            },
            {
                "artist": "Eminem",
                "emotion": "Aggressive Determination",
                "genre": "Hip-Hop",
                "theme": "Overcoming Adversity",
                "lyrics": (
                    "[Verse 1]\n"
                    "His palms are sweaty, knees weak, arms are heavy\n"
                    "There's vomit on his sweater already, mom's spaghetti\n"
                    "He's nervous, but on the surface he looks calm and ready\n"
                    "To drop bombs, but he keeps on forgetting\n\n"
                    "[Chorus]\n"
                    "You better lose yourself in the music, the moment\n"
                    "You own it, you better never let it go\n"
                    "You only get one shot, do not miss your chance to blow\n"
                    "This opportunity comes once in a lifetime"
                ),
            },
            {
                "artist": "Coldplay",
                "emotion": "Euphoric Hope",
                "genre": "Alternative Rock",
                "theme": "Cosmic Unity and Wonder",
                "lyrics": (
                    "[Verse 1]\n"
                    "Look at the stars, look how they shine for you\n"
                    "And everything you do, yeah they were all yellow\n"
                    "I came along, I wrote a song for you\n"
                    "And all the things you do, and it was called Yellow\n\n"
                    "[Chorus]\n"
                    "Your skin, oh yeah your skin and bones\n"
                    "Turn into something beautiful\n"
                    "Do you know, you know I love you so\n"
                    "You know I love you so"
                ),
            },
            {
                "artist": "Bob Dylan",
                "emotion": "Philosophical Contemplation",
                "genre": "Folk",
                "theme": "Social Change and Truth",
                "lyrics": (
                    "[Verse 1]\n"
                    "How many roads must a man walk down\n"
                    "Before you call him a man?\n"
                    "How many seas must a white dove sail\n"
                    "Before she sleeps in the sand?\n\n"
                    "[Chorus]\n"
                    "The answer, my friend, is blowin' in the wind\n"
                    "The answer is blowin' in the wind"
                ),
            },
        ]
        # Repeat to create a reasonable sample size for testing
        extended_records = sample_records * 25
        df = pd.DataFrame(extended_records)
        hf_dataset = Dataset.from_pandas(df)
        return DatasetDict({"train": hf_dataset})


# ==============================================================================
# PHASE 2: Tamil / Regional Lyric Dataset Loader (Extensible Placeholder)
# ==============================================================================
class TamilLyricDatasetLoader(BaseLyricDatasetLoader):
    """
    Phase 2: Pluggable loader for Regional / Tamil Lyric and Kavithai corpora.
    Example HuggingFace dataset: `abishekmahi/tamil-kavithai` or local CSV/JSON corpus.
    
    Seamlessly integrates with the same downstream SFT and PPO pipeline without
    re-architecting the core training loop.
    """

    def __init__(
        self,
        dataset_name_or_path: str = "abishekmahi/tamil-kavithai",
        split_ratio: float = 0.9,
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        super().__init__(dataset_name_or_path, split_ratio, max_samples, seed)

    def load_raw_data(self) -> Any:
        try:
            return load_dataset(self.dataset_name_or_path)
        except Exception as e:
            logger.warning(
                f"Tamil dataset '{self.dataset_name_or_path}' not found online ({e}). "
                f"Creating synthetic Tamil Kavithai dataset samples for Phase 2 readiness."
            )
            return self._create_tamil_synthetic_samples()

    def extract_metadata(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Maps Tamil poetic metadata (Kavignar / Paadal / Unarvu / Nilai):
        - Artist -> கவிஞர் / பாடலாசிரியர் (e.g. வைரமுத்து, கண்ணதாசன், பாரதியார்)
        - Mood -> உணர்வு / சுவை (e.g. காதல், வீரம், தத்துவம், சோகம்)
        - Genre -> பா வகை / பாடல் வகை (e.g. திரைப்பாடல், கவிதை, வெண்பா)
        - Theme -> கருப்பொருள் (e.g. இயற்கை, வாழ்க்கை, தாயன்பு)
        """
        artist = (
            example.get("artist")
            or example.get("poet")
            or example.get("author")
            or "பாரதியார்"
        )
        mood = (
            example.get("mood")
            or example.get("emotion")
            or example.get("Category")
            or example.get("category")
            or "வீரம் & எழுச்சி"
        )
        genre = (
            example.get("genre")
            or example.get("Category")
            or example.get("category")
            or "புரட்சிக் கவிதை"
        )
        theme = (
            example.get("theme")
            or example.get("topic")
            or example.get("Title")
            or example.get("title")
            or "விடுதலை & சமத்துவம்"
        )
        lyrics = (
            example.get("Content")
            or example.get("content")
            or example.get("lyrics")
            or example.get("poem")
            or example.get("kavithai")
            or example.get("text")
            or ""
        )

        return {
            "artist": str(artist),
            "mood": str(mood),
            "genre": str(genre),
            "theme": str(theme),
            "lyrics": str(lyrics),
        }

    def _create_tamil_synthetic_samples(self) -> DatasetDict:
        """Synthetic Tamil poetic verses for Phase 2 pipeline verification."""
        tamil_samples = [
            {
                "artist": "பாரதியார் (Bharathiyar)",
                "mood": "வீரம் (Courage & Empowerment)",
                "genre": "புரட்சிக் கவிதை (Revolutionary Poetry)",
                "theme": "பெண்ணுரிமை & சுதந்திரம்",
                "lyrics": (
                    "பட்டங்கள் ஆள்வதும் சட்டங்கள் செய்வதும்\n"
                    "பாரினில் பெண்கள் நடத்த வந்தோம்\n"
                    "எட்டும் அறிவினில் ஆணுக்குப் பெண் இங்கே\n"
                    "இளைப்பில்லை காண் என்று கும்மியடி!"
                ),
            },
            {
                "artist": "கண்ணதாசன் (Kannadasan)",
                "mood": "தத்துவம் (Philosophical Melancholy)",
                "genre": "திரை இசைப் பாடல் (Film Melody)",
                "theme": "வாழ்க்கை நிலையாமை",
                "lyrics": (
                    "வீடு வரை உறவு\n"
                    "வீதி வரை மனைவி\n"
                    "காடு வரை பிள்ளை\n"
                    "கடைசி வரை யாரோ?"
                ),
            },
            {
                "artist": "வைரமுத்து (Vairamuthu)",
                "mood": "காதல் (Romantic Euphoria)",
                "genre": "மெல்லிசை (Modern Lyricism)",
                "theme": "இயற்கை & காதல்",
                "lyrics": (
                    "சின்னச் சின்ன ரோஜா பூவே\n"
                    "செவ்வானம் போல மின்னும் நிலவே\n"
                    "தென்றல் வந்து தீண்டும் போதெல்லாம்\n"
                    "உந்தன் நினைவே என்னைத் தாலாட்டும்"
                ),
            },
        ]
        df = pd.DataFrame(tamil_samples * 10)
        hf_dataset = Dataset.from_pandas(df)
        return DatasetDict({"train": hf_dataset})


# ==============================================================================
# Helper / Factory Method
# ==============================================================================
def get_dataset_loader(
    dataset_name_or_path: str = "manoh2f2/tsterbak-lyrics-dataset-with-emotions",
    language: str = "english",
    split_ratio: float = 0.9,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> BaseLyricDatasetLoader:
    """Factory function to instantiate the corresponding language/phase dataset loader."""
    if language.lower() in ["tamil", "ta"]:
        return TamilLyricDatasetLoader(
            dataset_name_or_path=dataset_name_or_path,
            split_ratio=split_ratio,
            max_samples=max_samples,
            seed=seed
        )
    else:
        return EnglishLyricDatasetLoader(
            dataset_name_or_path=dataset_name_or_path,
            split_ratio=split_ratio,
            max_samples=max_samples,
            seed=seed
        )


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("--- Testing English Dataset Loader (Phase 1) ---")
    loader = EnglishLyricDatasetLoader(
        dataset_name_or_path="manoh2f2/tsterbak-lyrics-dataset-with-emotions",
        max_samples=50
    )
    dataset_dict = loader.prepare_dataset()
    print("Train Split Size:", len(dataset_dict["train"]))
    print("Validation Split Size:", len(dataset_dict["validation"]))
    if len(dataset_dict["train"]) > 0:
        print("\nSample SFT Prompt & Target:")
        sample = dataset_dict["train"][0]
        print(sample["text"])
    else:
        print("\nWarning: Train split is empty.")

    print("\n--- Testing Tamil Dataset Loader (Phase 2 Placeholder) ---")
    tamil_loader = TamilLyricDatasetLoader(max_samples=10)
    tamil_dict = tamil_loader.prepare_dataset()
    if len(tamil_dict["train"]) > 0:
        print("\nTamil Train Sample:\n", tamil_dict["train"][0]["text"].encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
