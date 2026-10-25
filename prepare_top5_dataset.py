"""
================================================================================
AI-Driven Lyric Generation: Top 5 Artists Dataset Preparation
Filters the dataset strictly to the Top 5 artists with maximum songs:
1. Elvis Presley (598 songs) - Rock & Roll / Rockabilly
2. Willie Nelson (427 songs) - Country & Outlaw Folk
3. Ella Fitzgerald (338 songs) - Vocal Jazz & Swing
4. Frank Sinatra (302 songs) - Traditional Pop & Big Band
5. Dean Martin (289 songs) - Easy Listening & Crooner Pop
================================================================================
"""

import os
import re
import ast
import gc
import json
import logging
from typing import Dict, Any, List
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TOP_5_ARTISTS_META = {
    "Elvis Presley": {
        "genre": "Rock & Roll / Rockabilly",
        "style_desc": "Soulful rockabilly rhythm, charismatic vibrato, energetic guitar hooks",
        "default_theme": "Passionate Romance and Heartbreak",
    },
    "Willie Nelson": {
        "genre": "Country & Outlaw Folk",
        "style_desc": "Acoustic storytelling, laid-back narrative cadence, rustic melancholy",
        "default_theme": "Rambling Roads, Whiskey and Solitude",
    },
    "Ella Fitzgerald": {
        "genre": "Vocal Jazz & Swing",
        "style_desc": "Impeccable diction, swinging rhythm, playful and elegant phrasing",
        "default_theme": "Starlight Romance and Sophisticated Longing",
    },
    "Frank Sinatra": {
        "genre": "Traditional Pop & Big Band",
        "style_desc": "Dramatic phrasing, velvety baritone swagger, timeless urban romance",
        "default_theme": "Midnight Melancholy and Resilient Ambition",
    },
    "Dean Martin": {
        "genre": "Easy Listening & Crooner Pop",
        "style_desc": "Effortless casual charm, warm melodic flow, romantic nonchalance",
        "default_theme": "Sunlit Nostalgia and Carefree Love",
    },
}

EMOTION_MAP = {
    "sadness": "Melancholic Grief & Heartbreak",
    "joy": "Euphoric Romance & Joy",
    "fear": "Haunted Longing & Vulnerability",
    "anger": "Defiant Passion & Frustration",
    "surprise": "Bittersweet Wonder & Shock",
    "neutral": "Reflective Nostalgia & Serenity",
    "disgust": "Disillusionment & Cynicism",
}


def clean_lyrics_text(text: str) -> str:
    """Clean lyric text by removing web artifacts, carriage returns, and excessive blanks."""
    if not isinstance(text, str) or not text.strip():
        return ""

    # Remove Excel / XML export tokens
    cleaned = text.replace("_x000D_", "").replace("\r", "")
    
    # Remove metadata noise
    cleaned = re.sub(r"^\d+\s+Contributors?.*?\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\d*Embed$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[.*?See .*? Live.*?\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(.*?Produced by.*?\)", "", cleaned, flags=re.IGNORECASE)

    # Normalize stanzas
    lines = [line.strip() for line in cleaned.split("\n")]
    result_lines = []
    consecutive_blank = 0
    for line in lines:
        if not line:
            consecutive_blank += 1
            if consecutive_blank <= 1 and result_lines:
                result_lines.append("")
        else:
            consecutive_blank = 0
            result_lines.append(line)

    return "\n".join(result_lines).strip()


def extract_emotion(raw_emotions: Any) -> str:
    """Normalize emotions field into a descriptive mood string."""
    if isinstance(raw_emotions, list) and raw_emotions:
        primary = str(raw_emotions[0]).lower().strip()
    elif isinstance(raw_emotions, str):
        try:
            parsed = ast.literal_eval(raw_emotions)
            if isinstance(parsed, list) and parsed:
                primary = str(parsed[0]).lower().strip()
            else:
                primary = str(raw_emotions).lower().strip()
        except Exception:
            primary = str(raw_emotions).lower().strip()
    else:
        primary = "neutral"

    return EMOTION_MAP.get(primary, f"{primary.title()} & Melodic Sentiment")


def prepare_top5_dataset(
    output_dir: str = "./top5_dataset",
    hf_dataset_name: str = "manoh2f2/tsterbak-lyrics-dataset-with-emotions",
    split_ratio: float = 0.9,
    seed: int = 42,
) -> DatasetDict:
    """Load, filter to top 5 artists, clean, format, and save."""
    os.makedirs(output_dir, exist_ok=True)
    meta_cache_path = os.path.join(output_dir, "metadata_summary.json")

    logger.info("Loading base dataset from Hugging Face cache...")
    raw_ds = load_dataset(hf_dataset_name, split="train")

    top_5_names = set(TOP_5_ARTISTS_META.keys())
    logger.info(f"Filtering dataset strictly for top 5 artists: {top_5_names}")

    # Streamlined filtering
    filtered_rows = []
    for row in raw_ds:
        artist_raw = str(row.get("artist", "")).strip()
        if artist_raw in top_5_names:
            filtered_rows.append(row)

    logger.info(f"Retained {len(filtered_rows)} records for the Top 5 artists.")
    del raw_ds
    gc.collect()

    # Process and build prompts
    processed_records = []
    artist_counts = {name: 0 for name in top_5_names}

    for row in filtered_rows:
        artist = str(row.get("artist", "")).strip()
        song = str(row.get("song", "")).strip() or "Untitled Song"
        raw_emotions = row.get("emotions")
        seq = row.get("seq", "")

        lyrics = clean_lyrics_text(seq)
        line_count = len([l for l in lyrics.split("\n") if l.strip()])
        
        # Quality threshold: At least 4 lines and at least 40 characters
        if line_count < 4 or len(lyrics) < 40:
            continue

        meta = TOP_5_ARTISTS_META.get(artist, {})
        genre = meta.get("genre", "Classic Pop & Rock")
        mood = extract_emotion(raw_emotions)
        theme = f"{song} - {meta.get('default_theme', 'Love & Life')}"

        prompt = (
            f"[Artist]: {artist} | [Mood]: {mood} | "
            f"[Genre]: {genre} | [Theme]: {theme}"
        )
        completion = f"\n[Lyrics]:\n{lyrics}"
        full_text = f"{prompt}{completion}"

        processed_records.append({
            "artist": artist,
            "song": song,
            "mood": mood,
            "genre": genre,
            "theme": theme,
            "lyrics": lyrics,
            "prompt": prompt,
            "completion": completion,
            "text": full_text,
            "line_count": line_count,
        })
        artist_counts[artist] += 1

    logger.info(f"Valid cleaned records ready for training: {len(processed_records)}")
    logger.info(f"Breakdown per artist: {artist_counts}")

    # Create HuggingFace Dataset
    df = pd.DataFrame(processed_records)
    hf_dataset = Dataset.from_pandas(df)

    # Train / Val Split
    split_dataset = hf_dataset.train_test_split(
        train_size=split_ratio,
        seed=seed,
        shuffle=True
    )
    
    ds_dict = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

    # Save to disk
    logger.info(f"Saving prepared dataset to {output_dir}...")
    ds_dict.save_to_disk(output_dir)

    # Save metadata summary
    summary = {
        "top_5_artists": list(top_5_names),
        "artists_meta": TOP_5_ARTISTS_META,
        "emotion_map": EMOTION_MAP,
        "artist_counts": artist_counts,
        "total_valid_songs": len(processed_records),
        "train_samples": len(split_dataset["train"]),
        "val_samples": len(split_dataset["test"]),
    }
    with open(meta_cache_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Dataset preparation complete! Saved to {output_dir}")
    return ds_dict


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    prepare_top5_dataset()
