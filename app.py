"""
================================================================================
AI-Driven Lyric Generation through Artistic Style Modeling and Context Adaptation
Capstone Review 2: Interactive Studio & Research Evaluation Platform
================================================================================
AR Gladiators:
- R. Hariharan (23011101104)
- Srihari (23011101142)
- Yashwanth Krishna D (23011101166)
Faculty Guide: Dr. Sundhara Kumar K.B. | Shiv Nadar University Chennai
================================================================================
"""

import os
import sys
import io
import re
import json
import time
import base64
import logging
from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import custom engine
from inference_top5 import Top5LyricEngine, TOP_5_ARTISTS_INFO

# Page configuration
st.set_page_config(
    page_title="AI Lyric Generation Studio | Capstone Review 2",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Glassmorphism, Modern Typography, Vibrant Dark Mode)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, .artist-title {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
}

/* Header Banner */
.main-header {
    background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(67, 24, 255, 0.7) 50%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
    backdrop-filter: blur(16px);
}

.header-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
    border: 1px solid rgba(165, 180, 252, 0.35);
    border-radius: 9999px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

.team-badge {
    display: inline-block;
    background: rgba(16, 185, 129, 0.2);
    color: #6ee7b7;
    border: 1px solid rgba(110, 231, 183, 0.3);
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 8px;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}

/* Metric Display Cards */
.metric-box {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-box:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.5);
}
.metric-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: #818cf8;
    margin: 4px 0;
}
.metric-lbl {
    font-size: 0.76rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

/* Lyric Box */
.lyric-container {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 28px 32px;
    font-family: 'Fira Code', monospace;
    font-size: 0.95rem;
    line-height: 1.85;
    color: #f1f5f9;
    white-space: pre-wrap;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.5);
    margin: 16px 0;
}

/* Artist Selector Badge */
.artist-badge-pill {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 10px 16px;
    margin-bottom: 12px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_engine():
    """Load model once into memory."""
    adapter_candidate = "./checkpoints/top5_artists_lyric_model/final_adapter"
    if not os.path.exists(adapter_candidate):
        # Fallback to general SFT adapter if top5 adapter not finished saving yet
        adapter_candidate = "./checkpoints/sft_lyric_model/final_adapter"
    return Top5LyricEngine(
        base_model_name="Qwen/Qwen2.5-3B-Instruct",
        adapter_path=adapter_candidate if os.path.exists(adapter_candidate) else None,
    )


def text_to_speech_mp3(text: str) -> Optional[bytes]:
    """Generate vocal recitation audio for lyrics."""
    try:
        from gtts import gTTS
        clean_text = re.sub(r"\[.*?\]", "", text).strip()
        if not clean_text:
            return None
        tts = gTTS(text=clean_text[:500], lang="en", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        logger.warning(f"gTTS audio synthesis failed: {e}")
        return None


# ==============================================================================
# HEADER BANNER
# ==============================================================================
st.markdown(
    """
    <div class="main-header">
        <span class="header-badge">Capstone Review 2 — Research & Demonstration Studio</span>
        <h1 style="color: #ffffff; margin: 4px 0 10px 0; font-size: 2.2rem; font-weight: 800;">
            AI-Driven Lyric Generation through Artistic Style Modeling
        </h1>
        <p style="color: #cbd5e1; margin-bottom: 14px; font-size: 0.98rem; max-width: 900px; line-height: 1.5;">
            Instruction-driven controllable song composition conditioned on iconic artists, exact emotional valence, 
            poetic rhyme structures, and dynamic meter constraints.
        </p>
        <div>
            <span class="team-badge">AR Gladiators</span>
            <span class="team-badge">R. Hariharan (23011101104)</span>
            <span class="team-badge">Srihari (23011101142)</span>
            <span class="team-badge">Yashwanth Krishna D (23011101166)</span>
            <span class="team-badge" style="background: rgba(99, 102, 241, 0.2); color: #c7d2fe;">Guide: Dr. Sundhara Kumar K.B.</span>
            <span class="team-badge" style="background: rgba(245, 158, 11, 0.2); color: #fde68a;">Shiv Nadar University Chennai</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_studio, tab_research, tab_dataset = st.tabs([
    "🎵 Interactive Lyric Studio",
    "📊 Review 2 Research Framework",
    "📁 Top 5 Artists Dataset Provenance"
])

# ==============================================================================
# TAB 1: INTERACTIVE LYRIC STUDIO
# ==============================================================================
with tab_studio:
    # Sidebar Controls
    with st.sidebar:
        st.markdown("### 🎛️ Artistic Conditioning Controls")
        st.caption("Configure multi-attribute constraints per Review 2 Schema (Slide 8)")

        # 1. Artist Dropdown (Strictly Top 5 Artists)
        artist_options = list(TOP_5_ARTISTS_INFO.keys())
        selected_artist = st.selectbox(
            "🎤 Artist / Style (Dropdown):",
            options=artist_options,
            index=0,
            help="Trained exclusively on Top 5 artists with maximum song counts."
        )

        artist_meta = TOP_5_ARTISTS_INFO[selected_artist]
        st.markdown(
            f"""
            <div class="artist-badge-pill">
                <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem;">
                    {artist_meta['icon']} {selected_artist}
                </div>
                <div style="font-size: 0.78rem; color: #a5b4fc; margin-top: 2px;">
                    Genre: <b>{artist_meta['genre']}</b> ({artist_meta['era']})
                </div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px; line-height: 1.35;">
                    {artist_meta['desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 2. Genre Input (Pre-filled with artist default)
        genre_input = st.text_input(
            "🎸 Genre / Musical Style:",
            value=artist_meta["genre"],
            help="Musical genre envelope (e.g. Rockabilly, Outlaw Country, Big Band Jazz)"
        )

        # 3. Emotion / Mood Selection
        emotion_categories = [
            "Melancholic Grief & Heartbreak",
            "Euphoric Romance & Joy",
            "Midnight Nostalgia & Solitude",
            "Defiant Passion & Triumph",
            "Haunted Longing & Vulnerability",
            "Smoky Barstool Reflection",
            "Bittersweet Wonder & Regret"
        ]
        selected_mood = st.selectbox(
            "🎭 Emotion / Mood Valence:",
            options=emotion_categories,
            index=0,
            help="Target emotional tone measured by sentence embedding cosine similarity."
        )

        # 4. Length of Song / Structure
        length_option = st.radio(
            "📏 Length of Song / Structure:",
            options=[
                "Short (1 Verse / 4–8 lines)",
                "Medium (2 Verses + Chorus / 12–16 lines)",
                "Full Song (Verse, Chorus, Bridge, Outro / 20+ lines)"
            ],
            index=1,
            help="Controls line output and generation token budget."
        )

        token_map = {
            "Short (1 Verse / 4–8 lines)": 160,
            "Medium (2 Verses + Chorus / 12–16 lines)": 280,
            "Full Song (Verse, Chorus, Bridge, Outro / 20+ lines)": 420
        }
        max_tokens = token_map[length_option]

        # 5. Rhyme Scheme Constraint (Slide 8)
        rhyme_scheme = st.selectbox(
            "🔤 Target Rhyme Scheme:",
            options=["AABB (Couplets)", "ABAB (Alternating)", "AABBCC (Multi-Couplet)", "Free Verse / Expressive"],
            index=0,
        )

        # 6. Song Section Target
        section_option = st.selectbox(
            "📑 Song Section:",
            options=["Full Song", "[Verse 1]", "[Chorus]", "[Verse & Chorus]", "[Bridge]"],
            index=0,
        )

        # Advanced generation parameters
        with st.expander("⚙️ Advanced Hyperparameters (LLM)"):
            temperature = st.slider("Temperature (Creativity):", 0.4, 1.2, 0.75, 0.05)
            top_p = st.slider("Top-p Nucleus Sampling:", 0.6, 1.0, 0.90, 0.02)
            repetition_penalty = st.slider("Repetition Penalty:", 1.0, 1.35, 1.15, 0.05)

        st.markdown("---")
        st.caption("⚡ Hardware: RTX 5050 GPU (4-bit QLoRA) | <8GB RAM Safe")

    # Main Panel Inputs
    col_main_left, col_main_right = st.columns([7, 5])

    with col_main_left:
        st.markdown("#### 🎯 Theme & Narrative Concept")
        
        # Suggested preset pills
        sample_themes = artist_meta["sample_themes"]
        st.caption("Quick Select Signature Themes:")
        preset_cols = st.columns(len(sample_themes[:3]))
        
        default_theme_text = sample_themes[0]
        selected_theme_pill = None
        for idx, th in enumerate(sample_themes[:3]):
            if preset_cols[idx].button(f"💡 {th.split('(')[0].strip()}", key=f"pill_{idx}", use_container_width=True):
                default_theme_text = th

        theme_input = st.text_input(
            "Song Theme / Narrative Goal:",
            value=default_theme_text,
            help="Specific topic or story line for the song lyrics."
        )

    with col_main_right:
        st.markdown("#### 📋 Conditioning Summary")
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px; font-size: 0.85rem; line-height: 1.6;">
                <div><b>Artist:</b> <span style="color: #818cf8;">{selected_artist}</span></div>
                <div><b>Genre:</b> <span style="color: #cbd5e1;">{genre_input}</span></div>
                <div><b>Mood:</b> <span style="color: #cbd5e1;">{selected_mood}</span></div>
                <div><b>Length:</b> <span style="color: #cbd5e1;">{length_option.split('(')[0].strip()} (~{max_tokens} tokens)</span></div>
                <div><b>Rhyme:</b> <span style="color: #cbd5e1;">{rhyme_scheme}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Generate CTA
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("🚀 Generate AI Song Lyrics", type="primary", use_container_width=True)

    # State container for generated output
    if "latest_generation" not in st.session_state:
        st.session_state.latest_generation = None

    if generate_btn:
        with st.spinner("🎸 Generating artist-styled song lyrics on NVIDIA RTX 5050 GPU..."):
            t_start = time.time()
            engine = get_engine()
            output = engine.generate(
                artist=selected_artist,
                mood=selected_mood,
                genre=genre_input,
                theme=theme_input,
                section=section_option,
                rhyme_scheme=rhyme_scheme,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            output["latency_sec"] = round(time.time() - t_start, 2)
            st.session_state.latest_generation = output

    # Display Results
    if st.session_state.latest_generation:
        gen = st.session_state.latest_generation
        metrics = gen["metrics"]

        st.markdown("---")
        st.markdown("### 🎙️ Generated Song Lyrics")
        st.caption(f"Prompt Header: `{gen['prompt']}` | Generation Time: **{gen['latency_sec']}s** | LoRA Adapter Active: **{gen['has_lora_adapter']}**")

        col_lyric_disp, col_metrics_disp = st.columns([7, 5])

        with col_lyric_disp:
            st.markdown(f'<div class="lyric-container">{gen["lyrics"]}</div>', unsafe_allow_html=True)

            # Audio Player Preview
            audio_bytes = text_to_speech_mp3(gen["lyrics"])
            if audio_bytes:
                st.markdown("##### 🔊 Vocalization & Meter Preview (Audio):")
                st.audio(audio_bytes, format="audio/mp3")

            # Download Buttons
            dl_col1, dl_col2 = st.columns(2)
            lyric_download = f"AI-Generated Song Lyrics\nArtist: {gen['artist']}\nGenre: {gen['genre']}\nMood: {gen['mood']}\nTheme: {gen['theme']}\n\n[Lyrics]:\n{gen['lyrics']}"
            dl_col1.download_button(
                "📥 Download Lyrics (.txt)",
                data=lyric_download,
                file_name=f"{gen['artist'].replace(' ', '_')}_lyrics.txt",
                mime="text/plain",
                use_container_width=True,
            )
            dl_col2.download_button(
                "📊 Export Evaluation JSON",
                data=json.dumps(gen, indent=2),
                file_name=f"{gen['artist'].replace(' ', '_')}_evaluation.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_metrics_disp:
            st.markdown("### 📊 Multi-Objective Quality Scores")
            st.caption("Automated evaluation signals mapped from Review 2 (Slide 10)")

            # Metric Boxes
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-lbl">Composite Quality</div>
                        <div class="metric-val" style="color: #38bdf8;">{metrics['composite_score']}%</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Calibrated Index</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-lbl">Phonetic Rhyme</div>
                        <div class="metric-val" style="color: #a855f7;">{metrics['rhyme_score']}%</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">CMUdict Match</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            m_col3, m_col4 = st.columns(2)
            with m_col3:
                st.markdown(
                    f"""
                    <div class="metric-box" style="margin-top: 12px;">
                        <div class="metric-lbl">Mood Alignment</div>
                        <div class="metric-val" style="color: #ec4899;">{metrics['mood_alignment']}%</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Cosine Similarity</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m_col4:
                st.markdown(
                    f"""
                    <div class="metric-box" style="margin-top: 12px;">
                        <div class="metric-lbl">Fluency & Flow</div>
                        <div class="metric-val" style="color: #10b981;">{metrics['fluency_score']}%</div>
                        <div style="font-size: 0.72rem; color: #94a3b8;">Non-Repetitive Lines</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Meter Stats
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### 📐 Meter & Stanza Architecture:")
            st.markdown(
                f"""
                - **Total Lyric Lines**: {metrics['num_lines']} lines
                - **Average Syllables per Line**: {metrics['avg_syllables']} syllables
                - **Syllable Variance**: {metrics['syllable_variance']} (Lower indicates tighter rhythm)
                """
            )

            # Rhyme Pairs Detected
            st.markdown("##### 🔗 Detected End-Rhyme Couplets:")
            if metrics["identified_pairs"]:
                for pair in metrics["identified_pairs"][:5]:
                    st.markdown(f"- 🎵 `{pair}`")
            else:
                st.info("Slant or expressive free verse phrasing detected.")


# ==============================================================================
# TAB 2: REVIEW 2 RESEARCH FRAMEWORK & ARCHITECTURE
# ==============================================================================
with tab_research:
    st.markdown("## 📚 Review 2 Research Foundation & Literature Synthesis")
    st.caption("Directly referencing Capstone Review 2 Presentation & Research Document")

    col_res_l, col_res_r = st.columns([6, 6])

    with col_res_l:
        st.markdown("### 1. Research Evolution (Slide 3)")
        evolution_df = pd.DataFrame([
            {"Year": "2018", "Milestone": "Artist-conditioned lyric lines", "Method": "VAE + Artist Embeddings (Vechtomova)"},
            {"Year": "2019", "Milestone": "Context & Conditional Templates", "Method": "Hierarchical RNNs"},
            {"Year": "2020–2021", "Milestone": "Transformer Lyric Generation", "Method": "Denoising, Pretraining, Rhyme & Rhythm (Rapformer / DeepRapper)"},
            {"Year": "2022–2024", "Milestone": "Fine-Grained Poetic Control", "Method": "Length, Syllables, Planning (ChipSong / LYRA)"},
            {"Year": "2025", "Milestone": "Instruction-Driven Songwriting", "Method": "LLaMA 3.1 8B + Multi-Objective Line Losses (LYRICS ICTC 2025)"},
            {"Year": "Proposed", "Milestone": "Unified Artistic Style Modeling", "Method": "QLoRA Top-5 Artists + Multi-Objective Rhyme/Mood/Meter Reward"},
        ])
        st.dataframe(evolution_df, use_container_width=True, hide_index=True)

        st.markdown("### 2. Closest Technical Analogue: LYRICS (ICTC 2025) vs Baseline (Slide 5)")
        lyrics_comp = pd.DataFrame([
            {"Metric": "BERTScore F1", "Baseline (CE-only)": "0.4232", "LYRICS 2025": "0.4630", "Improvement": "+9.40%"},
            {"Metric": "ROUGE-1", "Baseline (CE-only)": "0.1689", "LYRICS 2025": "0.2127", "Improvement": "+25.93%"},
            {"Metric": "ROUGE-2", "Baseline (CE-only)": "0.0823", "LYRICS 2025": "0.1193", "Improvement": "+44.96%"},
            {"Metric": "ROUGE-L", "Baseline (CE-only)": "0.1660", "LYRICS 2025": "0.2095", "Improvement": "+26.20%"},
        ])
        st.dataframe(lyrics_comp, use_container_width=True, hide_index=True)

    with col_res_r:
        st.markdown("### 3. Multi-Dimensional Evaluation Framework (Slide 10)")
        eval_framework = pd.DataFrame([
            {"Dimension": "Style Modeling", "Metric": "Artist classifier & style-calibrated similarity", "Failure Caught": "Generic or style-drifting text"},
            {"Dimension": "Mood / Emotion", "Metric": "Sentence-Transformers MiniLM Cosine Similarity", "Failure Caught": "Emotional tone mismatch"},
            {"Dimension": "Meter & Rhythm", "Metric": "CMUdict exact syllable count & variance", "Failure Caught": "Too-short / too-long run-on lines"},
            {"Dimension": "Phonetic Rhyme", "Metric": "CMU pronouncing phonetic end-rhyme match", "Failure Caught": "String-similar but unrhymed text"},
            {"Dimension": "Diversity & Fluency", "Metric": "Distinct-n & line loop penalty", "Failure Caught": "Degenerative line repetition"},
        ])
        st.dataframe(eval_framework, use_container_width=True, hide_index=True)

        st.markdown("### 4. System Pipeline Architecture (Slide 7)")
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A[User Instruction: Artist, Mood, Genre, Theme, Structure] --> B[Instruction-Tuned LLM: Qwen2.5-3B]
                B --> C[Parameter-Efficient QLoRA Adapter]
                C --> D[Multi-Objective Quality Engine]
                D --> E1[Phonetic Rhyme CMUdict]
                D --> E2[Mood Cosine MiniLM]
                D --> E3[Fluency & Meter Regularity]
                D --> F[Evaluated Lyrics & Audio Synthesis]
            ```
            """
        )


# ==============================================================================
# TAB 3: TOP 5 ARTISTS DATASET PROVENANCE
# ==============================================================================
with tab_dataset:
    st.markdown("## 📁 Top 5 Artists Dataset Provenance & Distribution")
    st.caption("Trained on the Top 5 artists with maximum song counts from `manoh2f2/tsterbak-lyrics-dataset-with-emotions`")

    col_data_l, col_data_r = st.columns([6, 6])

    with col_data_l:
        st.markdown("### 🏆 Top 5 Artists by Song Count")
        top5_counts_df = pd.DataFrame([
            {"Artist": "Elvis Presley", "Songs in Corpus": 598, "Primary Genre": "Rock & Roll / Rockabilly", "Era": "1950s - 1970s"},
            {"Artist": "Willie Nelson", "Songs in Corpus": 427, "Primary Genre": "Country & Outlaw Folk", "Era": "1960s - Present"},
            {"Artist": "Ella Fitzgerald", "Songs in Corpus": 338, "Primary Genre": "Vocal Jazz & Swing", "Era": "1930s - 1980s"},
            {"Artist": "Frank Sinatra", "Songs in Corpus": 302, "Primary Genre": "Traditional Pop & Big Band", "Era": "1940s - 1990s"},
            {"Artist": "Dean Martin", "Songs in Corpus": 289, "Primary Genre": "Easy Listening & Crooner Pop", "Era": "1940s - 1980s"},
        ])
        st.dataframe(top5_counts_df, use_container_width=True, hide_index=True)

        st.bar_chart(
            top5_counts_df.set_index("Artist")["Songs in Corpus"],
            color="#818cf8"
        )

    with col_data_r:
        st.markdown("### 🎭 Emotion Distribution across Top 5 Artists")
        emotions_df = pd.DataFrame([
            {"Emotion": "Sadness (Heartbreak / Solitude)", "Occurrences": 749, "Mapped Mood": "Melancholic Grief & Longing"},
            {"Emotion": "Fear (Vulnerability / Trepidation)", "Occurrences": 372, "Mapped Mood": "Haunted Longing & Vulnerability"},
            {"Emotion": "Joy (Romance / Celebration)", "Occurrences": 322, "Mapped Mood": "Euphoric Romance & Joy"},
            {"Emotion": "Neutral (Reflective / Easy-going)", "Occurrences": 289, "Mapped Mood": "Reflective Nostalgia & Serenity"},
            {"Emotion": "Anger (Defiance / Sarcasm)", "Occurrences": 100, "Mapped Mood": "Defiant Passion & Frustration"},
            {"Emotion": "Surprise (Bittersweet Wonder)", "Occurrences": 99, "Mapped Mood": "Bittersweet Wonder & Shock"},
        ])
        st.dataframe(emotions_df, use_container_width=True, hide_index=True)

        st.markdown("### 🔖 Standardized Training Sample Format")
        st.code(
            """[Artist]: Frank Sinatra | [Mood]: Midnight Melancholy & Resilient Hope | [Genre]: Traditional Pop & Big Band | [Theme]: In the Wee Small Hours of the Morning
[Lyrics]:
In the wee small hours of the morning
While the whole wide world is fast asleep
You lie awake and think about the girl
And never, ever think of counting sheep""",
            language="text"
        )
