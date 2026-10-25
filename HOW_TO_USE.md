# AI-Driven Lyric Generation through Artistic Style Modeling and Context Adaptation
## Comprehensive Operational & Training Guide (Phase 1 & Phase 2)

---

## 1. Executive Summary & Architecture

This repository delivers an end-to-end, production-grade framework for **controllable, artistically styled, and emotionally conditioned song lyric generation**. 

The system leverages **Hugging Face Transformers**, **PEFT (Parameter-Efficient Fine-Tuning with LoRA/QLoRA)**, and **TRL (Transformer Reinforcement Learning)** using **Multi-Objective Reinforcement Learning (PPO)** to steer generated lyrics toward rich poetic rhyme structures and exact emotional alignment.

```mermaid
flowchart TD
    A[Hugging Face Lyrics Corpus<br/>manoh2f2/tsterbak-lyrics-dataset-with-emotions] -->|Step 1: Ingestion & Formatting| B[dataset_loader.py<br/>Prompt: Artist | Mood | Genre | Theme]
    B -->|Step 2: PEFT LoRA Fine-Tuning| C[sft_trainer.py<br/>Llama-3-8B / Mistral-7B + LoRA]
    C -->|SFT Checkpoint / Adapters| D[rl_ppo_trainer.py<br/>Policy Network + Value Head]
    E[reward_engine.py<br/>Multi-Objective Scoring] -->|Phonetic Rhyme CMUdict + Mood Cosine Embeddings| D
    D -->|Step 4: PPO Policy Optimization| F[PPO Aligned Model<br/>High Rhyme & Mood Fidelity]
    F -->|Step 5: Generation| G[inference.py<br/>Interactive Studio & CLI]
```

### The 4-Step Pipeline:
1. **Data Ingestion (`dataset_loader.py`)**: Parses emotional annotations, genres, artists, and themes into structured conditioning headers.
2. **Supervised Fine-Tuning (`sft_trainer.py`)**: Teaches the base LLM (Llama-3-8B or Mistral-7B) to adhere to the style prompt structure and stanza formatting.
3. **Multi-Objective Reward Engine (`reward_engine.py`)**: Computes mathematical quality signals:
   - **Phonetic Rhyme Score**: CMU pronouncing dictionary parsing of end-rhymes (couplets, alternating stanzas, slant rhymes).
   - **Mood Alignment Score**: Cosine similarity in embedding space via `sentence-transformers/all-MiniLM-L6-v2`.
   - **Fluency & Repetition Penalty**: Penalizes line loops and degenerated text.
4. **Policy Optimization (`rl_ppo_trainer.py`)**: Reinforcement learning via PPO to optimize the policy model directly against composite lyric reward.

---

## 2. Installation & High-Performance Hardware Tuning

### 2.1 Virtual Environment Setup

```bash
# 1. Create and activate a Python 3.10+ virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# 2. Install PyTorch with CUDA support (e.g. CUDA 12.1 / 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install repository dependencies
pip install -r requirements.txt
```

### 2.2 Low-VRAM & High-Compute Hardware Tuning

> 💡 **Windows PowerShell Notice**: In PowerShell, Linux multi-line backslash (`\`) continuation characters cause `unrecognized arguments: \ \ \` errors. Use **single-line commands** (recommended) or PowerShell backticks (`` ` ``).

#### Recommended Settings by GPU VRAM:

| GPU Tier / VRAM | Batch Size | Grad Accum | LoRA Rank ($r$) | Quantization | Max Samples (Recommended) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **8GB VRAM (RTX 5050 / 3050 / 4050)** | `1` | `16` | `16` | QLoRA 4-bit (default) | `500` (for fast runs) |
| **12GB–16GB VRAM (RTX 3060 / 4070)** | `2`–`4` | `4`–`8` | `32` | QLoRA 4-bit (default) | Full dataset |
| **24GB+ VRAM (RTX 3090 / 4090 / A100)** | `8` | `2` | `64` | `--no_4bit` (bfloat16) | Full dataset |

---

## 3. Step-by-Step Pipeline Execution

### Step 1: Ingest & Inspect Dataset (`dataset_loader.py`)

Test the dataset ingestion and formatting logic:

```powershell
python dataset_loader.py
```

This loads `manoh2f2/tsterbak-lyrics-dataset-with-emotions`, cleans the text, and outputs sample structured prompts:
```text
[Artist]: Adele | [Mood]: Melancholic Grief | [Genre]: Soul Pop | [Theme]: Lost Love and Reflection
[Lyrics]:
[Verse 1]
I heard that you're settled down...
```

---

### Step 2: Supervised Fine-Tuning (`sft_trainer.py`)

Fine-tune the base model to learn the prompt format and lyric generation patterns.

#### A. 8GB VRAM & 16GB RAM Laptop Run (Verified 3B Architecture):
> 🚀 **Fastest & Lightest**: Loads in 2.5 seconds, uses only **~1.9 GB VRAM** and **~3.5 GB RAM**, and trains smoothly on your RTX 5050 Laptop GPU!

**PowerShell (Qwen 2.5 3B - Open Access):**
```powershell
python sft_trainer.py --model_name "Qwen/Qwen2.5-3B-Instruct" --dataset_name "manoh2f2/tsterbak-lyrics-dataset-with-emotions" --output_dir "./checkpoints/sft_lyric_model" --epochs 3 --batch_size 1 --grad_accum 16 --lr 2e-4 --lora_r 16 --max_seq_length 512
```

**PowerShell (Meta LLaMA 3.2 3B):**
```powershell
python sft_trainer.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --dataset_name "manoh2f2/tsterbak-lyrics-dataset-with-emotions" --output_dir "./checkpoints/sft_lyric_model" --epochs 3 --batch_size 1 --grad_accum 16 --lr 2e-4 --lora_r 16 --max_seq_length 512
```

#### B. 8GB VRAM (Full LLaMA 3 8B with Low-RAM Tuning):
```powershell
python sft_trainer.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --dataset_name "manoh2f2/tsterbak-lyrics-dataset-with-emotions" --output_dir "./checkpoints/sft_lyric_model" --epochs 3 --batch_size 1 --grad_accum 16 --lr 2e-4 --lora_r 16 --max_seq_length 512
```

#### C. Standard 12GB–16GB VRAM Run:
```powershell
python sft_trainer.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --dataset_name "manoh2f2/tsterbak-lyrics-dataset-with-emotions" --output_dir "./checkpoints/sft_lyric_model" --epochs 3 --batch_size 4 --grad_accum 4 --lr 2e-4 --lora_r 32
```

#### C. High-End Heavy Training (Full bfloat16 LoRA on 24GB+ VRAM):
```powershell
python sft_trainer.py --model_name "meta-llama/Meta-Llama-3-8B-Instruct" --output_dir "./checkpoints/sft_lyric_model" --no_4bit --epochs 4 --batch_size 8 --grad_accum 2 --lr 2e-4 --lora_r 64 --lora_alpha 128
```

*Output Checkpoint*: Saved to `./checkpoints/sft_lyric_model/final_adapter`.

---

### Step 3: Multi-Objective Reward Engine Testing (`reward_engine.py`)

Verify the multi-objective reward engine against sample verses:

```powershell
python reward_engine.py
```

#### Reward Math Formulation:
$$R_{\text{composite}} = w_{\text{rhyme}} \cdot R_{\text{rhyme}} + w_{\text{mood}} \cdot R_{\text{mood}} + w_{\text{fluency}} \cdot R_{\text{fluency}}$$
- **$R_{\text{rhyme}}$ (Weight: 0.45)**: Phonetic end-rhyme match density calculated via CMUdict phonetic transcriptions.
- **$R_{\text{mood}}$ (Weight: 0.40)**: Cosine similarity between sentence embeddings of the target emotion and the generated verses.
- **$R_{\text{fluency}}$ (Weight: 0.15)**: Line variance balance and repetition penalty.

---

### Step 4: Policy Optimization with PPO (`rl_ppo_trainer.py`)

Optimize the SFT model using Reinforcement Learning to maximize poetic rhyme and emotional fidelity:

#### A. 8GB VRAM (RTX 5050 with Qwen2.5-3B-Instruct):
```powershell
python rl_ppo_trainer.py --sft_model_path "./checkpoints/sft_lyric_model/final_adapter" --base_model "Qwen/Qwen2.5-3B-Instruct" --output_dir "./checkpoints/ppo_lyric_model" --steps 200 --batch_size 4 --mini_batch_size 1 --lr 1.41e-5 --log_with "tensorboard"
```

#### B. Standard Run (Llama-3-8B):
```powershell
python rl_ppo_trainer.py --sft_model_path "./checkpoints/sft_lyric_model/final_adapter" --base_model "meta-llama/Meta-Llama-3-8B-Instruct" --output_dir "./checkpoints/ppo_lyric_model" --steps 300 --batch_size 8 --mini_batch_size 2 --lr 1.41e-5 --log_with "tensorboard"
```

To monitor training curves in real time:
```powershell
tensorboard --logdir ./checkpoints/ppo_lyric_model/logs
```

---

### Step 5: Controllable Lyric Generation (`inference.py`)

#### A. Interactive Studio Mode (Terminal Session):
```powershell
python inference.py --model_name "Qwen/Qwen2.5-3B-Instruct" --adapter_path "./checkpoints/ppo_lyric_model/final_ppo_model" --interactive
```

#### B. Single-Prompt Execution:
```powershell
python inference.py --model_name "Qwen/Qwen2.5-3B-Instruct" --adapter_path "./checkpoints/ppo_lyric_model/final_ppo_model" --artist "Freddie Mercury" --mood "Operatic Passion & Defiance" --genre "Progressive Rock" --theme "Breaking Free from the Chains of Time" --temperature 0.8 --max_tokens 350
```

---

## 4. Phase 2: Tamil / Regional Language Adaptation

The architecture is built with an abstracted, modular design so that transitioning to Phase 2 (e.g. Tamil poetry/lyrics) requires **zero architectural refactoring**.

### 1. Dataset Swapping:
In `dataset_loader.py`, `TamilLyricDatasetLoader` is already integrated to map Tamil poetic metadata:
- **Artist** $\rightarrow$ கவிஞர் (e.g., பாரதியார், கண்ணதாசன், வைரமுத்து)
- **Mood** $\rightarrow$ உணர்வு (e.g., வீரம், காதல், தத்துவம்)
- **Genre** $\rightarrow$ பா வகை (e.g., திரைப்பாடல், மரபுக்கவிதை, புதுக்கவிதை)
- **Theme** $\rightarrow$ கருப்பொருள் (e.g., விடுதலை, தாய்மை, இயற்கை)

To train on Tamil datasets (e.g., `abishekmahi/tamil-kavithai`):
```powershell
python sft_trainer.py --language "tamil" --dataset_name "abishekmahi/tamil-kavithai" --output_dir "./checkpoints/tamil_sft_model" --batch_size 1 --grad_accum 16
```

### 2. Tamil Prosody Reward Engine:
`TamilRhymeEvaluator` evaluates classical Tamil prosody rules (யாப்பிலக்கணம்):
- **மோனை (Monai)**: Initial letter/syllable matching across lines.
- **எதுகை (Etukai)**: Second letter harmony across lines (primary rhyming convention in Tamil poetry).
- **இயைபு (Iyalbu)**: End-line vowel resonance.

To run PPO on Tamil:
```powershell
python rl_ppo_trainer.py --language "tamil" --sft_model_path "./checkpoints/tamil_sft_model/final_adapter" --output_dir "./checkpoints/tamil_ppo_model" --batch_size 4 --mini_batch_size 1
```

---

## 5. Review 2: Top 5 Artists Streamlit Web Studio

A dedicated, evaluator-ready Streamlit front-end and Top 5 artists training framework:

### 1. Launching the Streamlit Web Application
```powershell
streamlit run app.py
```
Or double-click `run_app.bat` (in `./Code`) or `launch_streamlit.bat` (in project root).

### 2. Top 5 Artists Fine-Tuned Model Pipeline
- **Artists with Maximum Songs**:
  1. **Elvis Presley** (598 songs) - Rock & Roll / Rockabilly
  2. **Willie Nelson** (427 songs) - Country & Outlaw Folk
  3. **Ella Fitzgerald** (338 songs) - Vocal Jazz & Swing
  4. **Frank Sinatra** (302 songs) - Traditional Pop & Big Band
  5. **Dean Martin** (289 songs) - Easy Listening & Crooner Pop
- **Dataset Preparation**: `python prepare_top5_dataset.py` (filters strictly to top 5 artists and structures multi-attribute conditioning).
- **GPU Fine-Tuning**: `python train_top5_model.py` (trains LoRA adapters in 4-bit bfloat16 QLoRA on NVIDIA RTX 5050 GPU, output saved to `./checkpoints/top5_artists_lyric_model/final_adapter`).
- **Inference & Metrics**: `python inference_top5.py` (calculates Phonetic Rhyme, Mood Alignment, Fluency & Flow, and Meter Regularity).

---

## 6. Repository Structure Reference

```
d:/Capstone/Code/
├── app.py                  # Streamlit Interactive Web Studio & Evaluator Platform
├── inference_top5.py       # Top 5 Artists inference engine & live multi-metric scoring
├── prepare_top5_dataset.py # Data ingestion & cleaning for Top 5 artists
├── train_top5_model.py     # QLoRA fine-tuning pipeline for Top 5 artists
├── run_app.bat             # Double-click launcher for Streamlit front-end
├── top5_dataset/           # Memory-mapped dataset cache for Top 5 artists
├── prompt.md               # Original capstone specification & requirements
├── requirements.txt        # Pinned dependencies (PyTorch, Transformers, TRL, PEFT, Streamlit)
├── config.py               # Dataclass configurations & high-spec hardware presets
├── dataset_loader.py       # Step 1: English & Tamil dataset loaders and prompt builders
├── sft_trainer.py          # Step 2: LoRA / QLoRA SFT training loop
├── reward_engine.py        # Step 3: Multi-Objective Rhyme & Mood reward scoring
├── rl_ppo_trainer.py       # Step 4: TRL PPOTrainer reinforcement learning loop
├── inference.py            # Step 5: Interactive & CLI lyric generator with live evaluation
└── HOW_TO_USE.md           # Comprehensive system manual and execution guide
```
