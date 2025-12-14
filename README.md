# Language Model Training with Gated Linear Attention and DPO

**Authors:** 
- Chetan Krishna Kodeboyina (NYU ID: ck3399)
- Bryce Miranda (NYU NetID: bm3986)

**Institution:** New York University

This project implements a complete pipeline for training and aligning language models using:
- **Gated Linear Attention (GLA)**: An efficient linear-complexity attention mechanism
- **Direct Preference Optimization (DPO)**: Alignment training without a separate reward model

## Project Structure

- `gla_model.py`: GLA Transformer model implementation and pretraining script
- `train_dpo.py`: DPO fine-tuning script for preference-based alignment
- `generate.py`: Command-line text generation script
- `web_interface.py`: Interactive web interface for text generation (Gradio)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Pretrain GLA Transformer

```bash
python gla_model.py \
    --hf_dataset roneneldan/TinyStories \
    --num_epochs 3 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --save_path gla_model.pth
```

### 2. Fine-tune with DPO

```bash
python train_dpo.py \
    --checkpoint gla_model.pth \
    --jsonl_file tinystories_dpo_generated.jsonl \
    --num_epochs 3 \
    --beta 0.1 \
    --learning_rate 1e-6 \
    --save_path dpo_model.pt
```

### 3. Generate Text

**Option A: Interactive Web Interface (Recommended)**

```bash
python web_interface.py
```

Then open your browser to `http://localhost:7860` for an interactive interface!

**Option B: Command Line**

```bash
python generate.py \
    --load_path dpo_model.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 100 \
    --top_p 0.95 \
    --block_size 256 \
    --n_blocks 6
```

## Key Features

- **Gated Linear Attention**: Linear complexity attention with learned gates
- **DPO Training**: Preference alignment without reward models
- **Efficient Generation**: KV caching for fast inference
- **Modern Architecture**: RMSNorm, SwiGLU, pre-normalization

## Model Architecture

- Embedding dimension: 1024 (configurable)
- Attention heads: 8 (configurable)
- Transformer blocks: 6 (configurable)
- Sequence length: 1024 tokens (configurable)

## Results

Training results, metrics, and example generations can be obtained by running the training scripts and using the inference interface.

