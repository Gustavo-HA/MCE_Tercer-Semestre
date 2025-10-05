# Mistral Fine-Tuning, Inference & Perplexity

This directory contains scripts for fine-tuning the Mistral-7B-Instruct-v0.3 model using LoRA (Low-Rank Adaptation), performing inference with the fine-tuned model, and calculating perplexity metrics.

## 📁 Directory Structure

```
mistral/
├── fine-tune_mistral.py          # Fine-tuning script
├── inference_mistral.py          # Inference/generation script
├── calculate_perplexity.py       # Perplexity calculation script
├── mistral_fine-tuning.sh        # SLURM job script for fine-tuning
├── mistral_inference.sh          # SLURM job script for inference
├── mistral_perplexity.sh         # SLURM job script for perplexity
└── utils/
    └── perplexity_unsloth.py     # Perplexity calculation utilities
```

## 🚀 Features

- **LoRA Fine-Tuning**: Efficient fine-tuning using Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **4-bit Quantization**: Memory-efficient training with 4-bit quantized models via Unsloth
- **Offline Mode**: Support for offline training (no internet required after model download)
- **Alpaca Format**: Uses Alpaca instruction format for training and inference
- **Flexible Configuration**: Command-line arguments for all hyperparameters
- **Perplexity Evaluation**: Compare fine-tuned model against base model
- **SLURM Integration**: Ready-to-use SLURM job scripts for HPC clusters

## 📋 Requirements

The scripts use the following key dependencies:
- `unsloth` - Fast and memory-efficient fine-tuning
- `torch` - PyTorch deep learning framework
- `transformers` - HuggingFace transformers
- `trl` - Transformer Reinforcement Learning (for SFTTrainer)
- `datasets` - HuggingFace datasets

Install via the project's `requirements.txt` or `pyproject.toml`.

## 🎯 1. Fine-Tuning

### Script: `fine-tune_mistral.py`

Fine-tunes Mistral-7B-Instruct-v0.3 with LoRA adapters on custom dataset.

#### Usage

```bash
python codigo/generative/mistral/fine-tune_mistral.py \
    --lr 2e-4 \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lora_r 32 \
    --max_seq_length 2048 \
    --output_dir "custom_model_name"
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--lr` | float | `2e-4` | Learning rate |
| `--epochs` | int | `3` | Number of training epochs |
| `--batch_size` | int | `2` | Per-device batch size |
| `--grad_accum` | int | `8` | Gradient accumulation steps |
| `--lora_r` | int | `32` | LoRA rank (higher = more parameters) |
| `--max_seq_length` | int | `2048` | Maximum sequence length |
| `--output_dir` | str | Auto | Output directory name (auto-generated if not provided) |

#### Output

The model is saved to `models/<run_name>/final_model/` where `<run_name>` includes hyperparameters and timestamp:
```
models/mistral_lr0.0002_ep3_bs2x8_r32_20251004_133739/final_model/
```

#### SLURM Usage

```bash
sbatch codigo/generative/mistral/mistral_fine-tuning.sh
```

Edit the shell script to modify:
- SLURM partition, memory, time limits
- Hyperparameters passed to the Python script
- Email notifications

### Key Features

- **LoRA Configuration**: Targets all attention and MLP projection layers
- **Memory Efficient**: Uses gradient checkpointing and 4-bit quantization
- **Alpaca Chat Template**: Formats data with instruction-input-output structure
- **Auto Naming**: Creates timestamped output directories with hyperparameter info
- **GPU Monitoring**: Logs memory usage across all GPUs

## 🎨 2. Inference / Text Generation

### Script: `inference_mistral.py`

Generate lyrics or text using the fine-tuned model.

#### Usage

```bash
python codigo/generative/mistral/inference_mistral.py \
    --model_dir models/mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706/final_model \
    --artist "Kendrick Lamar" \
    --max_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --repetition_penalty 1.1
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_dir` | str | **Required** | Path to fine-tuned model directory |
| `--artist` | str | `"Kendrick Lamar"` | Artist style for generation |
| `--prompt` | str | `None` | Custom instruction (overrides default) |
| `--max_tokens` | int | `512` | Maximum tokens to generate |
| `--temperature` | float | `0.7` | Sampling temperature (higher = more random) |
| `--top_p` | float | `0.9` | Nucleus sampling threshold |
| `--repetition_penalty` | float | `1.1` | Penalty for repeating tokens |

#### Output

Generated lyrics are printed to console and saved to:
```
data/text_gen/inferences/generated_lyrics_<artist>.txt
```

#### Example with Custom Prompt

```bash
python codigo/generative/mistral/inference_mistral.py \
    --model_dir models/mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706/final_model \
    --prompt "Write a song about overcoming challenges and finding strength"
```

#### SLURM Usage

```bash
sbatch codigo/generative/mistral/mistral_inference.sh
```

## 📊 3. Perplexity Calculation

### Script: `calculate_perplexity.py`

Evaluate model performance using perplexity metric on test dataset.

#### Usage

```bash
# Evaluate fine-tuned model only
python codigo/generative/mistral/calculate_perplexity.py \
    --model_dir models/mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706/final_model \
    --test_dataset data/text_gen/test_lyrics_alpaca.json

# Compare against base model
python codigo/generative/mistral/calculate_perplexity.py \
    --model_dir models/mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706/final_model \
    --test_dataset data/text_gen/test_lyrics_alpaca.json \
    --compare_base
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_dir` | str | Path to model | Path to fine-tuned model directory |
| `--test_dataset` | str | `data/text_gen/test_lyrics_alpaca.json` | Test dataset path |
| `--max_seq_length` | int | `2048` | Maximum sequence length |
| `--dtype` | str | `None` | Data type (`float16`, `bfloat16`, or None) |
| `--load_in_4bit` | bool | `True` | Use 4-bit quantization |
| `--compare_base` | bool | `False` | Also evaluate base Mistral model |
| `--base_model_name` | str | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` | Base model to compare |

#### Output

Results are saved to: `<model_dir>/perplexity_results.txt`

Example output:
```
================================================================================
PERPLEXITY RESULTS
================================================================================

Fine-tuned Model: mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706
Model Directory: models/mistral_lr0.0002_ep3_bs2x8_r32_20251003_124706/final_model
Test Dataset: data/text_gen/test_lyrics_alpaca.json
Number of examples: 150

Fine-tuned Model Perplexity: 8.5234
Base Model Perplexity: 12.3456

Improvement: 30.96%
================================================================================
```

#### SLURM Usage

```bash
sbatch codigo/generative/mistral/mistral_perplexity.sh
```

### Perplexity Utility: `utils/perplexity_unsloth.py`

Contains helper functions:
- `ppl_model()`: Calculate perplexity with sliding window for long sequences
- `add_to_comparison()`: Add model results to comparison tracker
- `print_model_comparison()`: Display comparison table of multiple models

**Perplexity Calculation Details:**
- Uses sliding window approach (stride=512, max_length=2048)
- Processes each example individually
- Handles sequences longer than max context length
- Computes negative log-likelihood with proper attention masking

## 🔧 Configuration

### Offline Mode

All scripts support offline mode by setting environment variables:
```python
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
```

The scripts automatically detect cached models in `~/.cache/huggingface/hub/`.

### Data Format

**Input**: Alpaca JSON format
```json
{
    "instruction": "You're a hip-hop artist, create song lyrics.",
    "input": "Write lyrics in the style of Kendrick Lamar",
    "output": "Generated lyrics..."
}
```

**Training Data**: `data/text_gen/train_lyrics_alpaca.json`  
**Test Data**: `data/text_gen/test_lyrics_alpaca.json`

### LoRA Configuration

Default LoRA settings in `fine-tune_mistral.py`:
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"       # MLP
]
lora_alpha = lora_r * 2
lora_dropout = 0.00
```

## 📈 Training Tips

1. **Memory Management**
   - Use gradient accumulation to simulate larger batch sizes
   - Effective batch size = `batch_size × grad_accum`
   - For 24GB GPU: `batch_size=2`, `grad_accum=8` works well

2. **Hyperparameter Tuning**
   - Start with default values
   - Increase `lora_r` (16, 32, 64) for more capacity
   - Adjust `lr` if loss plateaus or diverges
   - More epochs (3-5) for better convergence

3. **Monitoring**
   - Check GPU memory usage in logs
   - Review training metrics (loss, runtime)
   - Validate with perplexity on test set

## 🐛 Troubleshooting

### Model Not Found in Cache
```bash
# Download model when you have internet:
python -c "from unsloth import FastLanguageModel; \
FastLanguageModel.from_pretrained('unsloth/mistral-7b-instruct-v0.3-bnb-4bit', \
max_seq_length=2048, load_in_4bit=True)"
```

### Out of Memory
- Reduce `batch_size` to 1
- Reduce `max_seq_length` to 1024
- Increase gradient accumulation
- Enable gradient checkpointing (already enabled)

### Poor Generation Quality
- Increase training epochs
- Increase LoRA rank
- Adjust sampling parameters (temperature, top_p)
- Check if model converged (review loss)

## 📝 Example Workflow

```bash
# 1. Fine-tune model
python codigo/generative/mistral/fine-tune_mistral.py \
    --lr 2e-4 --epochs 3 --batch_size 2 --grad_accum 8 --lora_r 32

# 2. Calculate perplexity (compare with base)
python codigo/generative/mistral/calculate_perplexity.py \
    --model_dir models/mistral_lr0.0002_ep3_bs2x8_r32_<timestamp>/final_model \
    --compare_base

# 3. Generate lyrics
python codigo/generative/mistral/inference_mistral.py \
    --model_dir models/mistral_lr0.0002_ep3_bs2x8_r32_<timestamp>/final_model \
    --artist "Kendrick Lamar" \
    --temperature 0.8
```

## 📚 References

- **Unsloth**: https://github.com/unslothai/unsloth
- **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Mistral-7B**: https://mistral.ai/
- **Alpaca Format**: Instruction-following format inspired by Stanford Alpaca

## 📄 License

Part of the NLP-Vision course project (CIMAT, Tercer Semestre).

---

**Note**: Replace `<timestamp>` in examples with actual timestamp from your training run.
