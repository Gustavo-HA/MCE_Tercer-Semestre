# Generación de texto RNN

This directory contains scripts for training, inference, and perplexity evaluation with simple RNN models for lyrics text generation at both character and word levels.

## Scripts

- **`rnn_training.py`**: Train RNN models at character or word level
- **`rnn_inference.py`**: Generate lyrics using trained models
- **`rnn_perplexity.py`**: Calculate perplexity of trained models on test set
- **`rnn_training.sh`**: SLURM job script for training
- **`rnn_perplexity.sh`**: SLURM job script for perplexity calculation

## Features

- **Dual-level tokenization**: Train at character-level or word-level
- **Configurable architecture**: Customize embedding size, hidden dimensions, layers, and dropout
- **Flexible training**: Control batch size, sequence length, learning rate, and epochs
- **Separate inference**: Dedicated script for text generation with trained models
- **Checkpointing**: Automatically saves the best model based on training loss

## Training Usage

### Train Character-Level Model

```bash
python rnn_training.py --level char \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.3 \
    --batch_size 64 \
    --seq_length 100 \
    --learning_rate 0.002 \
    --num_epochs 20
```

### Train Word-Level Model

```bash
python rnn_training.py --level word \
    --embedding_dim 256 \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.3 \
    --batch_size 32 \
    --seq_length 50 \
    --learning_rate 0.001 \
    --num_epochs 20
```

## Arguments

### Model Architecture
- `--level`: Tokenization level (`char` or `word`, default: `char`)
- `--embedding_dim`: Dimension of token embeddings (default: `128`)
- `--hidden_dim`: Dimension of LSTM hidden state (default: `256`)
- `--num_layers`: Number of LSTM layers (default: `2`)
- `--dropout`: Dropout probability (default: `0.3`)

### Training Parameters
- `--batch_size`: Batch size (default: `64`)
- `--seq_length`: Sequence length (default: `100`)
- `--learning_rate`: Learning rate (default: `0.002`)
- `--num_epochs`: Number of epochs (default: `20`)
- `--clip_grad`: Gradient clipping value (default: `5.0`)

### Data and Output
- `--data_dir`: Path to data directory (default: auto-detect)
- `--output_dir`: Path to output directory (default: auto-detect)

## Inference Usage

### Generate Lyrics with Trained Model

```bash
# Character-level generation
python rnn_inference.py --model_path models/rnn/char/best_model.pt \
    --level char \
    --length 500 \
    --temperature 0.8 \
    --num_samples 3

# Word-level generation
python rnn_inference.py --model_path models/rnn/word/best_model.pt \
    --level word \
    --length 300 \
    --temperature 1.0 \
    --num_samples 5 \
    --output_file data/text_gen/inferences/generated_lyrics.txt
```

### Inference Arguments

- `--model_path`: Path to trained model checkpoint (**required**)
- `--level`: Tokenization level (`char` or `word`, **required**)
- `--length`: Number of tokens to generate (default: `500`)
- `--temperature`: Sampling temperature (default: `0.8`)
- `--num_samples`: Number of samples to generate (default: `1`)
- `--vocab_path`: Path to vocabulary file (default: auto-detect)
- `--output_file`: Path to save generated text (default: print to console)

### Start Prompt

All generations start with the special token: **`<|song_start|>`**

This ensures generated lyrics follow the training data format.

## Generation Parameters

## Perplexity Calculation

### Calculate Model Perplexity on Test Set

The perplexity script evaluates how well the trained model predicts the test data. Lower perplexity indicates better model performance.

```bash
# Character-level model
python codigo/generative/rnn/rnn_perplexity.py \
    --model_path models/rnn/char/best_model.pt \
    --level char \
    --batch_size 64 \
    --seq_length 100 \
    --method dataloader

# Word-level model
python codigo/generative/rnn/rnn_perplexity.py \
    --model_path models/rnn/word/best_model.pt \
    --level word \
    --batch_size 64 \
    --seq_length 100 \
    --method dataloader

# Using sliding window method (alternative approach)
python codigo/generative/rnn/rnn_perplexity.py \
    --model_path models/rnn/char/best_model.pt \
    --level char \
    --method sliding_window \
    --stride 50
```

### Perplexity Arguments

- `--model_path`: Path to trained model checkpoint (**required**)
- `--level`: Tokenization level (`char` or `word`, **required**)
- `--test_file`: Path to test file (default: `data/text_gen/test_lyrics.txt`)
- `--vocab_path`: Path to vocabulary file (default: auto-detect)
- `--batch_size`: Batch size for evaluation (default: `64`)
- `--seq_length`: Sequence length (default: `100`)
- `--method`: Calculation method (`dataloader` or `sliding_window`, default: `dataloader`)
- `--stride`: Stride for sliding window method (default: `50`)
- `--output_file`: Path to save results (default: save to model directory)

### Calculation Methods

**Dataloader Method** (recommended):
- Creates overlapping sequences from test data
- Efficient batch processing
- Consistent with training procedure
- Faster for large datasets

**Sliding Window Method**:
- Processes text with a sliding window
- More flexible stride control
- Useful for custom evaluation scenarios
- Can be slower but more interpretable

### SLURM Usage

Calculate perplexity for both char and word models:

```bash
sbatch codigo/generative/rnn/rnn_perplexity.sh
```

Edit the shell script to modify model paths and evaluation parameters.

### Output

Results are saved to: `<model_directory>/perplexity_results_<level>.txt`

Example output:
```
================================================================================
RNN PERPLEXITY RESULTS - CHAR LEVEL
================================================================================

Model: models/rnn/char/best_model.pt
Test file: data/text_gen/test_lyrics.txt
Tokenization level: char
Vocabulary size: 157
Sequence length: 100
Method: dataloader
Batch size: 64
Number of test sequences: 6,048

--------------------------------------------------------------------------------
MODEL CONFIGURATION
--------------------------------------------------------------------------------
Embedding dimension: 128
Hidden dimension: 256
Number of layers: 2
Dropout: 0.3
Model parameters: 291,741
Training epochs: 20
Training loss: 1.2345

--------------------------------------------------------------------------------
PERPLEXITY RESULTS
--------------------------------------------------------------------------------
Average Loss (Cross-Entropy): 1.4567
Perplexity: 4.2901
================================================================================
```

### Interpreting Perplexity

- **Lower perplexity = Better model**: The model is more confident in its predictions
- **Perplexity represents uncertainty**: It shows the average number of choices the model considers when predicting the next token
- **Typical ranges**:
  - Character-level: 2-10 (excellent), 10-50 (good), >50 (needs improvement)
  - Word-level: 50-200 (excellent), 200-500 (good), >500 (needs improvement)
- **Compare**: Evaluate multiple models to see which architecture/hyperparameters work best

## Generation Parameters

## Output

Models are saved to:
- Character-level: `models/rnn/char/best_model.pt`
- Word-level: `models/rnn/word/best_model.pt`

## Recommendations

### Character-Level
- Best for: Capturing spelling patterns, inventing words, learning syntax from scratch
- Smaller vocabulary (~157 tokens)
- Longer sequences needed for context
- Slower to generate but more creative

### Word-Level
- Best for: Maintaining semantic meaning, faster generation
- Larger vocabulary (~19,107 words)
- Requires more memory
- Better coherence but less creative spelling

## Quick Start

Train both models with default settings:

```bash
# Character-level
python rnn_training.py --level char

# Word-level  
python rnn_training.py --level word --batch_size 32 --seq_length 50 --embedding_dim 256 --hidden_dim 512
```
