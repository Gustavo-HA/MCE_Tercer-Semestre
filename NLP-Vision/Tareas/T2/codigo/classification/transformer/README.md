# mDeBERTa-v3-base Fine-tuning for Sentiment Classification

This script fine-tunes the `microsoft/mdeberta-v3-base` encoder-only transformer model on the MEIA dataset for Spanish sentiment classification (1-5 star ratings).

## Dataset

The script uses `data/classification/meia_data.json` which contains:
- **Train**: ~18,000 samples
- **Validation**: ~2,000 samples  
- **Test**: ~5,000 samples

Each sample has:
- `text`: Spanish review text
- `label`: Sentiment rating (1-5)
- `uuid`: Unique identifier

## Requirements

```bash
pip install torch transformers datasets scikit-learn numpy
```

## Offline Mode

The script is configured to work offline by using the HuggingFace cache. Make sure to download the model first:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "microsoft/mdeberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
```

## Usage

### Basic Usage (default hyperparameters)

```bash
python codigo/classification/transformer/train.py
```

### Custom Hyperparameters

```bash
python codigo/classification/transformer/train.py \
    --lr 3e-5 \
    --epochs 5 \
    --batch_size 16 \
    --grad_accum 2 \
    --max_seq_length 256 \
    --output_dir my_custom_run
```

### All Available Arguments

- `--lr`: Learning rate (default: 2e-5)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Per-device batch size (default: 8)
- `--grad_accum`: Gradient accumulation steps (default: 4)
- `--output_dir`: Custom output directory name (default: auto-generated)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--warmup_ratio`: Warmup ratio (default: 0.1)
- `--weight_decay`: Weight decay (default: 0.01)
- `--save_steps`: Save checkpoint every N steps (default: 500)
- `--eval_steps`: Evaluate every N steps (default: 500)
- `--seed`: Random seed (default: 42)

## Output

The script creates a directory in `models/` with:

```
models/mdeberta_lr2e-05_ep3_bs8x4_TIMESTAMP/
├── final_model/              # Final trained model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── ...
├── test_predictions.json     # Predictions and metrics on test set
├── logs/                     # Training logs
└── checkpoint-*/             # Intermediate checkpoints
```

## Metrics

The script computes and logs:
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1**: Weighted averages
- **Per-class metrics**: F1, precision, recall for each of the 5 classes
- **Confusion matrix**: Printed to logs

## Example Run

```bash
# Run with custom hyperparameters for cluster
python codigo/classification/transformer/train.py \
    --lr 2e-5 \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 4 \
    --max_seq_length 512 \
    --output_dir mdeberta_sentiment_v1
```

## Notes

- Uses mixed precision (fp16) if GPU is available
- Implements early stopping based on validation F1 score
- Saves the best model checkpoint during training
- Works offline by using HuggingFace local cache
