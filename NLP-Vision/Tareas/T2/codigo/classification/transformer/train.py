import logging
import os
from datetime import datetime
import argparse
import json

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# Offline mode for cluster Bajio
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description='Fine-tune mDeBERTa-v3-base for sentiment classification')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs (default: 3)')
parser.add_argument('--batch_size', type=int, default=8, help='Per-device batch size (default: 8)')
parser.add_argument('--grad_accum', type=int, default=4, help='Gradient accumulation steps (default: 4)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory name (default: auto-generated)')
parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length (default: 512)')
parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio (default: 0.1)')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (default: 0.01)')
parser.add_argument('--save_steps', type=int, default=500, help='Save checkpoint every N steps (default: 500)')
parser.add_argument('--eval_steps', type=int, default=500, help='Evaluate every N steps (default: 500)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
args = parser.parse_args()

# Configuration
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
grad_accum = args.grad_accum
max_seq_length = args.max_seq_length
warmup_ratio = args.warmup_ratio
weight_decay = args.weight_decay
save_steps = args.save_steps
eval_steps = args.eval_steps
seed = args.seed

# Model configuration
model_name = "microsoft/mdeberta-v3-base"
num_labels = 5  # Labels are 1-5 for sentiment classification

logger.info("="*80)
logger.info("TRAINING CONFIGURATION")
logger.info("="*80)
logger.info(f"Model: {model_name}")
logger.info(f"Number of labels: {num_labels}")
logger.info(f"Learning rate: {lr}")
logger.info(f"Epochs: {epochs}")
logger.info(f"Batch size: {batch_size}")
logger.info(f"Gradient accumulation: {grad_accum}")
logger.info(f"Effective batch size: {batch_size * grad_accum}")
logger.info(f"Max sequence length: {max_seq_length}")
logger.info(f"Warmup ratio: {warmup_ratio}")
logger.info(f"Weight decay: {weight_decay}")
logger.info(f"Random seed: {seed}")
logger.info("="*80 + "\n")

# Set random seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

# Load model from local cache
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_cache_name = model_name.replace("/", "--")
model_cache_path = f"{cache_dir}/models--{model_cache_name}"

if os.path.exists(model_cache_path):
    refs_main = f"{model_cache_path}/refs/main"
    if os.path.exists(refs_main):
        with open(refs_main, 'r') as f:
            commit_hash = f.read().strip()
        model_path = f"{model_cache_path}/snapshots/{commit_hash}"
        logger.info(f"Found cached model at: {model_path}")
    else:
        logger.error(f"Cache exists but no 'main' ref found at: {refs_main}")
        model_path = model_name  # Fall back to online mode
else:
    logger.warning("Downloading from HuggingFace Hub...")
    model_path = model_name

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
    )
    logger.info("Model and tokenizer loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Load dataset
data_path = "./data/classification/meia_data.json"
logger.info(f"Loading dataset from: {data_path}")

try:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_list(data['train'])
    val_dataset = Dataset.from_list(data['validation'])
    test_dataset = Dataset.from_list(data['test'])
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise

# Preprocessing function
def preprocess_function(examples):
    """
    Tokenize the text and adjust labels.
    """
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_seq_length,
        padding=False,  # Usaremos DataCollator para padding dinámico
    )
    return tokenized

# Tokenize datasets
logger.info("Tokenizing datasets...")
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'uuid'])
val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'uuid'])
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=['text', 'uuid'])
logger.info("Tokenization complete!")

# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define metrics
def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1-score
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Also compute per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    # Add per-class metrics
    for i in range(num_labels):
        metrics[f'precision_class_{i+1}'] = precision_per_class[i]
        metrics[f'recall_class_{i+1}'] = recall_per_class[i]
        metrics[f'f1_class_{i+1}'] = f1_per_class[i]
    
    return metrics

# Generate output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if args.output_dir:
    output_dir = f"models/{args.output_dir}"
else:
    # Auto-generate directory name with hyperparameters
    run_name = f"mdeberta_lr{lr}_ep{epochs}_bs{batch_size}x{grad_accum}_{timestamp}"
    output_dir = f"models/classification/{run_name}"

logger.info(f"Output directory: {output_dir}")

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    warmup_ratio=warmup_ratio,
    weight_decay=weight_decay,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=seed,
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    report_to="none",  # Disable wandb/tensorboard
    remove_unused_columns=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# GPU memory stats before training
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of available GPUs: {num_gpus}")
    
    start_gpu_memory = {}
    max_memory = {}
    
    for device_id in range(num_gpus):
        gpu_stats = torch.cuda.get_device_properties(device_id)
        start_mem = round(torch.cuda.max_memory_reserved(device_id) / 1024 / 1024 / 1024, 3)
        max_mem = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        
        start_gpu_memory[device_id] = start_mem
        max_memory[device_id] = max_mem
        
        logger.info(f"GPU {device_id}: {gpu_stats.name}. Max memory = {max_mem} GB.")
        logger.info(f"GPU {device_id}: {start_mem} GB of memory reserved.")
else:
    logger.info("No GPU available, training on CPU")

# Train the model
logger.info("Starting training...")
trainer_stats = trainer.train()
logger.info(f"Training completed! Metrics: {trainer_stats.metrics}")

# Evaluate on validation set
logger.info("Evaluating on validation set...")
val_metrics = trainer.evaluate(eval_dataset=val_dataset)
logger.info(f"Validation metrics: {val_metrics}")

# Evaluate on test set
logger.info("Evaluating on test set...")
test_metrics = trainer.evaluate(eval_dataset=test_dataset)
logger.info(f"Test metrics: {test_metrics}")

# Get predictions for confusion matrix
predictions_output = trainer.predict(test_dataset)
predictions = np.argmax(predictions_output.predictions, axis=-1)
labels = predictions_output.label_ids

# Compute and log confusion matrix
cm = confusion_matrix(labels, predictions)
logger.info("Confusion Matrix (rows=true, cols=predicted):")
logger.info(f"\n{cm}")

# Save the final model
final_output_dir = f"{output_dir}/final_model"
model.save_pretrained(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
logger.info(f"✓ Model saved successfully at: {final_output_dir}")

# Save test predictions
predictions_file = f"{output_dir}/test_predictions.json"
with open(predictions_file, 'w', encoding='utf-8') as f:
    json.dump({
        'predictions': [int(p) + 1 for p in predictions],  # Convert back to 1-5
        'labels': [int(label) + 1 for label in labels],  # Convert back to 1-5
        'metrics': test_metrics
    }, f, indent=2)
logger.info(f"Test predictions saved to: {predictions_file}")

# Report training time and GPU memory usage
logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
logger.info(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")

# Log memory stats for all GPUs
if torch.cuda.is_available():
    for device_id in range(num_gpus):
        used_memory = round(torch.cuda.max_memory_reserved(device_id) / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory[device_id], 3)
        used_percentage = round(used_memory / max_memory[device_id] * 100, 3)
        training_percentage = round(used_memory_for_training / max_memory[device_id] * 100, 3)
        
        logger.info(f"GPU {device_id} - Peak reserved memory = {used_memory} GB.")
        logger.info(f"GPU {device_id} - Peak reserved memory for training = {used_memory_for_training} GB.")
        logger.info(f"GPU {device_id} - Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"GPU {device_id} - Peak reserved memory for training % of max memory = {training_percentage} %.")

logger.info("="*80)
logger.info("TRAINING COMPLETE!")
logger.info("="*80)
