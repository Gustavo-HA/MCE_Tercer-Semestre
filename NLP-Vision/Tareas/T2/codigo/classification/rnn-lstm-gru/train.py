import logging
import os
from datetime import datetime
import argparse
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from rnntype import RNNType
from dataset import TextClassificationDataset

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description='Train RNN/LSTM/GRU for sentiment classification')
parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'],
                    help='Type of RNN to use (default: LSTM)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs (default: 10)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension (default: 128)')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension (default: 256)')
parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers (default: 2)')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (default: 0.3)')
parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length (default: 128)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory name (default: auto-generated)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--vocab_path', type=str, default='./data/classification/meia_data_vocab.json',
                    help='Path to vocabulary file')
parser.add_argument('--train_path', type=str, default='./data/classification/meia_data_train.csv',
                    help='Path to training CSV file')
parser.add_argument('--val_path', type=str, default='./data/classification/meia_data_val.csv',
                    help='Path to validation CSV file')
parser.add_argument('--test_path', type=str, default='./data/classification/meia_data_test.csv',
                    help='Path to test CSV file')
parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (default: 3)')
args = parser.parse_args()

# Configuration
rnn_type = args.rnn_type
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
num_layers = args.num_layers
dropout = args.dropout
max_length = args.max_length
seed = args.seed
patience = args.patience

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info("="*80)
logger.info("TRAINING CONFIGURATION")
logger.info("="*80)
logger.info(f"RNN Type: {rnn_type}")
logger.info(f"Device: {device}")
logger.info(f"Learning rate: {lr}")
logger.info(f"Epochs: {epochs}")
logger.info(f"Batch size: {batch_size}")
logger.info(f"Embedding dimension: {embedding_dim}")
logger.info(f"Hidden dimension: {hidden_dim}")
logger.info(f"Number of layers: {num_layers}")
logger.info(f"Dropout: {dropout}")
logger.info(f"Max sequence length: {max_length}")
logger.info(f"Random seed: {seed}")
logger.info(f"Early stopping patience: {patience}")
logger.info("="*80 + "\n")

# Set random seed for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load datasets
logger.info("Loading datasets...")
train_dataset = TextClassificationDataset(args.train_path, args.vocab_path, max_length)
val_dataset = TextClassificationDataset(args.val_path, args.vocab_path, max_length)
test_dataset = TextClassificationDataset(args.test_path, args.vocab_path, max_length)

vocab_size = train_dataset.get_vocab_size()
num_classes = train_dataset.get_num_classes()

logger.info(f"Train samples: {len(train_dataset)}")
logger.info(f"Validation samples: {len(val_dataset)}")
logger.info(f"Test samples: {len(test_dataset)}")
logger.info(f"Vocabulary size: {vocab_size}")
logger.info(f"Number of classes: {num_classes}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize model
logger.info(f"\nInitializing {rnn_type} model...")
model = RNNType(
    rnn_type=rnn_type,
    vocab_size=vocab_size,
    num_classes=num_classes,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total parameters: {total_params:,}")
logger.info(f"Trainable parameters: {trainable_params:,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

# Generate output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if args.output_dir:
    output_dir = f"models/{args.output_dir}"
else:
    # Auto-generate directory name with hyperparameters
    run_name = f"{rnn_type.lower()}_emb{embedding_dim}_h{hidden_dim}_l{num_layers}_{timestamp}"
    output_dir = f"models/{run_name}"

os.makedirs(output_dir, exist_ok=True)
logger.info(f"Output directory: {output_dir}")

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels

# GPU memory stats before training
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    logger.info(f"\nNumber of available GPUs: {num_gpus}")
    
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
    logger.info("\nNo GPU available, training on CPU")

# Training loop
logger.info("\n" + "="*80)
logger.info("STARTING TRAINING")
logger.info("="*80)

best_val_f1 = 0
best_epoch = 0
patience_counter = 0
training_history = []

start_time = time.time()

for epoch in range(epochs):
    epoch_start_time = time.time()
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
        model, val_loader, criterion, device
    )
    
    epoch_time = time.time() - epoch_start_time
    
    # Log metrics
    logger.info(f"\nEpoch {epoch + 1}/{epochs} - {epoch_time:.2f}s")
    logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    logger.info(f"  Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
    
    # Save training history
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'epoch_time': epoch_time
    })
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0
        
        # Save model
        checkpoint_path = f"{output_dir}/best_model.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_acc': val_acc,
            'config': {
                'rnn_type': rnn_type,
                'vocab_size': vocab_size,
                'num_classes': num_classes,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout,
                'max_length': max_length,
            }
        }, checkpoint_path)
        logger.info(f"  ✓ New best model saved (F1: {val_f1:.4f})")
    else:
        patience_counter += 1
        logger.info(f"  No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

total_time = time.time() - start_time
logger.info(f"\n{'='*80}")
logger.info("TRAINING COMPLETED")
logger.info(f"{'='*80}")
logger.info(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
logger.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")

# Load best model for final evaluation
logger.info("\nLoading best model for final evaluation...")
checkpoint = torch.load(f"{output_dir}/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on validation set
logger.info("\n" + "="*80)
logger.info("VALIDATION SET EVALUATION")
logger.info("="*80)
val_loss, val_acc, val_precision, val_recall, val_f1, val_preds, val_labels = evaluate(
    model, val_loader, criterion, device
)
logger.info(f"Loss: {val_loss:.4f}")
logger.info(f"Accuracy: {val_acc:.4f}")
logger.info(f"Precision: {val_precision:.4f}")
logger.info(f"Recall: {val_recall:.4f}")
logger.info(f"F1 Score: {val_f1:.4f}")

# Evaluate on test set
logger.info("\n" + "="*80)
logger.info("TEST SET EVALUATION")
logger.info("="*80)
test_loss, test_acc, test_precision, test_recall, test_f1, test_preds, test_labels = evaluate(
    model, test_loader, criterion, device
)
logger.info(f"Loss: {test_loss:.4f}")
logger.info(f"Accuracy: {test_acc:.4f}")
logger.info(f"Precision: {test_precision:.4f}")
logger.info(f"Recall: {test_recall:.4f}")
logger.info(f"F1 Score: {test_f1:.4f}")

# Compute and log confusion matrix
cm = confusion_matrix(test_labels, test_preds)
logger.info("\nConfusion Matrix (rows=true, cols=predicted):")
logger.info(f"\n{cm}")

# Per-class metrics
logger.info("\nPer-Class Metrics:")
per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
    test_labels, test_preds, average=None, zero_division=0
)
for i in range(num_classes):
    logger.info(f"  Class {i}: Precision={per_class_precision[i]:.4f}, "
                f"Recall={per_class_recall[i]:.4f}, F1={per_class_f1[i]:.4f}, "
                f"Support={support[i]}")

# Save training history
history_path = f"{output_dir}/training_history.json"
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(training_history, f, indent=2)
logger.info(f"\nTraining history saved to: {history_path}")

# Save test results
results = {
    'best_epoch': best_epoch,
    'best_val_f1': best_val_f1,
    'test_metrics': {
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1
    },
    'per_class_metrics': {
        f'class_{i}': {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1': float(per_class_f1[i]),
            'support': int(support[i])
        } for i in range(num_classes)
    },
    'confusion_matrix': cm.tolist(),
    'config': checkpoint['config']
}

results_path = f"{output_dir}/test_results.json"
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
logger.info(f"Test results saved to: {results_path}")

# Save predictions
predictions_path = f"{output_dir}/test_predictions.json"
with open(predictions_path, 'w', encoding='utf-8') as f:
    json.dump({
        'predictions': [int(p) for p in test_preds],
        'labels': [int(label) for label in test_labels]
    }, f, indent=2)
logger.info(f"Test predictions saved to: {predictions_path}")

# GPU memory stats
if torch.cuda.is_available():
    logger.info("\n" + "="*80)
    logger.info("GPU MEMORY USAGE")
    logger.info("="*80)
    for device_id in range(num_gpus):
        used_memory = round(torch.cuda.max_memory_reserved(device_id) / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory[device_id], 3)
        used_percentage = round(used_memory / max_memory[device_id] * 100, 3)
        training_percentage = round(used_memory_for_training / max_memory[device_id] * 100, 3)
        
        logger.info(f"GPU {device_id} - Peak reserved memory = {used_memory} GB.")
        logger.info(f"GPU {device_id} - Peak reserved memory for training = {used_memory_for_training} GB.")
        logger.info(f"GPU {device_id} - Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"GPU {device_id} - Peak reserved memory for training % of max memory = {training_percentage} %.")

logger.info("\n" + "="*80)
logger.info("TRAINING COMPLETE!")
logger.info("="*80)
logger.info(f"Best model saved to: {output_dir}/best_model.pt")
logger.info(f"Test F1 Score: {test_f1:.4f}")
logger.info(f"Test Accuracy: {test_acc:.4f}")
