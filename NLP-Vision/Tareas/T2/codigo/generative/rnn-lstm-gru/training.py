import torch
import torch.nn as nn
from rnntype import RNNType
from lyrics_dataset import LyricsDataset
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from pathlib import Path
import logging
import argparse
import os
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=5.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0), device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, hidden = model(inputs, hidden)
        
        # Only use the last output for prediction
        outputs = outputs[:, -1, :]  # (batch_size, vocab_size)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Initialize hidden state
            hidden = model.init_hidden(inputs.size(0), device)
            
            # Forward pass
            outputs, hidden = model(inputs, hidden)
            outputs = outputs[:, -1, :]
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train RNN for lyrics text generation')
    
    # Model architecture
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'],
                        help='Type of RNN: RNN, LSTM, or GRU (default: LSTM)')
    parser.add_argument('--level', type=str, default='char', choices=['char', 'word'],
                        help='Tokenization level: char or word (default: char)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of token embeddings (default: 128)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Dimension of RNN hidden state (default: 256)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of RNN layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout probability (default: 0.3)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Sequence length (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='Learning rate (default: 0.002)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--clip_grad', type=float, default=5.0,
                        help='Gradient clipping value (default: 5.0)')
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum change to qualify as improvement (default: 0.001)')
    
    # Data and output
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to output directory (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Paths
    if args.data_dir is None:
        base_dir = Path(__file__).parent.parent.parent.parent
        data_dir = base_dir / "data" / "text_gen"
    else:
        data_dir = Path(args.data_dir)
    
    if args.output_dir is None:
        base_dir = Path(__file__).parent.parent.parent.parent
        output_dir = base_dir / "models" / args.rnn_type / args.level
        os.makedirs(output_dir, exist_ok=True)        
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load vocabulary
    logger.info("\nLoading vocabulary...")
    vocab_file = data_dir / f"vocab_{args.level}.json"
    
    if not vocab_file.exists():
        logger.error(f"Vocabulary file not found: {vocab_file}")
        return
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    if args.level == 'char':
        token_to_idx = vocab_data['char_to_idx']
    else:  # word level
        token_to_idx = vocab_data['word_to_idx']
    
    vocab_size = vocab_data['vocab_size']
    logger.info(f"Vocabulary size: {vocab_size:,}")
    
    # Load text data
    logger.info("\nLoading training data...")
    train_file = data_dir / "train_lyrics.txt"
    
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        return
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    if args.level == 'char':
        logger.info(f"Training text length: {len(train_text):,} characters")
    else:
        logger.info(f"Training text length: {len(train_text.split()):,} words")
    
    # Load validation data
    logger.info("\nLoading validation data...")
    val_file = data_dir / "test_lyrics.txt"
    
    if not val_file.exists():
        logger.warning(f"Validation file not found: {val_file}")
        logger.warning("Training without validation set!")
        val_text = None
    else:
        with open(val_file, 'r', encoding='utf-8') as f:
            val_text = f.read()
        
        if args.level == 'char':
            logger.info(f"Validation text length: {len(val_text):,} characters")
        else:
            logger.info(f"Validation text length: {len(val_text.split()):,} words")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = LyricsDataset(train_text, token_to_idx, seq_length=args.seq_length, level=args.level)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    logger.info(f"Number of training sequences: {len(train_dataset):,}")
    logger.info(f"Number of batches per epoch: {len(train_loader):,}")
    
    # Create validation dataset if available
    val_loader = None
    if val_text is not None:
        val_dataset = LyricsDataset(val_text, token_to_idx, seq_length=args.seq_length, level=args.level)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        logger.info(f"Number of validation sequences: {len(val_dataset):,}")
        logger.info(f"Number of validation batches: {len(val_loader):,}")
    
    # Create model
    logger.info("\nInitializing model...")
    model = RNNType(
        rnn_type=args.rnn_type,
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.clip_grad)
        train_perplexity = math.exp(train_loss)
        logger.info(f"Training Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
        
        # Validate
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            val_perplexity = math.exp(val_loss)
            logger.info(f"Validation Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
            
            # Update learning rate scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Save best model based on validation loss
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'vocab_size': vocab_size,
                    'embedding_dim': args.embedding_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'level': args.level,
                    'rnn_type': args.rnn_type
                }
                torch.save(checkpoint, output_dir / 'best_model.pt')
                logger.info(f"Saved best model (val_loss: {val_loss:.4f}, val_perplexity: {val_perplexity:.2f})")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if args.early_stopping and epochs_without_improvement >= args.patience:
                logger.info(f"\n Early stopping triggered after {epoch + 1} epochs")
                logger.info(f"No improvement for {args.patience} consecutive epochs")
                break
        else:
            # Usaremos train loss
            scheduler.step(train_loss)
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'vocab_size': vocab_size,
                    'embedding_dim': args.embedding_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'level': args.level,
                    'rnn_type': args.rnn_type
                }
                torch.save(checkpoint, output_dir / 'best_model.pt')
                logger.info(f"Saved best model (loss: {train_loss:.4f})")
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*80)
    best_perplexity = math.exp(best_val_loss)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation perplexity: {best_perplexity:.2f}")
    logger.info(f"Model saved to: {output_dir / 'best_model.pt'}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
