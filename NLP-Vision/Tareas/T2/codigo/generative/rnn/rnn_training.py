"""
RNN for lyrics text generation.
Trains an RNN model to generate lyrics at character or word level.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import logging
import argparse
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class LyricsDataset(Dataset):
    """Dataset for text generation at character or word level."""
    
    def __init__(self, text, token_to_idx, seq_length=100, level='char'):
        """
        Args:
            text: Full text string
            token_to_idx: Dictionary mapping tokens (chars or words) to indices
            seq_length: Length of input sequences
            level: 'char' or 'word' tokenization level
        """
        self.text = text
        self.token_to_idx = token_to_idx
        self.seq_length = seq_length
        self.level = level
        
        # Tokenize and encode text
        if level == 'char':
            self.tokens = list(text)
        else:  # word level
            self.tokens = text.split()
        
        self.encoded = [token_to_idx.get(token, 0) for token in self.tokens]
        
    def __len__(self):
        return len(self.encoded) - self.seq_length
    
    def __getitem__(self, idx):
        # Input: seq_length characters
        # Target: next character after the sequence
        input_seq = self.encoded[idx:idx + self.seq_length]
        target = self.encoded[idx + self.seq_length]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class SimpleRNN(nn.Module):
    """Simple RNN model for text generation (character or word level)."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of RNN hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(SimpleRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        rnn_out, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_length, hidden_dim)
        rnn_out = self.dropout(rnn_out)
        output = self.fc(rnn_out)  # (batch_size, seq_length, vocab_size)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h_0


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
        output_dir = base_dir / "models" / "rnn" / args.level
        os.makedirs(output_dir, exist_ok=True)        
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    logger.info("="*80)
    logger.info(f"TRAINING {args.level.upper()}-LEVEL RNN FOR LYRICS GENERATION")
    logger.info("="*80)
    logger.info("\nHyperparameters:")
    logger.info(f"  Level: {args.level}")
    logger.info(f"  Embedding dim: {args.embedding_dim}")
    logger.info(f"  Hidden dim: {args.hidden_dim}")
    logger.info(f"  Num layers: {args.num_layers}")
    logger.info(f"  Dropout: {args.dropout}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Sequence length: {args.seq_length}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Gradient clipping: {args.clip_grad}")
    
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
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = LyricsDataset(train_text, token_to_idx, seq_length=args.seq_length, level=args.level)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    logger.info(f"Number of training sequences: {len(train_dataset):,}")
    logger.info(f"Number of batches per epoch: {len(train_loader):,}")
    
    # Create model
    logger.info("\nInitializing model...")
    model = SimpleRNN(
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
    
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        logger.info("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.clip_grad)
        logger.info(f"Training Loss: {train_loss:.4f}")
        
        # Update learning rate
        scheduler.step(train_loss)
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'vocab_size': vocab_size,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'level': args.level,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"✓ Saved best model (loss: {train_loss:.4f})")
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {output_dir / 'best_model.pt'}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
