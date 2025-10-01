"""
Character-level RNN for lyrics text generation.
Trains an LSTM model to generate lyrics character by character.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class LyricsDataset(Dataset):
    """Character-level dataset for text generation."""
    
    def __init__(self, text, char_to_idx, seq_length=100):
        """
        Args:
            text: Full text string
            char_to_idx: Dictionary mapping characters to indices
            seq_length: Length of input sequences
        """
        self.text = text
        self.char_to_idx = char_to_idx
        self.seq_length = seq_length
        
        # Encode text
        self.encoded = [char_to_idx.get(ch, 0) for ch in text]
        
    def __len__(self):
        return len(self.encoded) - self.seq_length
    
    def __getitem__(self, idx):
        # Input: seq_length characters
        # Target: next character after the sequence
        input_seq = self.encoded[idx:idx + self.seq_length]
        target = self.encoded[idx + self.seq_length]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class CharRNN(nn.Module):
    """Character-level RNN model using LSTM."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size: Size of character vocabulary
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(CharRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Tuple of (h_0, c_0) hidden states
        
        Returns:
            output: Logits of shape (batch_size, seq_length, vocab_size)
            hidden: Updated hidden states
        """
        # Embed input
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Pass through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)  # (batch_size, seq_length, hidden_dim)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to vocabulary
        output = self.fc(lstm_out)  # (batch_size, seq_length, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden states."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h_0, c_0)


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


def generate_text(model, start_text, char_to_idx, idx_to_char, device, length=500, temperature=1.0):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained RNN model
        start_text: Starting prompt text
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        device: Device to run on
        length: Number of characters to generate
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode start text
    current_seq = [char_to_idx.get(ch, 0) for ch in start_text]
    generated = start_text
    
    # Initialize hidden state
    hidden = model.init_hidden(1, device)
    
    with torch.no_grad():
        for _ in range(length):
            # Prepare input
            x = torch.tensor([current_seq], dtype=torch.long).to(device)
            
            # Forward pass
            output, hidden = model(x, hidden)
            output = output[:, -1, :].squeeze()  # Get last time step
            
            # Apply temperature
            output = output / temperature
            
            # Sample from distribution
            probs = torch.softmax(output, dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Add to generated text
            next_char = idx_to_char[next_char_idx]
            generated += next_char
            
            # Update sequence
            current_seq.append(next_char_idx)
            current_seq = current_seq[-100:]  # Keep last 100 chars
    
    return generated


def main():
    # Hyperparameters
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 64
    SEQ_LENGTH = 100
    LEARNING_RATE = 0.002
    NUM_EPOCHS = 20
    CLIP_GRAD = 5.0
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "text_gen"
    output_dir = base_dir / "models" / "rnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    logger.info("="*80)
    logger.info("TRAINING CHARACTER-LEVEL RNN FOR LYRICS GENERATION")
    logger.info("="*80)
    
    # Load vocabulary
    logger.info("\nLoading vocabulary...")
    with open(data_dir / "vocab.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    char_to_idx = vocab_data['char_to_idx']
    idx_to_char = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
    vocab_size = vocab_data['vocab_size']
    
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Load text data
    logger.info("\nLoading training data...")
    with open(data_dir / "train_lyrics.txt", 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    logger.info(f"Training text length: {len(train_text):,} characters")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = LyricsDataset(train_text, char_to_idx, seq_length=SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    logger.info(f"Number of training sequences: {len(train_dataset):,}")
    logger.info(f"Number of batches per epoch: {len(train_loader):,}")
    
    # Create model
    logger.info("\nInitializing model...")
    model = CharRNN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        logger.info("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, CLIP_GRAD)
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
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
            }
            torch.save(checkpoint, output_dir / 'best_model.pt')
            logger.info(f"✓ Saved best model (loss: {train_loss:.4f})")
        
        # Generate sample text every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info("\nGenerating sample text...")
            sample_text = generate_text(
                model, 
                start_text="[Verse]\n", 
                char_to_idx=char_to_idx,
                idx_to_char=idx_to_char,
                device=device,
                length=300,
                temperature=0.8
            )
            logger.info(f"\nSample generation:\n{'-'*40}\n{sample_text}\n{'-'*40}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {output_dir / 'best_model.pt'}")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
