import torch
import torch.nn as nn
from rnntype import RNNType
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import argparse
import logging
import numpy as np
from tqdm import tqdm

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
        # Input: seq_length tokens
        # Target: next token after the sequence
        input_seq = self.encoded[idx:idx + self.seq_length]
        target = self.encoded[idx + self.seq_length]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def calculate_perplexity(model, dataloader, device):
    """
    Calculate perplexity of the model on a dataset.
    
    Args:
        model: Trained RNN model
        dataloader: DataLoader for test dataset
        device: Device to run on
    
    Returns:
        perplexity: Perplexity value
        avg_loss: Average cross-entropy loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    total_loss = 0.0
    total_tokens = 0
    
    logger.info("Calculating perplexity...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Processing batches"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            batch_size = inputs.size(0)
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)
            
            # Forward pass
            outputs, hidden = model(inputs, hidden)
            
            # Only use the last output for prediction
            outputs = outputs[:, -1, :]  # (batch_size, vocab_size)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            total_tokens += targets.size(0)
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def calculate_perplexity_sliding_window(model, text, token_to_idx, device, seq_length=100, stride=50, level='char'):
    """
    Calculate perplexity using sliding window approach on raw text.
    This method processes the entire text sequence by sequence.
    
    Args:
        model: Trained RNN model
        text: Test text string
        token_to_idx: Token to index mapping
        device: Device to run on
        seq_length: Length of input sequences
        stride: Stride for sliding window
        level: 'char' or 'word' tokenization level
    
    Returns:
        perplexity: Perplexity value
        avg_loss: Average cross-entropy loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Tokenize text
    if level == 'char':
        tokens = list(text)
    else:  # word level
        tokens = text.split()
    
    # Encode text
    encoded = [token_to_idx.get(token, 0) for token in tokens]
    
    if len(encoded) <= seq_length:
        logger.warning(f"Text too short ({len(encoded)} tokens). Need at least {seq_length + 1} tokens.")
        return float('inf'), float('inf')
    
    total_loss = 0.0
    num_predictions = 0
    
    logger.info(f"Calculating perplexity with sliding window (seq_length={seq_length}, stride={stride})...")
    
    with torch.no_grad():
        # Sliding window over the text
        for start_idx in tqdm(range(0, len(encoded) - seq_length, stride), desc="Processing windows"):
            end_idx = start_idx + seq_length
            
            if end_idx >= len(encoded):
                break
            
            # Get input sequence and target
            input_seq = encoded[start_idx:end_idx]
            target = encoded[end_idx]
            
            # Convert to tensors
            x = torch.tensor([input_seq], dtype=torch.long).to(device)
            y = torch.tensor([target], dtype=torch.long).to(device)
            
            # Initialize hidden state
            hidden = model.init_hidden(1, device)
            
            # Forward pass
            output, hidden = model(x, hidden)
            output = output[:, -1, :]  # Get last time step
            
            # Compute loss
            loss = criterion(output, y)
            
            total_loss += loss.item()
            num_predictions += 1
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / num_predictions
    perplexity = np.exp(avg_loss)
    
    return perplexity, avg_loss


def main():
    parser = argparse.ArgumentParser(description='Calculate perplexity of RNN model on test set')
    
    # Model path
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--level', type=str, required=True, choices=['char', 'word'],
                        help='Tokenization level: char or word')
    
    # Test data
    parser.add_argument('--test_file', type=str, default=None,
                        help='Path to test file (default: auto-detect)')
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='Path to vocabulary file (default: auto-detect)')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Sequence length (default: 100)')
    parser.add_argument('--method', type=str, default='dataloader', 
                        choices=['dataloader', 'sliding_window'],
                        help='Perplexity calculation method (default: dataloader)')
    parser.add_argument('--stride', type=int, default=50,
                        help='Stride for sliding window method (default: 50)')
    
    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save results (default: save to model directory)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    logger.info("="*80)
    logger.info(f"RNN PERPLEXITY CALCULATION - {args.level.upper()} LEVEL")
    logger.info("="*80)
    
    # Determine paths
    base_dir = Path(__file__).parent.parent.parent.parent
    
    if args.vocab_path is None:
        vocab_path = base_dir / "data" / "text_gen" / f"vocab_{args.level}.json"
    else:
        vocab_path = Path(args.vocab_path)
    
    if args.test_file is None:
        test_file = base_dir / "data" / "text_gen" / "test_lyrics.txt"
    else:
        test_file = Path(args.test_file)
    
    # Load vocabulary
    logger.info(f"\nLoading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    if args.level == 'char':
        token_to_idx = vocab_data['char_to_idx']
    else:  # word level
        token_to_idx = vocab_data['word_to_idx']
    
    vocab_size = vocab_data['vocab_size']
    logger.info(f"Vocabulary size: {vocab_size:,}")
    
    # Load model
    logger.info(f"\nLoading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = RNNType(
        rnn_type= checkpoint['rnn_type'],
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    logger.info(f"  - Trained for {checkpoint['epoch']} epochs")
    logger.info(f"  - Training loss: {checkpoint['loss']:.4f}")
    logger.info(f"  - Embedding dim: {checkpoint['embedding_dim']}")
    logger.info(f"  - Hidden dim: {checkpoint['hidden_dim']}")
    logger.info(f"  - Num layers: {checkpoint['num_layers']}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  - Model parameters: {num_params:,}")
    
    # Load test data
    logger.info(f"\nLoading test data from: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    if args.level == 'char':
        logger.info(f"Test text length: {len(test_text):,} characters")
    else:
        logger.info(f"Test text length: {len(test_text.split()):,} words")
    
    # Calculate perplexity
    logger.info("\n" + "="*80)
    logger.info(f"CALCULATING PERPLEXITY (Method: {args.method})")
    logger.info("="*80)
    
    if args.method == 'dataloader':
        # Create test dataset
        test_dataset = LyricsDataset(
            test_text, 
            token_to_idx, 
            seq_length=args.seq_length, 
            level=args.level
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Number of test sequences: {len(test_dataset):,}")
        logger.info(f"Number of batches: {len(test_loader):,}")
        
        perplexity, avg_loss = calculate_perplexity(model, test_loader, device)
    
    else:  # sliding_window
        perplexity, avg_loss = calculate_perplexity_sliding_window(
            model, 
            test_text, 
            token_to_idx, 
            device, 
            seq_length=args.seq_length,
            stride=args.stride,
            level=args.level
        )
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Average Loss (Cross-Entropy): {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.4f}")
    logger.info("="*80)
    
    # Save results
    if args.output_file is None:
        # Save to model directory
        model_dir = Path(args.model_path).parent
        output_file = model_dir / f"perplexity_results_{args.level}.txt"
    else:
        output_file = Path(args.output_file)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"RNN PERPLEXITY RESULTS - {args.level.upper()} LEVEL\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test file: {test_file}\n")
        f.write(f"Tokenization level: {args.level}\n")
        f.write(f"Vocabulary size: {vocab_size:,}\n")
        f.write(f"Sequence length: {args.seq_length}\n")
        f.write(f"Method: {args.method}\n")
        
        if args.method == 'dataloader':
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Number of test sequences: {len(test_dataset):,}\n")
        else:
            f.write(f"Stride: {args.stride}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Embedding dimension: {checkpoint['embedding_dim']}\n")
        f.write(f"Hidden dimension: {checkpoint['hidden_dim']}\n")
        f.write(f"Number of layers: {checkpoint['num_layers']}\n")
        f.write(f"Dropout: {checkpoint['dropout']}\n")
        f.write(f"Model parameters: {num_params:,}\n")
        f.write(f"Training epochs: {checkpoint['epoch']}\n")
        f.write(f"Training loss: {checkpoint['loss']:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("PERPLEXITY RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Loss (Cross-Entropy): {avg_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.4f}\n")
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"\n✓ Results saved to: {output_file}")
    
    # Interpretation
    logger.info("\n" + "="*80)
    logger.info("INTERPRETATION")
    logger.info("="*80)
    logger.info("Lower perplexity indicates better model performance.")
    logger.info("Perplexity represents the average number of choices the model")
    logger.info("has when predicting the next token (lower is better).")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
