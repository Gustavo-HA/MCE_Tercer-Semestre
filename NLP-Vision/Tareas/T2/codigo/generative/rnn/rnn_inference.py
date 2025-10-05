"""
RNN inference script for lyrics text generation.
Generates lyrics using a trained RNN model.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


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


def generate_text(model, start_text, token_to_idx, idx_to_token, device, length=500, temperature=1.0, level='char'):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained RNN model
        start_text: Starting prompt text
        token_to_idx: Token to index mapping
        idx_to_token: Index to token mapping
        device: Device to run on
        length: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        level: 'char' or 'word' tokenization level
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Tokenize start text
    if level == 'char':
        tokens = list(start_text)
    else:  # word level
        tokens = start_text.split()
    
    # Encode start text
    current_seq = [token_to_idx.get(token, 0) for token in tokens]
    generated_tokens = tokens.copy()
    
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
            next_token_idx = torch.multinomial(probs, 1).item()
            
            # Add to generated text
            next_token = idx_to_token[next_token_idx]
            generated_tokens.append(next_token)
            
            # Update sequence
            current_seq.append(next_token_idx)
            current_seq = current_seq[-100:]  # Keep last 100 tokens
    
    # Join tokens based on level
    if level == 'char':
        return ''.join(generated_tokens)
    else:  # word level
        return ' '.join(generated_tokens)


def main():
    parser = argparse.ArgumentParser(description='Generate lyrics using trained RNN model')
    
    # Model path
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--level', type=str, required=True, choices=['char', 'word'],
                        help='Tokenization level: char or word')
    
    # Generation parameters
    parser.add_argument('--length', type=int, default=500,
                        help='Number of tokens to generate (default: 500)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate (default: 1)')
    
    # Vocabulary path
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='Path to vocabulary file (default: auto-detect)')
    
    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save generated text (default: print to console)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load vocabulary
    if args.vocab_path is None:
        base_dir = Path(__file__).parent.parent.parent.parent
        vocab_path = base_dir / "data" / "text_gen" / f"vocab_{args.level}.json"
    else:
        vocab_path = Path(args.vocab_path)
    
    logger.info(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    if args.level == 'char':
        token_to_idx = vocab_data['char_to_idx']
        idx_to_token = {int(k): v for k, v in vocab_data['idx_to_char'].items()}
    else:  # word level
        token_to_idx = vocab_data['word_to_idx']
        idx_to_token = {int(k): v for k, v in vocab_data['idx_to_word'].items()}
    
    vocab_size = vocab_data['vocab_size']
    logger.info(f"Vocabulary size: {vocab_size:,}")
    
    # Load model
    logger.info(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model = SimpleRNN(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    logger.info(f"Best training loss: {checkpoint['loss']:.4f}")
    
    # Start prompt
    start_prompt = "<|song_start|>"
    logger.info(f"\nStart prompt: {start_prompt}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Length: {args.length} tokens")
    logger.info("="*80)
    
    # Generate samples
    generated_texts = []
    
    for i in range(args.num_samples):
        logger.info(f"\nGenerating sample {i+1}/{args.num_samples}...")
        
        generated_text = generate_text(
            model=model,
            start_text=start_prompt,
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            device=device,
            length=args.length,
            temperature=args.temperature,
            level=args.level
        )
        
        generated_texts.append(generated_text)
        
        logger.info("\n" + "="*80)
        logger.info(f"GENERATED SAMPLE {i+1}")
        logger.info("="*80)
        print(generated_text)
        logger.info("="*80)
    
    # Save to file if specified
    if args.output_file is not None:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(generated_texts):
                f.write(f"{'='*80}\n")
                f.write(f"SAMPLE {i+1}\n")
                f.write(f"{'='*80}\n")
                f.write(text)
                f.write("\n\n")
        
        logger.info(f"\n✓ Generated text saved to: {output_path}")


if __name__ == "__main__":
    main()
