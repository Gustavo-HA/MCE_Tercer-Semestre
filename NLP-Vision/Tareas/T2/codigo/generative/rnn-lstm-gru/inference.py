import torch
from rnntype import RNNType
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

def generate_text(model, start_text, token_to_idx, idx_to_token, device, length=500, temperature=1.0, level='char'):
    """
    Generar texto a partir de un modelo entrenado.
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
            # Preparar input
            x = torch.tensor([current_seq], dtype=torch.long).to(device)
            
            # Forward pass
            output, hidden = model(x, hidden)
            output = output[:, -1, :].squeeze()  # Get last time step
            
            # Apply temperature
            output = output / temperature
            
            # Muestrear siguiente token
            probs = torch.softmax(output, dim=0)
            next_token_idx = torch.multinomial(probs, 1).item()
            
            # Agregar token generado
            next_token = idx_to_token[next_token_idx]
            generated_tokens.append(next_token)
            
            # Actualizar secuencia total
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
    parser.add_argument('--rnn_type', type=str, default='LSTM',
                        help='Type of RNN: LSTM or GRU (default: LSTM)')
    
    # Generation parameters
    parser.add_argument('--length', type=int, default=500,
                        help='Number of tokens to generate (default: 500)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (default: 0.8)')
    
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
    
    model = RNNType(
        rnn_type=checkpoint['rnn_type'],
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")
    logger.info(f"Best training loss: {checkpoint['train_loss']:.4f}")
    
    # Iniciamos con la etiqueta de inicio de canción
    start_prompt = "<|song_start|>"
    logger.info(f"\nStart prompt: {start_prompt}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Length: {args.length} tokens")
    logger.info("="*80)

    # Generar muestras
    generated_texts = []
    
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
    logger.info("CANCION GENERADA:")
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
