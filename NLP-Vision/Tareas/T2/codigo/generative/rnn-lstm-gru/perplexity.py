import torch
import torch.nn as nn
from rnntype import RNNType
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


def calculate_perplexity_streaming(model, encoded_tokens, device, block_size=100):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # Necesitamos al menos 2 tokens para producir una predicción
    if len(encoded_tokens) < 2:
        logger.warning("Test sequence too short to compute perplexity.")
        return float('inf'), float('inf')

    total_loss = 0.0
    total_predictions = 0  # Número de tokens realmente predichos

    # Inicializamos hidden una sola vez
    hidden = model.init_hidden(1, device)

    pos = 0
    n_tokens = len(encoded_tokens)

    with torch.no_grad():
        pbar = tqdm(total=n_tokens - 1, desc="Streaming tokens", unit="tok")
        while pos < n_tokens - 1:
            # Garantizar que haya al menos (block_size + 1) tokens disponibles;
            # si no, reducimos el tamaño del bloque para abarcar el final.
            end = min(pos + block_size + 1, n_tokens)
            block = encoded_tokens[pos:end]

            # Si el bloque tiene solo 1 token, ya no hay nada que predecir
            if len(block) < 2:
                break

            # Inputs: todos menos el último; Targets: todos menos el primero
            inputs_block = block[:-1]
            targets_block = block[1:]

            inputs_tensor = torch.tensor(inputs_block, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
            targets_tensor = torch.tensor(targets_block, dtype=torch.long, device=device)  # (L,)

            # Forward
            outputs, hidden = model(inputs_tensor, hidden)
            # IMPORTANTE: detach para evitar acumulación del grafo en RNNs (aunque no hay backward)
            if isinstance(hidden, tuple):  # LSTM: (h, c)
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:
                hidden = hidden.detach()

            logits = outputs.squeeze(0)  # (L, vocab_size)
            # Pérdida: cada posición i predice targets_block[i]
            loss = criterion(logits, targets_tensor)

            total_loss += loss.item()
            total_predictions += len(targets_block)
            pbar.update(len(targets_block))

            # Avanzamos pos en block_size (no block_size+1) para reutilizar el último token del bloque como primer input siguiente
            pos += block_size
        pbar.close()

    avg_loss = total_loss / total_predictions
    ppl = float(np.exp(avg_loss))
    return ppl, avg_loss


def main():
    parser = argparse.ArgumentParser(description='Streaming perplexity calculation for RNN models')
    
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
    
    # Streaming block size
    parser.add_argument('--block_size', type=int, default=100,
                        help='Block size (number of input tokens per forward pass, default: 100)')
    
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
    logger.info(f"RNN PERPLEXITY CALCULATION (STREAMING) - {args.level.upper()} LEVEL")
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
    
    # Recuperar métricas guardadas (compatibilidad con claves posibles)
    train_loss = checkpoint.get('train_loss')
    val_loss = checkpoint.get('val_loss')
    logger.info("✓ Model loaded successfully")
    logger.info(f"  - Trained for {checkpoint.get('epoch','?')} epochs")
    if train_loss is not None:
        logger.info(f"  - Train loss (last epoch): {train_loss:.4f}")
    if val_loss is not None:
        logger.info(f"  - Val   loss (best): {val_loss:.4f}")
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
    logger.info("CALCULATING STREAMING PERPLEXITY")
    logger.info("="*80)

    # Tokenize full test text
    if args.level == 'char':
        tokens = list(test_text)
    else:
        tokens = test_text.split()
    encoded = [token_to_idx.get(tok, 0) for tok in tokens]
    logger.info(f"Encoded test length: {len(encoded):,} tokens ({args.level}-level)")

    perplexity, avg_loss = calculate_perplexity_streaming(
        model,
        encoded,
        device,
        block_size=args.block_size,
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
        model_dir = Path(args.model_path).parent
        output_file = model_dir / f"perplexity_streaming_{args.level}.txt"
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
        f.write(f"Block size: {args.block_size}\n")
        f.write("Evaluation: streaming continuous (state carried across blocks)\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Embedding dimension: {checkpoint['embedding_dim']}\n")
        f.write(f"Hidden dimension: {checkpoint['hidden_dim']}\n")
        f.write(f"Number of layers: {checkpoint['num_layers']}\n")
        f.write(f"Dropout: {checkpoint['dropout']}\n")
        f.write(f"Model parameters: {num_params:,}\n")
        f.write(f"Training epochs: {checkpoint['epoch']}\n")
        if train_loss is not None:
            f.write(f"Training loss (last): {train_loss:.4f}\n")
        if val_loss is not None:
            f.write(f"Validation loss (best): {val_loss:.4f}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("PERPLEXITY RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Average Loss (Cross-Entropy): {avg_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.4f}\n")
        f.write("\n" + "="*80 + "\n")
    
    logger.info(f"\n Results saved to: {output_file}")

if __name__ == "__main__":
    main()
