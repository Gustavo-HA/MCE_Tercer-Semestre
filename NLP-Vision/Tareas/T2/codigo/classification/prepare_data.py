import pandas as pd
import argparse
import logging
import json
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    return data

def fix_mojibake(text: str) -> str:
    if not isinstance(text, str):
        return text
    s = text.encode('latin-1').decode('utf-8', errors='ignore')
    return s

def preprocess_text(data: pd.DataFrame) -> pd.DataFrame:
    """ No utilizaremos las columnas 'Town', 'Region' y 'Type' """
    data = data.drop(columns=['Town', 'Region', 'Type'], errors='ignore')
    data['Review'] = data['Review'].astype(str).apply(fix_mojibake)
    data['Polarity'] = data['Polarity'].astype(int) - 1
    data = data.rename(columns={'Review': 'text', 'Polarity': 'label'})
    data['uuid'] = data.index
    return data


def simple_tokenize(text: str) -> list:
    """Simple tokenization: lowercase and split by whitespace/punctuation"""
    import re
    text = text.lower()
    # Split on whitespace and punctuation, keeping only alphanumeric and basic chars
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_vocab(texts: pd.Series, min_freq: int = 2, max_vocab_size: int = None) -> dict:
    """
    Build vocabulary from texts
    
    Args:
        texts: Series of text strings
        min_freq: Minimum frequency for a word to be included
        max_vocab_size: Maximum vocabulary size (most frequent words)
    
    Returns:
        Dictionary mapping words to indices
    """
    from collections import Counter
    
    logger.info("Building vocabulary from training data...")
    
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        tokens = simple_tokenize(text)
        word_counts.update(tokens)
    
    logger.info(f"Total unique words before filtering: {len(word_counts)}")
    
    # Filter by minimum frequency
    word_counts = {word: count for word, count in word_counts.items() if count >= min_freq}
    
    # Sort by frequency and limit vocabulary size
    if max_vocab_size:
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
        word_counts = dict(sorted_words)
    
    logger.info(f"Vocabulary size after filtering (min_freq={min_freq}): {len(word_counts)}")
    
    # Build vocab: word -> index
    # Reserve special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1
    }
    
    for word in sorted(word_counts.keys()):
        vocab[word] = len(vocab)
    
    logger.info(f"Final vocabulary size (with special tokens): {len(vocab)}")
    return vocab


def text_to_indices(text: str, vocab: dict) -> list:
    """Convert text to list of indices using vocabulary"""
    tokens = simple_tokenize(text)
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]


def get_sequence_stats(texts: pd.Series) -> dict:
    """Get statistics about sequence lengths"""
    lengths = [len(simple_tokenize(text)) for text in texts]
    return {
        'min': min(lengths),
        'max': max(lengths),
        'mean': sum(lengths) / len(lengths),
        'median': sorted(lengths)[len(lengths) // 2],
        'percentile_95': sorted(lengths)[int(len(lengths) * 0.95)],
        'percentile_99': sorted(lengths)[int(len(lengths) * 0.99)],
    }



def main():
    parser = argparse.ArgumentParser(description='Prepare data for classification models')
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to the input CSV file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the processed JSON file')
    parser.add_argument('--min_freq', type=int, default=2,
                        help='Minimum word frequency for vocabulary (default: 2)')
    parser.add_argument('--max_vocab_size', type=int, default=None,
                        help='Maximum vocabulary size (default: unlimited)')
    args = parser.parse_args()

    logger.info(f"Loading data from {args.file_path}")
    data = load_data(args.file_path)
    
    logger.info("Preprocessing text data")
    data = preprocess_text(data)
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42, stratify=train_data['label'])

    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
    
    # Get sequence length statistics
    logger.info("Computing sequence length statistics...")
    seq_stats = get_sequence_stats(train_data['text'])
    logger.info(f"Sequence length stats: {seq_stats}")
    
    # Build vocabulary from training data only
    vocab = build_vocab(
        train_data['text'], 
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size
    )
    
    # Save vocabulary
    vocab_path = args.output_path.replace('.json', '_vocab.json')
    logger.info(f"Saving vocabulary to {vocab_path}")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Save sequence stats
    stats_path = args.output_path.replace('.json', '_stats.json')
    logger.info(f"Saving sequence statistics to {stats_path}")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(seq_stats, f, ensure_ascii=False, indent=2)

    logger.info(f"Saving processed data to {args.output_path}")

    # For transformer finetuning (JSON format)
    data_dict = {
        'train': train_data.to_dict(orient='records'),
        'validation': val_data.to_dict(orient='records'),
        'test': test_data.to_dict(orient='records')
    }
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4)

    # For RNN/LSTM/GRU/CNN training (CSV format)
    train_csv = args.output_path.replace('.json', '_train.csv')
    val_csv = args.output_path.replace('.json', '_val.csv')
    test_csv = args.output_path.replace('.json', '_test.csv')
    
    train_data.to_csv(train_csv, index=False, encoding='utf-8')
    val_data.to_csv(val_csv, index=False, encoding='utf-8')
    test_data.to_csv(test_csv, index=False, encoding='utf-8')

    logger.info("Data saved successfully!")
    logger.info(f"  - Transformer format: {args.output_path}")
    logger.info(f"  - CSV files: {train_csv}, {val_csv}, {test_csv}")
    logger.info(f"  - Vocabulary: {vocab_path}")
    logger.info(f"  - Sequence stats: {stats_path}")
    logger.info(f"  - Vocabulary size: {len(vocab)}")
    logger.info(f"  - Recommended max_seq_length: {int(seq_stats['percentile_95'])}-{int(seq_stats['percentile_99'])}")
    
    
if __name__ == "__main__":
    main()