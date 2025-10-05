"""
Prepare data for text generation models (RNN/LSTM/Transformer) from Alpaca format.
Extracts lyrics text and creates processed datasets for both character-level and word-level text generation.
"""

import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_alpaca_data(filepath):
    """Load data from Alpaca format JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} samples from {filepath}")
    return data


def extract_lyrics(alpaca_data):
    """Extract lyrics text from Alpaca format."""
    lyrics_list = []
    for item in alpaca_data:
        # Extract the output field which contains the lyrics
        lyrics = item.get('output', '')
        if lyrics:
            lyrics_list.append(lyrics)
    return lyrics_list


def create_char_dataset(lyrics_list, output_file):
    """
    Create character-level dataset.
    Wraps each song with start and end tags.
    """
    # Wrap each song with start and end tags
    tagged_songs = [f"<|song_start|>\n{lyrics}\n<|song_end|>" for lyrics in lyrics_list]
    
    # Join all songs with newlines
    full_text = "\n".join(tagged_songs)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    # Get statistics
    vocab = sorted(set(full_text))
    
    logger.info("Character-level dataset statistics:")
    logger.info(f"  Total characters: {len(full_text):,}")
    logger.info(f"  Vocabulary size: {len(vocab)}")
    logger.info(f"  Number of songs: {len(lyrics_list)}")
    logger.info(f"  Saved to: {output_file}")
    
    return full_text, vocab


def extract_word_vocab(text):
    """
    Extract word-level vocabulary from text.
    Tokenizes by whitespace and keeps special tags intact.
    """
    # Simple whitespace tokenization
    words = text.split()
    vocab = sorted(set(words))
    return vocab, words


def create_vocab_mapping(vocab, output_dir, level='char'):
    """Create and save character-to-index or word-to-index mappings."""
    if level == 'char':
        token_to_idx = {ch: i for i, ch in enumerate(vocab)}
        idx_to_token = {i: ch for i, ch in enumerate(vocab)}
        vocab_file = os.path.join(output_dir, 'vocab_char.json')
    else:  # word level
        token_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_token = {i: word for i, word in enumerate(vocab)}
        vocab_file = os.path.join(output_dir, 'vocab_word.json')
    
    # Save mappings
    with open(vocab_file, 'w', encoding='utf-8') as f:
        if level == 'char':
            json.dump({
                'char_to_idx': token_to_idx,
                'idx_to_char': {str(k): v for k, v in idx_to_token.items()},  # JSON requires string keys
                'vocab_size': len(vocab),
                'level': 'character'
            }, f, indent=2, ensure_ascii=False)
        else:
            json.dump({
                'word_to_idx': token_to_idx,
                'idx_to_word': {str(k): v for k, v in idx_to_token.items()},
                'vocab_size': len(vocab),
                'level': 'word'
            }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"{level.capitalize()}-level vocabulary mappings saved to: {vocab_file}")
    return token_to_idx, idx_to_token


def main():
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "text_gen"
    output_dir = data_dir
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("PREPARING TEXT GENERATION DATA FROM ALPACA FORMAT")
    logger.info("="*80)
    
    # Process training data
    logger.info("\nProcessing training data...")
    train_alpaca = load_alpaca_data(data_dir / "train_lyrics_alpaca.json")
    train_lyrics = extract_lyrics(train_alpaca)
    train_text, train_vocab = create_char_dataset(
        train_lyrics, 
        output_dir / "train_lyrics.txt"
    )
    
    # Process test data
    logger.info("\nProcessing test data...")
    test_alpaca = load_alpaca_data(data_dir / "test_lyrics_alpaca.json")
    test_lyrics = extract_lyrics(test_alpaca)
    test_text, test_vocab = create_char_dataset(
        test_lyrics, 
        output_dir / "test_lyrics.txt"
    )
    
    # Create character-level vocabulary from training data
    logger.info("\nCreating character-level vocabulary mappings...")
    char_to_idx, idx_to_char = create_vocab_mapping(train_vocab, output_dir, level='char')
    
    # Extract and create word-level vocabulary
    logger.info("\nCreating word-level vocabulary mappings...")
    train_word_vocab, train_words = extract_word_vocab(train_text)
    test_word_vocab, test_words = extract_word_vocab(test_text)
    word_to_idx, idx_to_word = create_vocab_mapping(train_word_vocab, output_dir, level='word')
    
    logger.info("\nWord-level dataset statistics:")
    logger.info(f"  Training - Total words: {len(train_words):,}")
    logger.info(f"  Training - Vocabulary size: {len(train_word_vocab)}")
    logger.info(f"  Test - Total words: {len(test_words):,}")
    logger.info(f"  Test - Vocabulary size: {len(test_word_vocab)}")
    
    # Save dataset info
    info = {
        "char_level": {
            "train": {
                "num_songs": len(train_lyrics),
                "total_chars": len(train_text),
                "vocab_size": len(train_vocab)
            },
            "test": {
                "num_songs": len(test_lyrics),
                "total_chars": len(test_text),
                "vocab_size": len(test_vocab)
            }
        },
        "word_level": {
            "train": {
                "num_songs": len(train_lyrics),
                "total_words": len(train_words),
                "vocab_size": len(train_word_vocab)
            },
            "test": {
                "num_songs": len(test_lyrics),
                "total_words": len(test_words),
                "vocab_size": len(test_word_vocab)
            }
        },
        "start_tag": "<|song_start|>",
        "end_tag": "<|song_end|>"
    }
    
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"\nDataset info saved to: {info_file}")
    logger.info("\n" + "="*80)
    logger.info("✓ DATA PREPARATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Processed data saved in: {output_dir}")
    logger.info("\nFiles created:")
    logger.info("  - train_lyrics.txt, test_lyrics.txt (text files)")
    logger.info("  - vocab_char.json (character-level vocabulary)")
    logger.info("  - vocab_word.json (word-level vocabulary)")
    logger.info("  - dataset_info.json (statistics)")
    logger.info("\nYou can now run text generation training scripts (RNN/LSTM/etc).")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
