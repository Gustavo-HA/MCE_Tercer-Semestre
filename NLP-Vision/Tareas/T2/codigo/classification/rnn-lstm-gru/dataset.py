import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import re


def simple_tokenize(text: str) -> list:
    """Simple tokenization: lowercase and split by whitespace/punctuation"""
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification using RNN/LSTM/GRU/CNN models.
    
    Args:
        csv_path: Path to CSV file with 'text' and 'label' columns
        vocab_path: Path to vocabulary JSON file
        max_length: Maximum sequence length (pad/truncate)
        pad_token: Padding token (default: '<PAD>')
    """
    
    def __init__(self, csv_path: str, vocab_path: str, max_length: int = 128, pad_token: str = '<PAD>'):
        self.max_length = max_length
        
        # Load data
        self.data = pd.read_csv(csv_path, encoding='utf-8')
        
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.pad_idx = self.vocab.get(pad_token, 0)
        self.unk_idx = self.vocab.get('<UNK>', 1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        label = row['label']
        
        # Tokenize and convert to indices
        tokens = simple_tokenize(text)
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        
        # Truncate or pad to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [self.pad_idx] * (self.max_length - len(indices))
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_num_classes(self):
        return len(self.data['label'].unique())


def create_dataloader(csv_path: str, vocab_path: str, batch_size: int = 32, 
                     max_length: int = 128, shuffle: bool = True, num_workers: int = 0):
    """
    Create a DataLoader for text classification.
    
    Args:
        csv_path: Path to CSV file
        vocab_path: Path to vocabulary JSON file
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader instance
    """
    dataset = TextClassificationDataset(csv_path, vocab_path, max_length)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == "__main__":
    # Example usage
    dataset = TextClassificationDataset(
        csv_path="./data/classification/meia_data_train.csv",
        vocab_path="./data/classification/meia_data_vocab.json",
        max_length=128
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary size: {dataset.get_vocab_size()}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    
    # Test first sample
    sample = dataset[0]
    print("\nFirst sample:")
    print(f"  Input shape: {sample['input_ids'].shape}")
    print(f"  Label: {sample['label']}")
