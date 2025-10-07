from torch.utils.data import Dataset
import torch

class LyricsDataset(Dataset):
    def __init__(self, text, token_to_idx, seq_length=100, level='char'):
        self.text = text
        self.token_to_idx = token_to_idx
        self.seq_length = seq_length
        self.level = level
        
        if level == 'char':
            self.tokens = list(text)
        else:  # word level
            self.tokens = text.split()
        
        self.encoded = [token_to_idx.get(token, 0) for token in self.tokens]
        
    def __len__(self):
        return len(self.encoded) - self.seq_length
    
    def __getitem__(self, idx):
        input_seq = self.encoded[idx:idx + self.seq_length]
        target = self.encoded[idx + self.seq_length]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)
