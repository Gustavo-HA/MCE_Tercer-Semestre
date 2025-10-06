import torch
from torch import nn

class RNNType(nn.Module):    
    def __init__(self, rnn_type, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        Args:
            rnn_type: Type of RNN ('RNN', 'LSTM', 'GRU')
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of RNN hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(RNNType, self).__init__()
        type_dict = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }
        
        self.rnn_type = type_dict.get(rnn_type.upper())
        if self.rnn_type is None:
            raise ValueError(f"Invalid rnn_type: {rnn_type}. Choose from 'RNN', 'LSTM', 'GRU'.")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = self.rnn_type(
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
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        if self.rnn_type == nn.LSTM:
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
            return (h_0, c_0)
        return h_0