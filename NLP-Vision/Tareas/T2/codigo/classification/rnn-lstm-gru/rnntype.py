from torch import nn

class RNNType(nn.Module):    
    def __init__(self, rnn_type, vocab_size, num_classes, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        RNN-based classifier for sequence classification.
        
        Args:
            rnn_type: Type of RNN ('RNN', 'LSTM', 'GRU')
            vocab_size: Size of vocabulary
            num_classes: Number of output classes
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
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = self.rnn_type(
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        rnn_out, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_length, hidden_dim)
        
        # Use the last hidden state for classification
        last_hidden = rnn_out[:, -1, :]  # (batch_size, hidden_dim)
        
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # (batch_size, num_classes)
        
        return logits