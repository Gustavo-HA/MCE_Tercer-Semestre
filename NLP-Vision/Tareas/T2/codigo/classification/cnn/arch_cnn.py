import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    CNN-based classifier for text classification.
    Uses multiple kernel sizes to capture different n-gram features.
    
    Args:
        vocab_size: Size of vocabulary
        num_classes: Number of output classes
        embedding_dim: Dimension of token embeddings
        num_filters: Number of filters per kernel size
        kernel_sizes: List of kernel sizes (e.g., [3, 4, 5] for trigrams, 4-grams, 5-grams)
        dropout: Dropout probability
    """
    
    def __init__(self, vocab_size, num_classes, embedding_dim=128, num_filters=100, 
                 kernel_sizes=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                     out_channels=num_filters, 
                     kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        # Total features = num_filters * len(kernel_sizes)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)
        
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        
    def forward(self, x):
        """
        Forward pass for classification.
        
        Args:
            x: Input tensor of token indices (batch_size, seq_length)
            
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Transpose for Conv1d: (batch_size, embedding_dim, seq_length)
        embedded = embedded.transpose(1, 2)
        
        # Apply convolution + ReLU + max pooling for each kernel size
        conv_outputs = []
        for conv in self.convs:
            # Convolution: (batch_size, num_filters, seq_length - kernel_size + 1)
            conv_out = F.relu(conv(embedded))
            
            # Max pooling over time: (batch_size, num_filters)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        # Concatenate all pooled features: (batch_size, num_filters * len(kernel_sizes))
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Apply dropout
        concatenated = self.dropout(concatenated)
        
        # Fully connected layer: (batch_size, num_classes)
        logits = self.fc(concatenated)
        
        return logits


if __name__ == "__main__":
    # Test the models
    vocab_size = 10000
    num_classes = 5
    batch_size = 16
    seq_length = 100
    
    # Test TextCNN
    print("Testing TextCNN...")
    model_cnn = TextCNN(vocab_size, num_classes)
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    output = model_cnn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model_cnn.parameters()):,}")
