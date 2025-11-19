import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 Convolutional Neural Network
    
    Original architecture from: LeCun et al., 1998
    "Gradient-Based Learning Applied to Document Recognition"
    
    Architecture:
    - Input: 32x32 grayscale image
    - Conv1: 6 filters, 5x5 kernel
    - Pool1: 2x2 average pooling
    - Conv2: 16 filters, 5x5 kernel
    - Pool2: 2x2 average pooling
    - FC1: 120 units
    - FC2: 84 units
    - Output: num_classes units
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize LeNet-5 model
        
        Args:
            num_classes (int): Number of output classes (default: 10)
        """
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 32, 32)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Conv1 + ReLU + Pool
        x = self.conv1(x)           # (batch, 1, 32, 32) -> (batch, 6, 28, 28)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)      # (batch, 6, 28, 28) -> (batch, 6, 14, 14)
        
        # Conv2 + ReLU + Pool
        x = self.conv2(x)           # (batch, 6, 14, 14) -> (batch, 16, 10, 10)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)      # (batch, 16, 10, 10) -> (batch, 16, 5, 5)
        
        # Flatten
        x = x.view(x.size(0), -1)   # (batch, 16, 5, 5) -> (batch, 400)
        
        # Fully connected layers
        x = self.fc1(x)             # (batch, 400) -> (batch, 120)
        x = F.relu(x)
        
        x = self.fc2(x)             # (batch, 120) -> (batch, 84)
        x = F.relu(x)
        
        x = self.fc3(x)             # (batch, 84) -> (batch, num_classes)
        
        return x
