import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet Convolutional Neural Network
    
    Original architecture from: Krizhevsky et al., 2012
    "ImageNet Classification with Deep Convolutional Neural Networks"
    
    Architecture (adapted for 32x32 images):
    - Input: 32x32 RGB image
    - Conv1: 64 filters, 3x3 kernel, padding 1
    - MaxPool1: 2x2, stride 2
    - Conv2: 192 filters, 3x3 kernel, padding 1
    - MaxPool2: 2x2, stride 2
    - Conv3: 384 filters, 3x3 kernel, padding 1
    - Conv4: 256 filters, 3x3 kernel, padding 1
    - Conv5: 256 filters, 3x3 kernel, padding 1
    - MaxPool3: 2x2, stride 2
    - FC1: 4096 units with Dropout
    - FC2: 4096 units with Dropout
    - FC3: num_classes units
    
    Note: Smaller kernels and strides adapted for smaller input size (32x32)
    """
    
    def __init__(self, num_classes=10, dropout=0.5, input_size=32):
        """
        Initialize AlexNet model (adapted for small images)
        
        Args:
            num_classes (int): Number of output classes (default: 10 for CIFAR-10)
            dropout (float): Dropout probability (default: 0.5)
            input_size (int): Input image size (default: 32 for CIFAR-10 and MNIST)
        """
        super(AlexNet, self).__init__()
        
        # Feature extraction layers (adapted for 32x32 images)
        self.features = nn.Sequential(
            # Conv1: 32x32x3 -> 32x32x64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32x64 -> 16x16x64
            
            # Conv2: 16x16x64 -> 16x16x192
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16x192 -> 8x8x192
            
            # Conv3: 8x8x192 -> 8x8x384
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 8x8x384 -> 8x8x256
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 8x8x256 -> 8x8x256
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8x256 -> 4x4x256
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)      # (batch, 3, 32, 32) -> (batch, 256, 4, 4)
        x = x.view(x.size(0), -1) # (batch, 256, 4, 4) -> (batch, 4096)
        x = self.classifier(x)    # (batch, 4096) -> (batch, num_classes)
        return x
