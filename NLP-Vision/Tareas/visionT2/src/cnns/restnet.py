import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Bloque Residual Básico para ResNet
    
    Usado en ResNet-18 y ResNet-34.
    Estructura:
    - Conv 3x3 -> BN -> ReLU
    - Conv 3x3 -> BN
    - Suma con entrada (shortcut)
    - ReLU
    
    El factor de expansión es 1 (sin cambio en número de canales dentro del bloque)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Inicializa el bloque residual básico
        
        Args:
            in_channels (int): Número de canales de entrada
            out_channels (int): Número de canales de salida
            stride (int): Stride para la primera convolución (default: 1)
            downsample (nn.Module): Capa para ajustar dimensiones del shortcut (default: None)
        """
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        """
        Propagación hacia adelante
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Tensor de salida
        """
        identity = x
        
        # Primera convolución
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Segunda convolución
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """
    Bloque Bottleneck para ResNet
    
    Usado en ResNet-50, ResNet-101 y ResNet-152.
    Estructura:
    - Conv 1x1 -> BN -> ReLU (reduce dimensiones)
    - Conv 3x3 -> BN -> ReLU
    - Conv 1x1 -> BN (expande dimensiones)
    - Suma con entrada (shortcut)
    - ReLU
    
    El factor de expansión es 4 (la última capa expande 4x los canales)
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Inicializa el bloque bottleneck
        
        Args:
            in_channels (int): Número de canales de entrada
            out_channels (int): Número de canales intermedios
            stride (int): Stride para la convolución 3x3 (default: 1)
            downsample (nn.Module): Capa para ajustar dimensiones del shortcut (default: None)
        """
        super(Bottleneck, self).__init__()
        
        # Reduce dimensiones (1x1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Convolución 3x3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Expande dimensiones (1x1)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        """
        Propagación hacia adelante
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Tensor de salida
        """
        identity = x
        
        # Reducción 1x1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Convolución 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Expansión 1x1
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Shortcut connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    Red Neuronal Convolucional ResNet (Residual Network)
    
    Arquitectura original de: He et al., 2015
    "Deep Residual Learning for Image Recognition"
    
    Características principales:
    - Usa conexiones residuales (skip connections) para entrenar redes muy profundas
    - Resuelve el problema de gradientes que desaparecen
    - Permite entrenar redes de 50, 101, 152+ capas
    - Adaptada para imágenes de 32x32 (CIFAR-10)
    
    Variantes disponibles:
    - ResNet-18: 18 capas con BasicBlock
    - ResNet-34: 34 capas con BasicBlock
    - ResNet-50: 50 capas con Bottleneck
    - ResNet-101: 101 capas con Bottleneck
    - ResNet-152: 152 capas con Bottleneck
    """
    
    def __init__(self, block, layers, num_classes=10):
        """
        Inicializa el modelo ResNet
        
        Args:
            block (nn.Module): Tipo de bloque (BasicBlock o Bottleneck)
            layers (list): Número de bloques en cada capa [layer1, layer2, layer3, layer4]
            num_classes (int): Número de clases de salida (default: 10 para CIFAR-10)
        """
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        
        # Capa inicial adaptada para 32x32 (sin pooling agresivo)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Capas residuales
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Pooling global y clasificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Inicialización de pesos
        self._initialize_weights()
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Construye una capa residual con múltiples bloques
        
        Args:
            block (nn.Module): Tipo de bloque (BasicBlock o Bottleneck)
            out_channels (int): Número de canales de salida
            num_blocks (int): Número de bloques en la capa
            stride (int): Stride para el primer bloque
            
        Returns:
            nn.Sequential: Capa residual completa
        """
        downsample = None
        
        # Si las dimensiones cambian, necesitamos ajustar el shortcut
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        # Primer bloque (puede cambiar dimensiones)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Bloques restantes (mantienen dimensiones)
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Propagación hacia adelante a través de la red
        
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Logits de salida de forma (batch_size, num_classes)
        """
        # Capa inicial
        x = self.conv1(x)         # (batch, 3, 32, 32) -> (batch, 64, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Capas residuales
        x = self.layer1(x)        # (batch, 64, 32, 32) -> (batch, 64/256, 32, 32)
        x = self.layer2(x)        # -> (batch, 128/512, 16, 16)
        x = self.layer3(x)        # -> (batch, 256/1024, 8, 8)
        x = self.layer4(x)        # -> (batch, 512/2048, 4, 4)
        
        # Clasificador
        x = self.avgpool(x)       # -> (batch, 512/2048, 1, 1)
        x = x.view(x.size(0), -1) # -> (batch, 512/2048)
        x = self.fc(x)            # -> (batch, num_classes)
        
        return x
    
    def _initialize_weights(self):
        """
        Inicializa los pesos del modelo
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# Funciones de conveniencia para crear variantes específicas
def ResNet18(num_classes=10):
    """Crea ResNet-18 (18 capas con BasicBlock)"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10):
    """Crea ResNet-34 (34 capas con BasicBlock)"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=10):
    """Crea ResNet-50 (50 capas con Bottleneck)"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=10):
    """Crea ResNet-101 (101 capas con Bottleneck)"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=10):
    """Crea ResNet-152 (152 capas con Bottleneck)"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)
