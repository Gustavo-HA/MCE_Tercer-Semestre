import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    """
    Módulo Inception básico
    
    Aplica convoluciones de diferentes tamaños en paralelo y concatena los resultados.
    Usa convoluciones 1x1 para reducir dimensionalidad antes de las convoluciones más grandes.
    """
    
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        """
        Inicializa el módulo Inception
        
        Args:
            in_channels (int): Número de canales de entrada
            out_1x1 (int): Filtros para rama de convolución 1x1
            red_3x3 (int): Filtros para reducción 1x1 antes de 3x3
            out_3x3 (int): Filtros para convolución 3x3
            red_5x5 (int): Filtros para reducción 1x1 antes de 5x5
            out_5x5 (int): Filtros para convolución 5x5
            out_pool (int): Filtros para reducción 1x1 después de pooling
        """
        super(InceptionModule, self).__init__()
        
        # Rama 1: convolución 1x1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        # Rama 2: reducción 1x1 -> convolución 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.BatchNorm2d(red_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )
        
        # Rama 3: reducción 1x1 -> convolución 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.BatchNorm2d(red_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        # Rama 4: max pooling 3x3 -> reducción 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Propagación hacia adelante
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Concatenación de todas las ramas
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """
    Clasificador auxiliar para Inception
    
    Usado durante el entrenamiento para combatir el problema de gradientes que desaparecen
    """
    
    def __init__(self, in_channels, num_classes):
        """
        Inicializa el clasificador auxiliar
        
        Args:
            in_channels (int): Número de canales de entrada
            num_classes (int): Número de clases de salida
        """
        super(InceptionAux, self).__init__()
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)
        
    def forward(self, x):
        """
        Propagación hacia adelante
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Logits de clasificación
        """
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    """
    Red Neuronal Convolucional GoogLeNet (Inception v1)
    
    Arquitectura original de: Szegedy et al., 2014
    "Going Deeper with Convolutions"
    
    Características principales:
    - Usa módulos Inception con convoluciones paralelas de múltiples tamaños
    - Reducción de dimensionalidad con convoluciones 1x1
    - Clasificadores auxiliares para mejorar el gradiente
    - Adaptada para imágenes de 32x32 (CIFAR-10)
    
    Arquitectura (simplificada para 32x32):
    - Stem: Conv inicial + MaxPool
    - Inception 3a, 3b
    - MaxPool
    - Inception 4a, 4b, 4c, 4d, 4e (con clasificador auxiliar en 4a)
    - MaxPool
    - Inception 5a, 5b (con clasificador auxiliar en 5a)
    - AvgPool + Dropout + FC
    """
    
    def __init__(self, num_classes=10, aux_logits=True):
        """
        Inicializa el modelo GoogLeNet
        
        Args:
            num_classes (int): Número de clases de salida (default: 10 para CIFAR-10)
            aux_logits (bool): Si True, usa clasificadores auxiliares durante entrenamiento
        """
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        
        # Capas iniciales (stem)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloques Inception 3
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloques Inception 4
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Bloques Inception 5
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        # Clasificadores auxiliares
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(832, num_classes)
        
        # Clasificador principal
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        """
        Propagación hacia adelante a través de la red
        
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor o tuple: Si training y aux_logits, retorna (salida_principal, aux1, aux2)
                                  Si no, retorna solo salida_principal
        """
        # Stem
        x = self.conv1(x)      # 32x32x3 -> 32x32x64
        x = self.maxpool1(x)   # 32x32x64 -> 16x16x64
        x = self.conv2(x)      # 16x16x64 -> 16x16x192
        x = self.maxpool2(x)   # 16x16x192 -> 8x8x192
        
        # Inception 3
        x = self.inception3a(x)  # 8x8x192 -> 8x8x256
        x = self.inception3b(x)  # 8x8x256 -> 8x8x480
        x = self.maxpool3(x)     # 8x8x480 -> 4x4x480
        
        # Inception 4
        x = self.inception4a(x)  # 4x4x480 -> 4x4x512
        
        # Primer clasificador auxiliar
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)  # 4x4x512 -> 4x4x512
        x = self.inception4c(x)  # 4x4x512 -> 4x4x512
        x = self.inception4d(x)  # 4x4x512 -> 4x4x528
        x = self.inception4e(x)  # 4x4x528 -> 4x4x832
        x = self.maxpool4(x)     # 4x4x832 -> 2x2x832
        
        # Inception 5
        x = self.inception5a(x)  # 2x2x832 -> 2x2x832
        
        # Segundo clasificador auxiliar
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        
        x = self.inception5b(x)  # 2x2x832 -> 2x2x1024
        
        # Clasificador principal
        x = self.avgpool(x)      # 2x2x1024 -> 1x1x1024
        x = x.view(x.size(0), -1) # 1x1x1024 -> 1024
        x = self.dropout(x)
        x = self.fc(x)           # 1024 -> num_classes
        
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x
