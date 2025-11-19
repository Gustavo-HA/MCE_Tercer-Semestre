import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Convolución Separable en Profundidad (Depthwise Separable Convolution)
    
    Componente clave de MobileNet que reduce el costo computacional.
    Consiste en dos pasos:
    1. Convolución depthwise: aplica un filtro por canal de entrada
    2. Convolución pointwise (1x1): combina las salidas de la convolución depthwise
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Inicializa el bloque de convolución separable en profundidad
        
        Args:
            in_channels (int): Número de canales de entrada
            out_channels (int): Número de canales de salida
            stride (int): Stride para la convolución depthwise (default: 1)
        """
        super(DepthwiseSeparableConv, self).__init__()
        
        # Convolución Depthwise (un filtro por canal de entrada)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Convolución Pointwise (1x1 para combinar canales)
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                     padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Propagación hacia adelante
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Tensor de salida
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNet(nn.Module):
    """
    Red Neuronal Convolucional MobileNet v1
    
    Arquitectura original de: Howard et al., 2017
    "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    
    Características principales:
    - Usa convoluciones separables en profundidad para reducir parámetros y cómputo
    - Factoriza convoluciones estándar en depthwise y pointwise
    - Reduce parámetros ~8-9x comparado con redes convencionales
    - Adaptada para imágenes de 32x32 (CIFAR-10)
    
    Arquitectura (adaptada para 32x32):
    - Conv estándar inicial: 3x3, 32 filtros
    - 13 bloques de convolución separable en profundidad
    - Average pooling global
    - Fully connected final
    """
    
    def __init__(self, num_classes=10, width_mult=1.0):
        """
        Inicializa el modelo MobileNet
        
        Args:
            num_classes (int): Número de clases de salida (default: 10 para CIFAR-10)
            width_mult (float): Multiplicador de ancho para hacer el modelo más pequeño/grande
                               (default: 1.0, valores típicos: 0.25, 0.5, 0.75, 1.0)
        """
        super(MobileNet, self).__init__()
        
        # Ajusta el número de canales según el multiplicador de ancho
        def conv_channels(channels):
            return int(channels * width_mult)
        
        # Primera capa: convolución estándar
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv_channels(32), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels(32)),
            nn.ReLU(inplace=True)
        )
        
        # Bloques de convolución separable en profundidad
        # Formato: [canales_salida, stride]
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(conv_channels(32), conv_channels(64), stride=1),   # 32x32
            DepthwiseSeparableConv(conv_channels(64), conv_channels(128), stride=2),  # 16x16
            DepthwiseSeparableConv(conv_channels(128), conv_channels(128), stride=1), # 16x16
            DepthwiseSeparableConv(conv_channels(128), conv_channels(256), stride=2), # 8x8
            DepthwiseSeparableConv(conv_channels(256), conv_channels(256), stride=1), # 8x8
            DepthwiseSeparableConv(conv_channels(256), conv_channels(512), stride=2), # 4x4
            
            # 5 bloques con 512 filtros
            DepthwiseSeparableConv(conv_channels(512), conv_channels(512), stride=1), # 4x4
            DepthwiseSeparableConv(conv_channels(512), conv_channels(512), stride=1), # 4x4
            DepthwiseSeparableConv(conv_channels(512), conv_channels(512), stride=1), # 4x4
            DepthwiseSeparableConv(conv_channels(512), conv_channels(512), stride=1), # 4x4
            DepthwiseSeparableConv(conv_channels(512), conv_channels(512), stride=1), # 4x4
            
            DepthwiseSeparableConv(conv_channels(512), conv_channels(1024), stride=2), # 2x2
            DepthwiseSeparableConv(conv_channels(1024), conv_channels(1024), stride=1), # 2x2
        )
        
        # Pooling global y clasificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(conv_channels(1024), num_classes)
        
    def forward(self, x):
        """
        Propagación hacia adelante a través de la red
        
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Logits de salida de forma (batch_size, num_classes)
        """
        x = self.conv1(x)         # (batch, 3, 32, 32) -> (batch, 32, 32, 32)
        x = self.layers(x)        # (batch, 32, 32, 32) -> (batch, 1024, 2, 2)
        x = self.avgpool(x)       # (batch, 1024, 2, 2) -> (batch, 1024, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 1024, 1, 1) -> (batch, 1024)
        x = self.fc(x)            # (batch, 1024) -> (batch, num_classes)
        return x


# Funciones de conveniencia para crear variantes con diferentes anchos
def MobileNet_1_0(num_classes=10):
    """Crea MobileNet con multiplicador de ancho 1.0 (modelo completo)"""
    return MobileNet(num_classes=num_classes, width_mult=1.0)


def MobileNet_0_75(num_classes=10):
    """Crea MobileNet con multiplicador de ancho 0.75"""
    return MobileNet(num_classes=num_classes, width_mult=0.75)


def MobileNet_0_5(num_classes=10):
    """Crea MobileNet con multiplicador de ancho 0.5"""
    return MobileNet(num_classes=num_classes, width_mult=0.5)


def MobileNet_0_25(num_classes=10):
    """Crea MobileNet con multiplicador de ancho 0.25 (modelo más pequeño)"""
    return MobileNet(num_classes=num_classes, width_mult=0.25)
