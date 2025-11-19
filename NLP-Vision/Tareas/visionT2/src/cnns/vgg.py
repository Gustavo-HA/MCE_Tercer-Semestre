import torch.nn as nn


# VGG configurations for different variants
vgg_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """
    Red Neuronal Convolucional VGG
    
    Arquitectura original de: Simonyan & Zisserman, 2014
    "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    
    Características:
    - Usa únicamente filtros de convolución 3x3
    - Max pooling de 2x2 con stride 2
    - Aumenta el número de filtros progresivamente: 64 -> 128 -> 256 -> 512
    - Capas completamente conectadas con Dropout
    
    Variantes disponibles: VGG11, VGG13, VGG16, VGG19
    """
    
    def __init__(self, vgg_name='VGG16', num_classes=10, dropout=0.5, input_channels=3):
        """
        Inicializa el modelo VGG
        
        Args:
            vgg_name (str): Variante de VGG a usar ('VGG11', 'VGG13', 'VGG16', 'VGG19')
            num_classes (int): Número de clases de salida (default: 10 para CIFAR-10)
            dropout (float): Probabilidad de dropout (default: 0.5)
            input_channels (int): Número de canales de entrada (default: 3 para RGB)
        """
        super(VGG, self).__init__()
        self.features = self._make_layers(vgg_cfg[vgg_name], input_channels)
        
        # Capas de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        """
        Propagación hacia adelante a través de la red
        
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Logits de salida de forma (batch_size, num_classes)
        """
        x = self.features(x)      # (batch, 3, 32, 32) -> (batch, 512, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 512, 1, 1) -> (batch, 512)
        x = self.classifier(x)    # (batch, 512) -> (batch, num_classes)
        return x
    
    def _make_layers(self, cfg, input_channels):
        """
        Construye las capas de extracción de características según la configuración
        
        Args:
            cfg (list): Lista de configuración de capas (números para conv, 'M' para maxpool)
            input_channels (int): Número de canales de entrada
            
        Returns:
            nn.Sequential: Módulo secuencial con las capas de características
        """
        layers = []
        in_channels = input_channels
        
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        
        return nn.Sequential(*layers)


# Funciones de conveniencia para crear variantes específicas
def VGG11(num_classes=10, dropout=0.5, input_channels=3):
    """Crea un modelo VGG11"""
    return VGG('VGG11', num_classes, dropout, input_channels)


def VGG13(num_classes=10, dropout=0.5, input_channels=3):
    """Crea un modelo VGG13"""
    return VGG('VGG13', num_classes, dropout, input_channels)


def VGG16(num_classes=10, dropout=0.5, input_channels=3):
    """Crea un modelo VGG16"""
    return VGG('VGG16', num_classes, dropout, input_channels)


def VGG19(num_classes=10, dropout=0.5, input_channels=3):
    """Crea un modelo VGG19"""
    return VGG('VGG19', num_classes, dropout, input_channels)
