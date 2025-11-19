import torch.nn as nn


class InvertedResidual(nn.Module):
    """
    Bloque Residual Invertido (Inverted Residual Block)
    
    Componente clave de MobileNet v2. A diferencia de los bloques residuales tradicionales,
    este bloque:
    1. Expande los canales con una capa 1x1 (expansion)
    2. Aplica convolución depthwise 3x3
    3. Reduce los canales con una capa 1x1 (projection)
    4. Usa conexión residual si stride=1 y mismos canales de entrada/salida
    
    La estructura es "angosta -> ancha -> angosta" (inversa a ResNet)
    """
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        """
        Inicializa el bloque residual invertido
        
        Args:
            in_channels (int): Número de canales de entrada
            out_channels (int): Número de canales de salida
            stride (int): Stride para la convolución depthwise
            expand_ratio (int): Factor de expansión para la capa intermedia
        """
        super(InvertedResidual, self).__init__()
        self.stride = stride
        
        # Conexión residual solo si stride=1 y dimensiones coinciden
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        
        # Expansion: solo si expand_ratio != 1
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Projection: reduce dimensiones (linear bottleneck, sin ReLU)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Propagación hacia adelante
        
        Args:
            x (torch.Tensor): Tensor de entrada
            
        Returns:
            torch.Tensor: Tensor de salida
        """
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    Red Neuronal Convolucional MobileNet v2
    
    Arquitectura original de: Sandler et al., 2018
    "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
    
    Características principales:
    - Usa bloques residuales invertidos (narrow -> wide -> narrow)
    - Linear bottlenecks (sin ReLU después de la proyección)
    - ReLU6 para mejor cuantización en dispositivos móviles
    - Conexiones residuales cuando stride=1
    - Adaptada para imágenes de 32x32 (CIFAR-10)
    
    Mejoras sobre v1:
    - Conexiones residuales mejoran el flujo del gradiente
    - Linear bottlenecks preservan mejor la información
    - Más eficiente y preciso que v1
    """
    
    def __init__(self, num_classes=10, width_mult=1.0):
        """
        Inicializa el modelo MobileNet v2
        
        Args:
            num_classes (int): Número de clases de salida (default: 10 para CIFAR-10)
            width_mult (float): Multiplicador de ancho para ajustar el tamaño del modelo
                               (default: 1.0, valores típicos: 0.35, 0.5, 0.75, 1.0, 1.4)
        """
        super(MobileNetV2, self).__init__()
        
        # Configuración de los bloques: [expansion, out_channels, num_blocks, stride]
        # Adaptada para imágenes de 32x32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   # 32x32
            [6, 24, 2, 1],   # 32x32 (stride 1 para mantener resolución en CIFAR-10)
            [6, 32, 3, 2],   # 16x16
            [6, 64, 4, 2],   # 8x8
            [6, 96, 3, 1],   # 8x8
            [6, 160, 3, 2],  # 4x4
            [6, 320, 1, 1],  # 4x4
        ]
        
        # Ajusta canales según width_mult
        def make_divisible(v, divisor=8):
            """Asegura que todos los canales sean divisibles por 8"""
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        input_channel = make_divisible(32 * width_mult)
        
        # Primera capa: convolución estándar
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Bloques residuales invertidos
        layers = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        self.layers = nn.Sequential(*layers)
        
        # Última capa convolucional
        last_channel = make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Clasificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(last_channel, num_classes)
        
        # Inicialización de pesos
        self._initialize_weights()
        
    def forward(self, x):
        """
        Propagación hacia adelante a través de la red
        
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, 3, 32, 32)
            
        Returns:
            torch.Tensor: Logits de salida de forma (batch_size, num_classes)
        """
        x = self.conv1(x)         # (batch, 3, 32, 32) -> (batch, 32, 32, 32)
        x = self.layers(x)        # (batch, 32, 32, 32) -> (batch, 320, 4, 4)
        x = self.conv2(x)         # (batch, 320, 4, 4) -> (batch, 1280, 4, 4)
        x = self.avgpool(x)       # (batch, 1280, 4, 4) -> (batch, 1280, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 1280, 1, 1) -> (batch, 1280)
        x = self.dropout(x)
        x = self.fc(x)            # (batch, 1280) -> (batch, num_classes)
        return x
    
    def _initialize_weights(self):
        """
        Inicializa los pesos del modelo
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


# Funciones de conveniencia para crear variantes con diferentes anchos
def MobileNetV2_1_4(num_classes=10):
    """Crea MobileNet v2 con multiplicador de ancho 1.4"""
    return MobileNetV2(num_classes=num_classes, width_mult=1.4)


def MobileNetV2_1_0(num_classes=10):
    """Crea MobileNet v2 con multiplicador de ancho 1.0 (modelo estándar)"""
    return MobileNetV2(num_classes=num_classes, width_mult=1.0)


def MobileNetV2_0_75(num_classes=10):
    """Crea MobileNet v2 con multiplicador de ancho 0.75"""
    return MobileNetV2(num_classes=num_classes, width_mult=0.75)


def MobileNetV2_0_5(num_classes=10):
    """Crea MobileNet v2 con multiplicador de ancho 0.5"""
    return MobileNetV2(num_classes=num_classes, width_mult=0.5)


def MobileNetV2_0_35(num_classes=10):
    """Crea MobileNet v2 con multiplicador de ancho 0.35 (modelo más pequeño)"""
    return MobileNetV2(num_classes=num_classes, width_mult=0.35)
