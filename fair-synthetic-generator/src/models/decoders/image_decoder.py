import torch
import torch.nn as nn
from src.core.base_module import BaseModule

class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """Simple residual block."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

        out += residual
        return self.relu(out)

class UpsampleBlock(nn.Module):
    """Upsampling block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.upsample(x))))

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.upsample(x))))

class SelfAttention(nn.Module):
    """Self-attention block for images."""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        # Simplified self-attention implementation
        return x + self.gamma * self.value(x)

class ImageDecoder(BaseModule):
    """
    Standard image decoder for generative models.
    
    Transforms latent representations back into image space using
    transposed convolutions or upsampling blocks.
    """
    def __init__(self, latent_dim=512, channels=3, image_size=64, config=None):
        super().__init__(name="image_decoder", config=config)
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        
        # Simple decoder architecture for testing/integration
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 4, 2, 1), # 64x64
            nn.Tanh()
        )

    def forward(self, z):
        """Decode latent vector to image."""
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        return self.deconv(x)

# Aliases for compatibility
ImageVAEDecoder = ImageDecoder
ViTImageDecoder = ImageDecoder
DiffusionImageDecoder = ImageDecoder
