import logging
import torch
import torch.nn as nn
from contextual_conv import ContextualConv2d
from utils import layer_norm

logger = logging.getLogger(__name__)

class ContextualConvBlock(nn.Module):
    """
    First-layer block using ContextualConv2d with optional pooling.
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int, context_dim: int, pool_size: int = 1):
        super().__init__()
        self.use_pool = pool_size > 1
        if self.use_pool:
            self.pool = nn.MaxPool2d(pool_size)
        self.act = nn.ReLU()

        self.conv = ContextualConv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            c_dim=context_dim
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.use_pool:
                x = self.pool(x)
            x = layer_norm(x)
        x = self.conv(x, context)
        x  = self.act(x)
        return x

class PoolConv(nn.Module):
    """
    Standard convolutional block with optional pooling.
    """
    def __init__(self, c_in: int, c_out: int, kernel_size: int, pool_size: int = 1):
        super().__init__()
        self.use_pool = pool_size > 1
        if self.use_pool:
            self.pool = nn.MaxPool2d(pool_size)
        self.act = nn.ReLU()

        self.conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.use_pool:
                x = self.pool(x)
            x = layer_norm(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class GlobalAvgPoolClassifier(nn.Module):
    """
    Adaptive average pooling followed by linear classification.
    """
    def __init__(self, c_in: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(c_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.pool(x).view(x.size(0), -1)
        logits = self.classifier(x)
        return logits

# Quick debug/testing (prefer pytest for actual testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Test ContextualConvBlock
    batch_size, c_in, c_out, context_dim = 4, 3, 16, 10
    images = torch.randn(batch_size, c_in, 32, 32)
    context = torch.randn(batch_size, context_dim)

    contextual_block = ContextualConvBlock(c_in, c_out, kernel_size=3, context_dim=context_dim, pool_size=2)
    output = contextual_block(images, context)
    logger.info(f"ContextualConvBlock output shape: {output.shape}")

    # Test PoolConv block
    pool_conv_block = PoolConv(c_in=c_out, c_out=32, kernel_size=3, pool_size=2)
    output_pool_conv = pool_conv_block(output)
    logger.info(f"PoolConv output shape: {output_pool_conv.shape}")
