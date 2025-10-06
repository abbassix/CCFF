import logging
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from typing import Optional
from blocks import PoolConv, ContextualConvBlock, GlobalAvgPoolClassifier

logger = logging.getLogger(__name__)

class ModularModel(nn.Module):
    def __init__(self, input_channels: int, block_configs: list, num_classes: int, context_dim: int):
        """
        Constructs a modular model with a contextual first block.

        Parameters:
            input_channels (int): Number of input channels (e.g., 3 for CIFAR-10).
            block_configs (list): List of tuples [(m, n, k), ...] specifying pooling size, filters, and kernel size.
            num_classes (int): Number of classes for classification.
            context_dim (int): Dimension of context tensor.
        """
        super().__init__()
        self.blocks = nn.ModuleList()

        current_channels = input_channels

        # First layer is contextual
        m, n, k = block_configs[0]
        self.blocks = nn.ModuleList([
            ContextualConvBlock(c_in=input_channels, c_out=n, kernel_size=k, context_dim=num_classes, pool_size=m)
        ])
        current_channels = n

        # Subsequent layers are standard PoolConv
        for idx, (m, n, k) in enumerate(block_configs[1:], start=1):
            block = PoolConv(c_in=current_channels, c_out=n, kernel_size=k, pool_size=m)
            self.blocks.append(block)
            current_channels = n
            logging.info(f"Added PoolConv block {idx + 1}: c_in={current_channels}, c_out={n}, kernel={k}, pool={m}")

        # Final classifier
        self.blocks.append(GlobalAvgPoolClassifier(current_channels, num_classes))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, layer_index: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters:
            x (torch.Tensor): Input tensor.
            context (torch.Tensor, optional): Context tensor for the first block.
            layer_index (int, optional): Index of block to train specifically.

        Returns:
            torch.Tensor: Output tensor.
        """
        if layer_index is not None:
            with torch.no_grad():
                for idx in range(layer_index):
                    block = self.blocks[idx]
                    x = block(x, context) if idx == 0 else block(x)

            block = self.blocks[layer_index]
            return block(x, context) if layer_index == 0 else block(x)

        for idx, block in enumerate(self.blocks):
            x = block(x, context) if idx == 0 else block(x)
        return x


def build_model(cfg) -> nn.Module:
    """
    Factory function to build a ModularModel with a contextual first block.

    Parameters:
        cfg (OmegaConf): Configuration object including dataset, model blocks, and num_classes.

    Returns:
        ModularModel: Configured modular neural network model.
    """
    dataset = cfg.get("dataset", "cifar10").lower()
    input_channels = 1 if dataset == "mnist" else 3

    block_configs = cfg.get("model", [])
    if not block_configs:
        raise ValueError("Model configuration missing in cfg.model.")

    model = ModularModel(input_channels, block_configs, cfg.num_classes, context_dim=cfg.num_classes)

    # Dummy input validation
    dummy_x = torch.randn(1, input_channels, 32, 32) if dataset != "mnist" else torch.randn(1, 1, 28, 28)
    dummy_context = torch.randn(1, cfg.num_classes)
    with torch.no_grad():
        output = model(dummy_x, context=dummy_context)
    logger.info(f"Model output shape after dummy input: {output.shape}")

    return model

# Quick debugging/testing (replace with dedicated test suite)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cfg_dict = {
        "dataset": "cifar10",
        "model": [
            [2, 16, 3],  # Contextual first layer
            [2, 32, 3],  # Standard subsequent layer
        ],
        "num_classes": 10
    }
    cfg = OmegaConf.create(cfg_dict)
    model = build_model(cfg)

    x = torch.randn(1, 3, 32, 32)
    context = torch.randn(1, 10)
    output = model(x, context=context)
    print("Final output shape:", output.shape)
