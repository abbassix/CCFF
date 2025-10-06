"""
utils.py

This file contains utility functions that are used in the implementation of the model.
"""

import torch


def layer_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Apply layer normalization to the input tensor.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying layer normalization.
    """
    
    # Flatten the input tensor except for the last dimension
    x_flat = x.reshape(x.size(0), -1)
    
    # divide by the square root of the sum of squares
    sum_squares = torch.sum(x_flat ** 2, dim=1, keepdim=True) + 1e-5
    x_flat = x_flat / torch.sqrt(sum_squares)
    
    # Reshape back to the original shape
    return x_flat.reshape(x.size())
