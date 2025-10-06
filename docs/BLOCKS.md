# Blocks Module Documentation

## Overview
This module (`blocks.py`) defines the architecture of the blocks you can use in your projects. It currently includes a placeholder model named **PoolConv**. This model is composed of:
- An **optional max-pooling layer** with a kernel size of **m x m** (applied only when m > 1).
- A convolutional layer with **c_out** filters of size **k x k**.

The model is designed to be flexible and robust, featuring input parameter validation, modern Python practices, and structured logging for debugging and testing. It is built using PyTorch's `nn.Module`.

## PoolConv Module

### Description
The **PoolConv** module applies the following operations sequentially:
1. **Max Pooling** (Optional): Reduces the spatial dimensions of the input by applying a max-pooling operation with a kernel size of **m x m**. This operation is applied only if **m > 1**.
2. **Convolution**: Processes the output from the pooling layer (or the input directly if pooling is skipped) with a convolutional layer using **c_out** filters of size **k x k**.

### Constructor Arguments
- **c_in (int)**: Number of input channels. Must be a positive integer.
- **m (int)**: Kernel size for max pooling (m x m). If **m == 1**, no pooling layer is applied. Must be at least 1.
- **c_out (int)**: Number of convolutional kernels (output channels). Must be a positive integer.
- **k (int)**: Kernel size for the convolution (k x k). Must be a positive integer.

### Usage Example
Below is an example of how to instantiate and use the **PoolConv** module:

```python
from blocks import PoolConv
import torch

# Create a PoolConv model with pooling (m > 1)
model_with_pool = PoolConv(c_in=3, m=2, c_out=16, k=3)

# Create a PoolConv model without pooling (m = 1)
model_no_pool = PoolConv(c_in=3, m=1, c_out=16, k=3)

# Create a random input tensor with shape (batch_size, channels, height, width)
x = torch.randn(1, 3, 32, 32)

# Forward pass through the model with pooling
output_with_pool = model_with_pool(x)
print("Output shape with pooling:", output_with_pool.shape)

# Forward pass through the model without pooling
output_no_pool = model_no_pool(x)
print("Output shape without pooling:", output_no_pool.shape)
```
