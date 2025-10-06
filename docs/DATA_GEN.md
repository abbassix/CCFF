# Data Generation Module Documentation

## Overview
This module (`data_gen.py`) loads data using the `load_and_split_data` function from `data.py` and applies data generation techniques based on a specified configuration. By default, it implements the **random_noise** method, which augments a batch of data by concatenating it with an equivalent batch of random white noise. If the original batch has shape `(B, C, W, H)`, the augmented batch will have shape `(2*B, C, W, H)`.

## Data Generation Process

### Description
The **random_noise** method performs the following steps:
1. **Data Loading**: Retrieves training data using the `load_and_split_data` function from `data.py`.
2. **Batch Extraction**: Extracts a batch of data from the training DataLoader (shape `(B, C, W, H)`).
3. **Noise Generation**: Generates a batch of white noise with the same shape as the original data.
4. **Concatenation**: Concatenates the original data and the noise along the batch dimension, resulting in a new batch of shape `(2*B, C, W, H)`.

### Configuration
The module uses a Hydra configuration class (`DataGenConfig`) that extends the base configuration (`DataConfig`) from `data.py` with an additional parameter:
- **data_gen_type (str)**: Specifies the data generation method to use. The default value is **"random_noise"**.

## Usage Example
This module is designed to be imported and used within another script rather than executed directly. For instance, in your main training pipeline you can import and call the `generate_data` function as follows:

```python
from data_gen import generate_data, DataGenConfig

# Assume `batch` is a torch.Tensor with shape (B, C, W, H)
# and `cfg` is an instance of DataGenConfig.
augmented_batch = generate_data(batch, cfg)
```

## The script will:
- Load and split the dataset.
- Retrieve a batch of data from the training DataLoader.
- Generate white noise matching the shape of the input batch.
- Concatenate the noise with the original batch, forming an augmented dataset.
- Log the shapes of the original and new batches, and report metrics to wandb.

## Additional Notes
- **Dependencies**: Ensure that you have installed PyTorch, torchvision, hydra-core, wandb, and OmegaConf.
- **Customization**: You can modify the configuration (e.g., via a config.yaml file) to change parameters such as the dataset type, batch size, and the data generation method.