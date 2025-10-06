# Data Loader Documentation

## Overview
This module (`data.py`) provides functions to load, split, and return DataLoader objects for common datasets such as MNIST, CIFAR-10, and CIFAR-100. It leverages Hydra for configuration management and wandb for experiment logging. Additionally, it ensures full reproducibility by setting seeds for Python, NumPy, and PyTorch. The module follows best practices by using type hints, robust file handling with `pathlib`, structured logging, and centralized error handling.

**Important**: This module is intended to be imported and used from other scripts (e.g., `main.py`) via the `load_and_split_data` function. Direct execution of `data.py` is only supported for basic testing and not for production use.

## Usage

### Importing and Using `load_and_split_data`
In another Python file, such as `main.py`, import and use the `load_and_split_data` function as follows:

```python
from omegaconf import OmegaConf
from data import load_and_split_data, Config

def main():
    # Create a configuration object using the Config dataclass
    cfg = Config(
        seed=42,
        dataset="cifar10",
        val_ratio=0.1,
        root="./data",
        download=True,
        project="data-loader",
        batch_size=32,        # Batch size for DataLoader
        num_workers=4         # Number of workers for DataLoader
    )
    
    # Print the configuration for verification
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    
    # Load and split the dataset, with logging and wandb tracking
    train_loader, val_loader, test_loader = load_and_split_data(cfg)
    
    # Use the DataLoader objects for training, validation, and testing.
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

if __name__ == "__main__":
    main()
```

### Configuration Options
- **dataset**: The dataset to load ("mnist", "cifar10", or "cifar100").
- **seed**: Random seed for full reproducibility.
- **val_ratio**: Ratio for splitting the training set into validation data.
- **root**: Directory for dataset storage.
- **download**: Automatically set based on dataset availability.
- **project**: The wandb project name for logging experiment details.
- **batch_size**: Batch size for DataLoader objects.
- **num_workers**: Number of workers for DataLoader objects.

## Functionality
The `load_and_split_data` function:
- Displays the configuration using structured logging and OmegaConf.
- Sets the random seed across Python, NumPy, and PyTorch for reproducibility.
- Initializes wandb for experiment logging.
- Checks for dataset availability in the specified root directory using `pathlib`.
- Downloads the dataset if it does not exist locally.
- Applies a default transform (`transforms.ToTensor()`) to the data.
- Splits the training set into training and validation subsets based on the provided `val_ratio`.
- Wraps the training, validation, and test datasets in PyTorch DataLoader objects.
- Logs dataset sizes using wandb.
- Returns DataLoader objects for the training, validation, and test sets.

## Dependencies
Ensure the following packages are installed:
- torch
- torchvision
- hydra-core
- wandb
- omegaconf
- numpy
