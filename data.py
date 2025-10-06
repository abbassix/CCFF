"""
data.py

This module provides functions to load, split, and return DataLoader objects for common datasets 
such as MNIST, CIFAR-10, and CIFAR-100. It uses Hydra for configuration management and wandb for logging,
ensuring complete reproducibility by setting seeds for Python, NumPy, and PyTorch.

Note: This module is intended to be imported and used from other scripts (e.g., main.py) via the 
load_and_split_data function. Direct execution of data.py is not supported.
"""

import logging
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import wandb
from omegaconf import OmegaConf
from dataclasses import dataclass

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration Dataclass for Hydra
# ------------------------------------------------------------------------------
@dataclass
class Config:
    seed: int = 42
    dataset: str = "mnist"          # Options: "mnist", "cifar10", "cifar100"
    val_ratio: float = 0.1          # Ratio for validation split from the training set
    root: str = "./data"            # Directory to store/load datasets
    download: bool = True           # Whether to download the dataset if not found
    project: str = "data-loader"    # wandb project name
    batch_size: int = 32            # Batch size for DataLoader
    num_workers: int = 4            # Number of workers for DataLoader

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dataset_exists(dataset_name: str, root: str) -> bool:
    """
    Check if the dataset already exists in the given root directory.
    """
    root_path = Path(root)
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        return (root_path / "MNIST").exists()
    elif dataset_name == "cifar10":
        return (root_path / "cifar-10-batches-py").exists()
    elif dataset_name == "cifar100":
        return (root_path / "cifar-100-python").exists()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def load_dataset(dataset_class, root: str, download: bool, transform) -> tuple:
    """
    Load a dataset given the dataset class, root path, download flag, and transform.
    
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    try:
        train_dataset = dataset_class(root=root, train=True, download=download, transform=transform)
        test_dataset = dataset_class(root=root, train=False, download=download, transform=transform)
        return train_dataset, test_dataset
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_class.__name__}: {e}")
        raise

def split_dataset(dataset, val_ratio: float = 0.1, seed: int = 42) -> tuple:
    """
    Split the dataset into training and validation subsets.
    
    Returns:
        Tuple of (train_subset, val_subset).
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset

# ------------------------------------------------------------------------------
# Core Data Preparation Function
# ------------------------------------------------------------------------------
def load_and_split_data(cfg: Config) -> tuple:
    """
    Load the specified dataset, split it into training, validation, and test sets,
    and create DataLoader objects for each subset.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    
    # Initialize wandb logging
    if wandb.run is None:
        wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True))

    
    download_flag = cfg.download
    if dataset_exists(cfg.dataset, cfg.root):
        logger.info("%s dataset found in %s. Skipping download.", cfg.dataset.upper(), cfg.root)
        download_flag = False
    else:
        logger.info("%s dataset not found in %s. Downloading dataset.", cfg.dataset.upper(), cfg.root)
        download_flag = True

    transform = transforms.ToTensor()
    dataset_map = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    
    dataset_name = cfg.dataset.lower()
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    
    train_dataset, test_dataset = load_dataset(dataset_map[dataset_name], cfg.root, download_flag, transform)
    train_subset, val_subset = split_dataset(train_dataset, val_ratio=cfg.val_ratio, seed=cfg.seed)
    
    # Create DataLoaders for training, validation, and testing
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    wandb.log({
        "train_dataset_size": len(train_dataset),
        "val_subset_size": len(val_subset),
        "test_dataset_size": len(test_dataset)
    })
    
    logger.info("Datasets loaded and split successfully:")
    logger.info("Training set size: %d", len(train_dataset))
    logger.info("Validation set size: %d", len(val_subset))
    logger.info("Test set size: %d", len(test_dataset))
    
    return train_loader, val_loader, test_loader

# ------------------------------------------------------------------------------
# Optional: Entry Point for Basic Testing
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example configuration for testing
    cfg = Config()
    load_and_split_data(cfg)
