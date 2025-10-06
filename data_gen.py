import torch
import logging
from typing import Optional
from data import load_and_split_data, Config as DataConfig
from dataclasses import dataclass
import wandb
from omegaconf import OmegaConf
import hydra

logger = logging.getLogger(__name__)

@dataclass
class DataGenConfig(DataConfig):
    data_gen_type: str = "contextual"  # default to 'contextual'

def generate_wrong_labels(true_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generate a tensor of wrong labels given the true labels.

    Args:
        true_labels (torch.Tensor): Tensor of true labels (B,)
        num_classes (int): Total number of classes

    Returns:
        torch.Tensor: Tensor of incorrect labels (B,)
    """
    B = true_labels.size(0)
    wrong_labels = torch.randint(0, num_classes - 1, (B,), device=true_labels.device)
    wrong_labels = torch.where(
        wrong_labels >= true_labels, wrong_labels + 1, wrong_labels
    )
    return wrong_labels

def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    One-hot encode labels.

    Args:
        labels (torch.Tensor): Labels tensor (B,)
        num_classes (int): Total number of classes

    Returns:
        torch.Tensor: One-hot encoded labels (B, num_classes)
    """
    return torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

def generate_contextual_data(
    images: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple:
    """
    Generate contextual positive and negative samples.

    Args:
        images (torch.Tensor): Input images tensor (B, C, H, W)
        labels (torch.Tensor): True labels tensor (B,)
        num_classes (int): Total number of classes

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            - Augmented images tensor (2B, C, H, W)
            - Context tensor (2B, num_classes)
    """
    # Positive contexts (correct labels)
    positive_context = one_hot_encode(labels, num_classes)

    # Negative contexts (wrong labels)
    negative_labels = generate_wrong_labels(labels, num_classes)
    negative_context = one_hot_encode(negative_labels, num_classes)

    # Duplicate images for positive and negative pairs
    augmented_images = torch.cat([images, images], dim=0)
    augmented_contexts = torch.cat([positive_context, negative_context], dim=0)

    return augmented_images, augmented_contexts

def generate_data(
    batch: tuple, cfg: DataGenConfig, k: Optional[int] = None
) -> tuple:
    """
    Generate augmented data based on the method specified in cfg.

    Args:
        batch (tuple): Tuple containing (images, labels)
        cfg (DataGenConfig): Configuration
        k (Optional[int]): Unused here, for compatibility

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Augmented images and context tensors
    """
    images, labels = batch
    if cfg.data_gen_type == "contextual":
        return generate_contextual_data(images, labels, cfg.num_classes)
    else:
        raise ValueError(f"Unsupported data_gen_type: {cfg.data_gen_type}")

@hydra.main(config_name="config", config_path=".", version_base="1.1")
def main(cfg: DataGenConfig):
    logger.info("Data Generation Config:\n%s", OmegaConf.to_yaml(cfg))

    train_loader, _, _ = load_and_split_data(cfg)
    images, labels = next(iter(train_loader))

    augmented_images, augmented_contexts = generate_data((images, labels), cfg)
    logger.info("Original batch shape: %s", images.shape)
    logger.info("Augmented images shape: %s", augmented_images.shape)
    logger.info("Context tensor shape: %s", augmented_contexts.shape)

    wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True))
    wandb.log({
        "original_batch_shape": images.shape,
        "augmented_images_shape": augmented_images.shape,
        "augmented_contexts_shape": augmented_contexts.shape
    })
    wandb.finish()

if __name__ == "__main__":
    main()
