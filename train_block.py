import logging
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import hydra
from omegaconf import OmegaConf
from data import load_and_split_data, Config as DataConfig
from data_gen import generate_data
from model_factory import build_model

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig(DataConfig):
    data_gen_type: str = "contextual"  # "contextual" for block 0; others may use different augmentation.
    epochs: int = 5
    lr: float = 0.001
    optimizer: str = "adam"
    model: list = None  # List of block configurations, e.g., [[m, n, k], [m, n, k], ...]
    num_classes: int = 10  # For MNIST/CIFAR-10; adjust as needed

def train_single_block(model, train_loader, cfg, block_index, device):
    """
    Train a single block of the modular model.
    - Block 0: Uses contextual augmentation (images paired with correct/incorrect one-hot contexts).
    - Intermediate blocks: Use non-contextual augmentation on the output from preceding blocks.
    - Final block: Supervised training using CrossEntropyLoss.
    """
    model.to(device)
    block = model.blocks[block_index]
    block.train()

    is_final_block = (block_index == len(model.blocks) - 1)
    optimizer_cls = optim.Adam if cfg.optimizer.lower() == "adam" else optim.SGD
    optimizer = optimizer_cls(block.parameters(), lr=cfg.lr)

    # Use CrossEntropyLoss for final block; for others, use BCEWithLogitsLoss (forward-forward training).
    criterion = nn.CrossEntropyLoss() if is_final_block else nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss, epoch_metric, batch_count = 0.0, 0.0, 0
        for batch in train_loader:
            # Expect batch to be a tuple (images, labels)
            images, labels = batch if isinstance(batch, (list, tuple)) else (batch, None)
            images = images.to(device)
            if labels is not None:
                labels = labels.to(device)

            # Run preceding blocks with no gradient computation.
            with torch.no_grad():
                x = images
                for idx, prev_block in enumerate(model.blocks[:block_index]):
                    if idx == 0:
                        # For previous contextual block, we require a context.
                        true_context = torch.nn.functional.one_hot(labels, num_classes=cfg.num_classes).float()
                        x = prev_block(x, true_context)
                    else:
                        x = prev_block(x)

            optimizer.zero_grad()

            if block_index == 0 and not is_final_block:
                # For block 0: contextual augmentation using raw images.
                augmented_images, augmented_contexts = generate_data((images, labels), cfg, k=cfg.model[block_index][2])
                augmented_images = augmented_images.to(device)
                augmented_contexts = augmented_contexts.to(device)
                outputs = block(augmented_images, augmented_contexts)
                # Compute forward-forward loss using goodness (sum of squares of outputs)
                B = outputs.size(0) // 2  # First half: positive samples; second half: negatives.
                outputs_flat = outputs.reshape(outputs.size(0), -1)
                goodness = torch.sum(outputs_flat ** 2, dim=-1)
                threshold = outputs_flat.shape[1]
                logits = goodness - threshold
                loss_labels = torch.zeros(B * 2, device=device)
                loss_labels[:B] = 1.0
                loss = criterion(logits, loss_labels)
                positive_goodness = goodness[:B]
                ff_accuracy = (positive_goodness > threshold).float().mean()
                epoch_metric += ff_accuracy.item()
                wandb.log({
                    "block_1_batch_loss": loss.item(),
                    "block_1_ff_accuracy": ff_accuracy.item()
                })
            elif is_final_block:
                # Final block: supervised training.
                outputs = block(x)
                loss = criterion(outputs, labels)
                pred_labels = outputs.argmax(dim=1)
                accuracy = (pred_labels == labels).float().mean()
                epoch_metric += accuracy.item()
                wandb.log({
                    "final_block_batch_loss": loss.item(),
                    "final_block_batch_accuracy": accuracy.item()
                })
            else:
                # Intermediate blocks: use augmentation on x (the output of preceding blocks).
                augmented_x = generate_data((x, labels), cfg, k=cfg.model[block_index][2])
                # If augmentation returns a tuple (contextual) by mistake, take only the images.
                if isinstance(augmented_x, tuple):
                    augmented_x = augmented_x[0]
                augmented_x = augmented_x.to(device)
                outputs = block(augmented_x)
                outputs_flat = outputs.view(outputs.size(0), -1)
                goodness = torch.sum(outputs_flat ** 2, dim=-1)
                threshold = outputs_flat.shape[1]
                logits = goodness - threshold
                B = outputs.size(0) // 2
                loss_labels = torch.zeros(B * 2, device=device)
                loss_labels[:B] = 1.0
                loss = criterion(logits, loss_labels)
                positive_goodness = goodness[:B]
                ff_accuracy = (positive_goodness > threshold).float().mean()
                epoch_metric += ff_accuracy.item()
                wandb.log({
                    f"block_{block_index+1}_batch_loss": loss.item(),
                    f"block_{block_index+1}_ff_accuracy": ff_accuracy.item()
                })

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        avg_metric = epoch_metric / batch_count
        metric_name = "Accuracy" if is_final_block else "FF Accuracy"
        logger.info(
            f"Block [{block_index+1}], Epoch [{epoch}/{cfg.epochs}] - "
            f"Avg Loss: {avg_loss:.4f}, Avg {metric_name}: {avg_metric:.4f}"
        )
        wandb.log({
            f"block_{block_index+1}_epoch_avg_loss": avg_loss,
            f"block_{block_index+1}_epoch_avg_{metric_name.lower().replace(' ', '_')}": avg_metric
        })

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    logger.info("Training single block with config:\n%s", OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True), reinit=True)
    train_loader, _, _ = load_and_split_data(cfg)
    model = build_model(cfg)
    # Example: train the first block (contextual block) - adjust block_index as needed.
    block_index = 0  
    train_single_block(model, train_loader, cfg, block_index, device)
    wandb.finish()

if __name__ == "__main__":
    main()
