import logging
import torch
import wandb
import hydra
from omegaconf import OmegaConf
from data import load_and_split_data
from model_factory import build_model
from train_block import train_single_block

logger = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg):
    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    wandb.init(
        project=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Load data and build model.
    train_loader, _, _ = load_and_split_data(cfg)
    model = build_model(cfg).to(device)
    
    num_blocks = len(model.blocks)
    logger.info("Starting sequential block training (%d blocks)", num_blocks)
    
    # Sequentially train each block.
    for block_idx in range(num_blocks):
        logger.info("Training block %d/%d", block_idx + 1, num_blocks)
        train_single_block(model, train_loader, cfg, block_idx, device)
    
    wandb.finish()

if __name__ == "__main__":
    main()
