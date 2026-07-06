"""Entrypoint for fine-tuning models."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from video_enhancer.config.loader import load_config
from video_enhancer.config.schema import TaskType
from video_enhancer.data.dataset import VideoPatchDataset
from video_enhancer.models.registry import create_model
from video_enhancer.training.realesrgan_trainer import RealESRGANFineTuner
from video_enhancer.training.nafnet_trainer import NAFNetFineTuner
from video_enhancer.utils.download import download_pretrained_model
from video_enhancer.utils.logging import setup_logging, get_logger

setup_logging()
logger = get_logger("finetune")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune video enhancement models.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    logger.info(f"Setting random seed to {config.seed}")
    torch.manual_seed(config.seed)

    # 1. Download Pretrained Weights if needed
    if config.checkpoint.pretrained_path:
        ckpt_path = Path(config.checkpoint.pretrained_path)
        if not ckpt_path.exists():
            # Attempt to auto-download based on expected filename
            model_name = ckpt_path.stem
            try:
                downloaded_path = download_pretrained_model(model_name, ckpt_path.parent)
                config.checkpoint.pretrained_path = downloaded_path
            except ValueError:
                logger.warning(f"Could not auto-download {model_name}. Ensure it exists at {ckpt_path}.")

    # 2. Data
    logger.info("Setting up datasets...")
    # For a real run, you'd ensure data_dir exists. If it doesn't, we can't train.
    if not config.data.train_dir.exists():
        logger.error(f"Training directory {config.data.train_dir} does not exist!")
        logger.info("Please create it and add some high-resolution .png/.jpg images.")
        return

    train_dataset = VideoPatchDataset(
        data_dir=config.data.train_dir,
        data_config=config.data,
        deg_config=config.degradation,
        scale=4  # Hardcoded for x4 models currently
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=config.data.num_workers > 0,
        prefetch_factor=2 if config.data.num_workers > 0 else None
    )
    
    # Optional Validation (using training data just for demonstration if val doesn't exist)
    val_dir = config.data.val_dir if config.data.val_dir.exists() else config.data.train_dir
    val_dataset = VideoPatchDataset(
        data_dir=val_dir,
        data_config=config.data,
        deg_config=config.degradation,
        scale=4
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 3. Model
    logger.info("Initializing model...")
    model = create_model(config)

    # 4. Optimizer
    # We only pass parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer.name.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas
        )
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=config.optimizer.lr)

    # 5. Trainer
    logger.info("Initializing trainer...")
    if config.task == TaskType.SUPER_RESOLUTION:
        trainer = RealESRGANFineTuner(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader
        )
    elif config.task == TaskType.DEBLURRING:
        from video_enhancer.training.nafnet_trainer import NAFNetFineTuner
        trainer = NAFNetFineTuner(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader
        )
    elif config.task == TaskType.FACE_ENHANCEMENT:
        from video_enhancer.training.gfpgan_trainer import GFPGANFineTuner
        trainer = GFPGANFineTuner(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader
        )
    elif config.task == TaskType.INTERPOLATION:
        from video_enhancer.training.rife_trainer import RIFEFineTuner
        trainer = RIFEFineTuner(
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader
        )
    else:
        raise NotImplementedError(f"Trainer for {config.task.value} not implemented.")

    # 6. Fit
    try:
        trainer.fit()
        logger.info("Finetuning completed successfully.")
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
            logger.info(f"Peak VRAM used: {peak_vram:.2f} MB")
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA Out Of Memory!")
        logger.error("Try reducing patch_size, batch_size, or enabling gradient checkpointing.")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()
