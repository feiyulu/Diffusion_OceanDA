# --- train.py ---
# This is the main script for training the model.
import torch
import torch.optim as optim
import os
import argparse
import wandb
from torch.utils.data import DataLoader, random_split
import torch.nn as nn # Import the nn module
import collections
import collections.abc
# This resolves the "AttributeError: module 'collections' has no attribute 'Container'"
if not hasattr(collections, 'Container'):
    collections.Container = collections.abc.Container

# Import components from other files
from config import Config # Already imported
from data_utils import load_ocean_data # Already imported
from unet_model import UNet, count_parameters # count_parameters imported for model size
from diffusion_process import Diffusion # Already imported
from training_utils import train_diffusion_model, create_training_animation
from plotting_utils import plot_losses

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ocean Diffusion Model Training.")
    parser.add_argument("--work_path", "-w", type=str, default=".", help="Working directory")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to the JSON config file.")
    args = parser.parse_args()

    config_path = os.path.join(args.work_path, args.config)
    try:
        config = Config.from_json_file(config_path)
    except FileNotFoundError as e:
        print(e)
        raise Exception(f"Please ensure '{config_path}' exists.")

    if config.use_wandb:
        mode = "offline" if config.wandb_offline else "online"
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=vars(config), mode=mode)
        wandb.run.name = config.test_id
        print(f"Weights & Biases initialized in '{mode}' mode.")

    print(f"Using device: {config.device}")
    
    # --- 1. Prepare Data ---
    data, land_mask, conditional_data, location_field_data, _, _ = load_ocean_data(
        config, time_range=config.training_day_range, is_test_data=False
    )
    print(f"Data tensor shape: {data.shape}")
    print(f"Land mask tensor shape: {land_mask.shape}")

    class CustomOceanDataset(torch.utils.data.Dataset):
        def __init__(self, data, conditional_data, location_field, land_mask): 
            self.data = data
            self.conditional_data = conditional_data
            self.location_field = location_field
            self.land_mask = land_mask

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            conditions_at_idx = {key: val[idx] for key, val in self.conditional_data.items()}
            # The location field and mask are static across all samples
            return self.data[idx], conditions_at_idx, self.location_field.squeeze(0), self.land_mask.squeeze(0)

    full_dataset = CustomOceanDataset(data, conditional_data, location_field_data, land_mask)
    val_size = int(config.validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    if config.generate_training_animation:
        create_training_animation(data, land_mask, config)

    # --- 2. Initialize Model and Diffusion Process ---
    # UNet is now initialized directly with the config object
    model = UNet(config, verbose_init=False).to(config.device)

    use_data_parallel = getattr(config, 'use_data_parallel', False)
    if use_data_parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training via DataParallel.")
        model = nn.DataParallel(model)

    print(f"Total trainable parameters in UNet model: {count_parameters(model):,}")

    if config.use_wandb:
        wandb.watch(model, log='all', log_freq=200)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.lr_scheduler_T_max, eta_min=config.lr_scheduler_eta_min
        )

    # Diffusion process no longer needs data_shape at initialization
    diffusion = Diffusion(
        timesteps=config.timesteps, beta_start=config.beta_start, beta_end=config.beta_end, device=config.device
    )

    # --- 3. Load Checkpoint and Train ---
    start_epoch, train_losses, val_losses = 0, [], []
    latest_checkpoint_path = None
    if os.path.exists(config.model_checkpoint_dir):
        checkpoints = [f for f in os.listdir(config.model_checkpoint_dir) if f.startswith(f"ODA_ch{config.channels}_{config.test_id}") and f.endswith(".pth")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
            latest_checkpoint_path = os.path.join(config.model_checkpoint_dir, checkpoints[-1])

    if latest_checkpoint_path:
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=config.device)

        state_dict = checkpoint['model_state_dict']
        if use_data_parallel and not isinstance(model, nn.DataParallel):
             # If checkpoint was saved with DataParallel but we are loading without it
             state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not use_data_parallel and isinstance(model, nn.DataParallel):
             # If checkpoint was saved without DataParallel but we are loading with it
             state_dict = {'module.' + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Training will resume from epoch {start_epoch}.")
    else:
        print("No existing checkpoint found. Starting training from scratch.")

    # Add start_epoch to config for the training function
    config.start_epoch = start_epoch
    
    # Train the model
    train_losses, val_losses = train_diffusion_model(
        model, train_loader, val_loader, diffusion, optimizer, scheduler, config
    )

    # --- 4. Finalize ---
    if config.save_model_after_training:
        final_checkpoint_path = os.path.join(config.model_checkpoint_dir, f"ODA_ch{config.channels}_{config.test_id}_epoch_{config.epochs+1}.pth")
        print(f"Saving final model checkpoint to {final_checkpoint_path}...")

        model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        torch.save({
            'epoch': config.epochs, 
            'model_state_dict': model_state_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(config)
        }, final_checkpoint_path)
        print("Final model checkpoint saved.")

    if train_losses and val_losses:
        plot_filename = f"loss_plot_{config.test_id}_final.png"
        plot_losses(train_losses, val_losses, os.path.join(config.loss_plot_dir, plot_filename))

    if config.use_wandb:
        wandb.finish()

    print("\nTraining complete.")