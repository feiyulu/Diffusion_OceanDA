# --- main.py ---
# This is the main script for training the model and performing conditional sampling.
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import xarray as xr
import os # Import the os module for path operations
import argparse
import imageio # For creating GIF animations
from torch.utils.data import DataLoader, random_split # Import random_split

import wandb

# Import components from other files
from config import Config # Already imported
from data_utils import load_ocean_data # Already imported
from unet_model import UNet, count_parameters # count_parameters imported for model size
from diffusion_process import Diffusion # Already imported
from training_utils import train_diffusion_model, plot_losses, create_training_animation
from sampling_utils import sample_conditional

# --- Main Execution Block ---
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Ocean Diffusion Model Training and Sampling.")
    
    parser.add_argument(
        "--work_path", "-w",
        type=str,
        help="Working directory"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.json", # Default name for your config file
        help="Path to the JSON configuration file."
    )

    args = parser.parse_args()

    # Initialize the config object by loading from a JSON file
    try:
        config = Config.from_json_file(f"{args.work_path}/{args.config}")
        if os.path.realpath(config.output_dir) != os.path.realpath(args.work_path):
            raise Exception('Working directory does not match case name.')
    except FileNotFoundError as e:
        print(e)
        raise Exception("Please ensure 'config.json' exists")

    if config.use_wandb:
        # Set the mode to 'offline' if the config flag is True
        mode = "offline" if config.wandb_offline else "online"
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config=vars(config), # Log all config parameters automatically
            mode=mode
        )
        # Give the run a more descriptive name
        wandb.run.name = config.test_id
        print(f"Weights & Biases initialized in '{mode}' mode.")

    print(f"Using device: {config.device}")
    print(f"Model will use {config.channels} channel(s). Salinity included: {config.use_salinity}")
    print(f"Sampling method: {config.sampling_method.upper()}")
    if config.sampling_method.upper()=="DDIM":
        print(f"DDIM stochastic eta: {config.ddim_eta}")
    print(f"Observation fidelity weight: {config.observation_fidelity_weight}")
    print(f"Generate training animation: {config.generate_training_animation}")
    print(f"Generalized conditioning configs: {config.conditioning_configs}")
    print(f"Generalized location embedding: {config.location_embedding_types} (channels: {config.location_embedding_channels})")
    print(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    print(f"Checkpoint save interval: {config.save_interval} epochs")

    # 1. Prepare Data
    data, land_mask, conditional_data, location_field_data, T_range, S_range = load_ocean_data(
        config,
        time_range=config.training_day_range,
        is_test_data=False
    )

    print(f"Data size: {data.shape}")
    print(f"Land mask size: {land_mask.shape}")
    if config.conditioning_configs:
        for key in config.conditioning_configs:
            print(f"{key} data size: {conditional_data[key].shape}")
    if config.location_embedding_types: print(f"Location Field data size: {location_field_data.shape}")

    # Create a custom dataset that yields the additional embeddings
    class CustomOceanDataset(torch.utils.data.Dataset):
        def __init__(self, data, conditional_data, location_field): 
            self.data = data
            self.conditional_data = conditional_data # This is a dict, e.g., {'dayofyear': tensor, 'co2': tensor}
            self.location_field = location_field

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # For each item, create a dictionary of the conditional values at that index
            conditions_at_idx = {key: val[idx] for key, val in self.conditional_data.items()}
            # Location field is static, so we just return it (squeezed to remove time dim)
            return self.data[idx], conditions_at_idx, self.location_field.squeeze(0)

    # Initialize CustomOceanDataset with the loaded data
    full_dataset = CustomOceanDataset(data, conditional_data, location_field_data)

    # Split dataset into training and validation sets
    val_size = int(config.validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for batching and shuffling data during training
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) # No shuffling for validation
    
    # Move the static land_mask to the chosen device once
    land_mask = land_mask.to(config.device)

    # Generate training data animation if enabled
    if config.generate_training_animation:
        create_training_animation(data, land_mask, config.channels, config.use_salinity, config.training_animation_path)

    # 2. Initialize Model and Diffusion Process
    model = UNet(
        in_channels=config.channels, 
        out_channels=config.channels, 
        base_channels=config.base_unet_channels,
        conditioning_configs=config.conditioning_configs,
        location_embedding_channels=config.location_embedding_channels
    ).to(config.device)

    if config.use_wandb:
        wandb.watch(model, log='all', log_freq=200)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Initialize the Learning Rate Scheduler ---
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.lr_scheduler_T_max,
            eta_min=config.lr_scheduler_eta_min
        )
        print(f"Initialized CosineAnnealingLR scheduler with T_max={config.lr_scheduler_T_max} and eta_min={config.lr_scheduler_eta_min}.")

    start_epoch = 0

    # Logic to find and load the latest checkpoint
    latest_checkpoint_path = None
    if os.path.exists(config.model_checkpoint_dir):
        checkpoints = [f for f in os.listdir(config.model_checkpoint_dir) if f.startswith(f"ODA_ch{config.channels}_{config.test_id}_epoch_") and f.endswith(".pth")]
        if checkpoints:
            # Sort by epoch number to find the latest
            checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
            latest_checkpoint_path = os.path.join(config.model_checkpoint_dir, checkpoints[-1])
            print(f"Found latest checkpoint: {latest_checkpoint_path}")

    if latest_checkpoint_path and os.path.exists(latest_checkpoint_path): # Resume training
        print(f"Resuming training from checkpoint: {latest_checkpoint_path}...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
        print(f"Training will resume from epoch {start_epoch}.")
    else:
        print(f"No existing checkpoint found in {config.model_checkpoint_dir}. Starting training from scratch.")
        start_epoch = 0 # Ensure training starts from 0 if no checkpoint is found
    
    print(f"\nTotal trainable parameters in UNet model: {count_parameters(model):,}") # Print model size

    diffusion = Diffusion(
        timesteps=config.timesteps, 
        beta_start=config.beta_start, 
        beta_end=config.beta_end, 
        data_shape=config.data_shape, 
        device=config.device) # Instantiate the Diffusion utility

    # 3. Train the model (if not loading for sampling only)
    train_losses, val_losses = [], [] # Initialize loss lists
    train_losses, val_losses = train_diffusion_model(
        model, train_loader, val_loader, diffusion, optimizer, scheduler,
        config.epochs, config.device, land_mask, 
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        start_epoch=start_epoch, 
        save_interval=config.save_interval, 
        checkpoint_dir=config.model_checkpoint_dir, 
        test_id=config.test_id,
        channels=config.channels,
        loss_plot_dir=config.loss_plot_dir,
        config=config
        )

    # Save model after training if configured
    if config.save_model_after_training:
        final_checkpoint_path = os.path.join(config.model_checkpoint_dir, 
                                                f"ODA_ch{config.channels}_{config.test_id}_epoch_{config.epochs}.pth")
        print(f"Saving final model checkpoint to {final_checkpoint_path}...")
        torch.save({
            'epoch': config.epochs, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_checkpoint_path)
        print("Final model checkpoint saved.")

    # Plot training and validation losses after training
    if train_losses and val_losses: # Only plot if training actually occurred and losses were collected
        plot_filename = f"loss_plot_{config.test_id}_epoch_{config.epochs}.png"
        specific_plot_path = os.path.join(config.loss_plot_dir, plot_filename)
        plot_losses(train_losses, val_losses, specific_plot_path)

    if config.use_wandb:
        wandb.finish()