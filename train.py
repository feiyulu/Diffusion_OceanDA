# --- train.py ---
# This is the main script for training the model.
import torch
import torch.optim as optim
import os
import argparse
import wandb
from torch.utils.data import DataLoader, random_split
import torch.nn as nn 
import collections
import collections.abc
from accelerate import Accelerator
import json
import re

if not hasattr(collections, 'Container'):
    collections.Container = collections.abc.Container

from config import Config 
from data_utils import get_time_coordinates, load_single_ocean_slice, load_static_data, load_2d_prior_slice
from unet_model import UNet, count_parameters 
from diffusion_process import Diffusion 
from training_utils import train_diffusion_model, create_training_animation
from plotting_utils import plot_losses

# Lazy-loading Dataset class
class LazyOceanDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that loads data from disk on-the-fly ("lazily").
    """
    def __init__(self, config, time_coords, land_mask, location_field, area_weights): 
        self.config = config
        self.original_time_coords = time_coords
        self.land_mask = land_mask
        self.location_field = location_field
        self.area_weights = area_weights

        self.start_index = 0
        if self.config.previous_states:
            # Calculate the maximum number of steps back we need to look.
            max_lag_days = max(state.get("lag_days", 0) for state in self.config.previous_states)
            self.start_index = max(self.start_index, max_lag_days * self.config.training_day_interval)
        
        if self.config.prior_2d_fields:
            max_lag_days = max(field.get("lag_days", 0) for field in self.config.prior_2d_fields)
            self.start_index = max(self.start_index, max_lag_days * self.config.training_day_interval)

        if self.start_index > 0:
            print(f"Dataset will skip the first {self.start_index} time steps to ensure history for all prior states.")
        self.time_coords = self.original_time_coords[self.start_index:]

    def __len__(self):
        # The length of the dataset is the number of time coordinates we can actually use.
        return len(self.time_coords)

    def __getitem__(self, idx):
        time_coord = self.time_coords[idx]
        data_slice, conditions_at_idx = load_single_ocean_slice(self.config, time_coord, return_doy=True)
        data_slice_masked = data_slice * self.land_mask.squeeze(0)
        
        original_idx = self.start_index + idx

        # Initialize all conditional inputs to None
        prev_state_surface, prior_2d_data = None, None

        if self.config.previous_states:
            prev_state_surfaces = []
            for state_config in self.config.previous_states:
                lag_days = state_config.get("lag_days", 0)
                channels_to_use = state_config.get("channels", [0])
                prev_original_idx = original_idx - lag_days * self.config.training_day_interval
                prev_time_coord = self.original_time_coords[prev_original_idx]
                prev_sample, _ = load_single_ocean_slice(self.config, prev_time_coord)
                surface_slice = prev_sample[channels_to_use, 0, :, :]
                prev_state_surfaces.append(surface_slice)
            prev_state_surface = torch.cat(prev_state_surfaces, dim=0)

        if self.config.prior_2d_fields:
            prior_2d_fields_list = []
            for field_config in self.config.prior_2d_fields:
                if not field_config.get("enabled", False):
                    continue
                lag_days = field_config.get("lag_days", 0)
                prev_original_idx = original_idx - lag_days * self.config.training_day_interval
                prev_time_coord = self.original_time_coords[prev_original_idx]
                # This will load all enabled 2D priors for that time coord
                prior_2d_slice = load_2d_prior_slice(self.config, prev_time_coord)
                if prior_2d_slice is not None:
                    prior_2d_fields_list.append(prior_2d_slice)
            if prior_2d_fields_list:
                prior_2d_data = torch.cat(prior_2d_fields_list, dim=0)

        # Ensure conditional inputs are valid tensors, not None, for the dataloader
        if prev_state_surface is None:
            prev_state_surface = torch.empty(0, self.config.data_shape[1], self.config.data_shape[2])
        if prior_2d_data is None:
            prior_2d_data = torch.empty(0, self.config.data_shape[1], self.config.data_shape[2])

        return (data_slice_masked, conditions_at_idx, self.location_field.squeeze(0), 
                self.land_mask.squeeze(0), self.area_weights.squeeze(0), prev_state_surface, prior_2d_data)

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ocean Diffusion Model Training.")
    parser.add_argument("--work_path", "-w", type=str, default=".", help="Working directory")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to the JSON config file.")
    args = parser.parse_args()

    config_path = os.path.join(args.work_path, args.config)
    try:
        config = Config.from_json_file(config_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(e); raise

    accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)

    if config.use_wandb and accelerator.is_main_process:
        mode = "offline" if config.wandb_offline else "online"
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=vars(config), mode=mode)
        wandb.run.name = config.test_id
        print(f"Weights & Biases initialized in '{mode}' mode.")

    device = accelerator.device
    print(f"Using device: {device}")
    
    if accelerator.is_main_process:
        print("Preparing data using lazy-loading...")
    
    land_mask, location_field_data, area_weights = load_static_data(config)
    if accelerator.is_main_process:
        print(f"Static land mask tensor shape: {land_mask.shape}")
    
    time_coords = get_time_coordinates(config)
    if accelerator.is_main_process:
        print(f"Found {len(time_coords)} total time steps for training.")

    full_dataset = LazyOceanDataset(config, time_coords, land_mask, location_field_data, area_weights)
    
    val_size = int(config.validation_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if config.generate_training_animation and accelerator.is_main_process:
        print("Warning: Training animation generation is disabled for lazy-loading mode.")

    model = UNet(config, verbose_init=accelerator.is_main_process)
    print(f"Total trainable parameters in UNet model: {count_parameters(model):,}")

    if config.use_wandb and accelerator.is_main_process:
        wandb.watch(model, log='all', log_freq=200)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = None
    if config.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.lr_scheduler_T_max, eta_min=config.lr_scheduler_eta_min
        )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    diffusion = Diffusion(
        timesteps=config.timesteps, beta_start=config.beta_start, beta_end=config.beta_end, device=device
    )

    start_epoch, train_losses, val_losses = 0, [], []
    latest_checkpoint_dir = None
    if os.path.exists(config.model_checkpoint_dir):
        epoch_dirs = sorted([d for d in os.listdir(config.model_checkpoint_dir) if d.startswith("epoch_") and os.path.isdir(os.path.join(config.model_checkpoint_dir, d))], 
                            key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)), reverse=True)
        if epoch_dirs:
            latest_checkpoint_dir = os.path.join(config.model_checkpoint_dir, epoch_dirs[0])
            # Verify the checkpoint is valid before proceeding
            if not (os.path.exists(os.path.join(latest_checkpoint_dir, "pytorch_model.bin")) or \
                    os.path.exists(os.path.join(latest_checkpoint_dir, "model.safetensors"))):
                latest_checkpoint_dir = None

    if latest_checkpoint_dir:
        print(f"Resuming training from checkpoint: {latest_checkpoint_dir}...")
        accelerator.load_state(latest_checkpoint_dir)
        
        state_path = os.path.join(latest_checkpoint_dir, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            start_epoch = training_state['epoch'] + 1
            train_losses = training_state.get('train_losses', [])
            val_losses = training_state.get('val_losses', [])
            print(f"Training will resume from epoch {start_epoch}.")
    else:
        print("No existing checkpoint found. Starting training from scratch.")

    config.start_epoch = start_epoch
    
    train_losses, val_losses = train_diffusion_model(
        accelerator, model, train_loader, val_loader, diffusion, optimizer, scheduler, config, train_losses, val_losses
    )

    if config.save_model_after_training and accelerator.is_main_process:
        final_checkpoint_path = os.path.join(config.model_checkpoint_dir, f"ODA_ch{config.channels}_{config.test_id}_epoch_{config.epochs+1}.pth")
        print(f"Saving final model checkpoint to {final_checkpoint_path}...")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': config.epochs, 
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(config)
        }, final_checkpoint_path)
        print("Final model checkpoint saved.")

    if train_losses and val_losses and accelerator.is_main_process:
        plot_filename = f"loss_plot_{config.test_id}_final.png"
        plot_losses(train_losses, val_losses, os.path.join(config.loss_plot_dir, plot_filename))

    if config.use_wandb and accelerator.is_main_process:
        wandb.finish()

    print("\nTraining complete.")
