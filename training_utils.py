# --- train_and_sample.py ---
# This is the main script for training the model and performing conditional sampling.
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os # Import the os module for path operations
import imageio # For creating GIF animations
from torch.utils.data import DataLoader, random_split # Import random_split
import xarray as xr
import cartopy.crs as ccrs
import wandb

from plotting_utils import plot_losses

# --- Training Function ---
def train_diffusion_model(model, train_loader, val_loader, diffusion, optimizer, scheduler, config):
    """
    Trains the 3D diffusion U-Net model with memory optimization techniques.
    """
    model.train()
    print("Starting training...")
    train_losses, val_losses = [], []
    os.makedirs(config.model_checkpoint_dir, exist_ok=True)
    os.makedirs(config.loss_plot_dir, exist_ok=True)

    # --- MEMORY SAVING: Initialize GradScaler for mixed precision ---
    use_amp = getattr(config, 'use_amp', False)
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Using Automatic Mixed Precision (AMP).")

    for epoch in range(config.start_epoch, config.epochs):
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} (Training)")
        optimizer.zero_grad() 

        for batch_idx, (x_0, conditions_batch, location_field_batch, land_mask_batch) in enumerate(pbar):
            x_0 = x_0.to(config.device)
            current_land_mask = land_mask_batch.to(config.device)
            
            t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=config.device).long()
            
            # --- MEMORY SAVING: Use autocast for the forward pass ---
            with autocast(enabled=use_amp):
                x_t, true_epsilon = diffusion.noise_images(x_0, t, current_land_mask)
                conditions_input = {k: v.to(config.device) for k, v in conditions_batch.items()}
                loc_field_input = location_field_batch.to(config.device)

                predicted_epsilon = model(x_t, t, current_land_mask, 
                                          conditions=conditions_input, 
                                          location_field=loc_field_input)

                loss = F.mse_loss(predicted_epsilon * current_land_mask, true_epsilon * current_land_mask)
                loss = loss / config.gradient_accumulation_steps
            
            # --- MEMORY SAVING: Scale the loss and backpropagate ---
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # --- MEMORY SAVING: Unscale gradients and step optimizer ---
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_train_loss += loss.item() * config.gradient_accumulation_steps
            pbar.set_postfix(loss=loss.item() * config.gradient_accumulation_steps)

        if (len(train_loader.dataset)) % config.gradient_accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} finished, Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} (Validation)")
            for x_0_val, conditions_batch_val, location_field_batch_val, land_mask_batch_val in val_pbar:
                x_0_val = x_0_val.to(config.device)
                current_land_mask_val = land_mask_batch_val.to(config.device)
                t_val = torch.randint(0, diffusion.timesteps, (x_0_val.shape[0],), device=config.device).long()
                
                with autocast(enabled=use_amp):
                    x_t_val, true_epsilon_val = diffusion.noise_images(x_0_val, t_val, current_land_mask_val)
                    conditions_input_val = {k: v.to(config.device) for k, v in conditions_batch_val.items()}
                    loc_field_input_val = location_field_batch_val.to(config.device)

                    predicted_epsilon_val = model(x_t_val, t_val, current_land_mask_val, 
                                                  conditions=conditions_input_val, 
                                                  location_field=loc_field_input_val)
                    val_loss = F.mse_loss(predicted_epsilon_val * current_land_mask_val, true_epsilon_val * current_land_mask_val)
                
                total_val_loss += val_loss.item()
                val_pbar.set_postfix(val_loss=val_loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1} finished, Average Validation Loss: {avg_val_loss:.4f}")

        if config.use_wandb:
            log_dict = {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
            if scheduler: log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            wandb.log(log_dict)
            
        if scheduler: scheduler.step()
        model.train()

        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(config.model_checkpoint_dir, f"ODA_ch{config.channels}_{config.test_id}_epoch_{epoch+1}.pth")
            print(f"Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(), # Save scaler state
                'train_losses': train_losses, 'val_losses': val_losses
            }, checkpoint_path)
            print("Checkpoint saved.")
            # plot_losses(train_losses, val_losses, os.path.join(config.loss_plot_dir, f"loss_plot_{config.test_id}_epoch_{epoch+1}.png"))

    return train_losses, val_losses

# --- Animation Function for Training Data ---
def create_training_animation(data_tensor, land_mask, config, save_path):
    """
    Creates a GIF animation of the surface layer of the 3D training data.
    """
    print(f"Creating training data animation and saving to {save_path}...")
    frames = []
    # Select the surface layer (depth index 0) for plotting
    land_mask_np = land_mask[0, 0, 0].cpu().numpy() 

    num_rows = config.channels
    
    # Limit number of frames to avoid excessively large GIFs
    max_frames = 100
    data_to_animate = data_tensor[:min(len(data_tensor), max_frames*5):5]

    static_ds = xr.open_dataset(config.filepath_static)
    if not ( config.varname_lat=='lat' and config.varname_lon=='lon'):
        static_ds = static_ds.rename({config.varname_lat:'lat', config.varname_lon:'lon'})
    lat_grid = static_ds['geolat'].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values
    lon_grid = static_ds['geolon'].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values

    for i, sample in enumerate(tqdm(data_to_animate, desc="Generating animation frames")):
        # Select the surface layer (depth index 0) of the sample for plotting
        sample_np_surface = sample[:, 0, :, :].cpu().numpy()

        fig, axes = plt.subplots(
            num_rows, 1, figsize=(8, 4 * num_rows), squeeze=False,
            subplot_kw={'projection':ccrs.Robinson()})

        for c in range(num_rows):
            var_name = "Temperature" if c == 0 else "Salinity"
            cmap = 'coolwarm' if c == 0 else 'plasma'
            
            ax = axes[c, 0]
            masked_data = np.ma.masked_where(land_mask_np == 0, sample_np_surface[c])
            im = ax.pcolormesh(
                lon_grid,lat_grid,sample_np_surface[c],
                cmap=cmap,vmin=0.,vmax=1.,transform=ccrs.PlateCarree())
            ax.set_title(f'Sample {i+1} - Surface {var_name}')
            ax.set_xlabel('Longitude Index'); ax.set_ylabel('Latitude Index')
            plt.colorbar(im, ax=ax, label=f'Normalized {var_name}')

        plt.tight_layout()
        
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    if frames:
        imageio.mimsave(save_path, frames, fps=5)
        print(f"Animation saved successfully to {save_path}")
    else:
        print("No frames were generated for the animation.")