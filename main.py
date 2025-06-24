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

# Import components from other files
from config import Config # Already imported
from data_utils import generate_synthetic_ocean_data, load_ocean_data # Already imported
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
    # This makes it easy to change settings without modifying code.
    try:
        config = Config.from_json_file(f"{args.work_path}/{args.config}")
        if os.path.realpath(config.output_dir) != os.path.realpath(args.work_path):
            raise Exception('Working directory does not match case name.')
    except FileNotFoundError as e:
        print(e)
        print("Please ensure 'config.json' exists in the same directory as 'main.py'.")
        print("Falling back to default configuration.")
        config = Config(
            test_id='test2',
            image_size=(128, 128),
            real_data=True,
            filepath_t=[f'/scratch/cimes/feiyul/Ocean_Data/obs_data/sst/sst.day.{year}.1x1.nc' for year in range(2003,2024)],
            filepath_s=[f'/scratch/cimes/feiyul/Ocean_Data/obs_data/sss/SSS.day.{year}.1x1.nc' for year in range(2003,2024)],
            filepath_t_test='/scratch/cimes/feiyul/Ocean_Data/obs_data/sst/sst.day.2024.1x1.nc',
            filepath_s_test='/scratch/cimes/feiyul/Ocean_Data/obs_data/sss/SSS.day.2024.1x1.nc',
            varname_t='sst',
            varname_s='sss',
            lat_range=[26,154],
            lon_range=[120,248],
            training_day_range=[0,10000,1],
            sample_day=270,
            use_salinity=False,
            timesteps=1000,
            beta_start=1e-5,
            beta_end=0.01,
            unet_width=64,
            epochs=100,
            batch_size=200,
            learning_rate=1e-5,
            data_points=500,
            validation_split=0.1,
            channel_wise_normalization=True,
            sampling_method='ddpm',
            observation_fidelity_weight=1.0,
            observation_samples=1000,
            save_model_after_training=True,
            load_model_for_sampling=False
        )

    print(f"Using device: {config.device}")
    print(f"Model will use {config.channels} channel(s). Salinity included: {config.use_salinity}")
    print(f"Sampling method: {config.sampling_method.upper()}")
    print(f"Observation fidelity weight: {config.observation_fidelity_weight}")
    print(f"Generate training animation: {config.generate_training_animation}")

    # 1. Prepare Data
    # Pass image_size as a tuple and the channel_wise_normalization flag
    if config.real_data:
        data, land_mask, min_val, max_val = load_ocean_data(
            config.filepath_t,
            config.image_size, config.channels,
            time_range=config.training_day_range,
            lat_range=config.lat_range,lon_range=config.lon_range,
            filepath_s=config.filepath_s,
            variable_names=[config.varname_t, config.varname_s],
            min_val_in=[config.T_range[0],config.S_range[0]],
            max_val_in=[config.T_range[1],config.S_range[1]]
        )
    else:
        data, land_mask = generate_synthetic_ocean_data(
            config.data_points, config.image_size, config.channels, 
            config.use_salinity, config.channel_wise_normalization
        )

    print(f"Data size: {data.shape}")
    print(f"Land mask size: {land_mask.shape}")

    # Split data into training and validation sets
    val_size = int(config.validation_split * len(data))
    train_size = len(data) - val_size
    train_data, val_data = random_split(data, [train_size, val_size])

    # Create DataLoaders for batching and shuffling data during training
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False) # No shuffling for validation
    
    # Move the static land_mask to the chosen device once
    land_mask = land_mask.to(config.device)

    # NEW: Generate training data animation if enabled
    if config.generate_training_animation:
        create_training_animation(data, land_mask, config.channels, config.use_salinity, config.training_animation_path)

    # 2. Initialize Model and Diffusion Process
    # Pass image_size to UNet constructor
    model = UNet(
        in_channels=config.channels, 
        out_channels=config.channels, 
        image_size=config.image_size,
        base_channels=config.unet_width).to(config.device) # Instantiate the U-Net model and move to device
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate) # Initialize Adam optimizer
    start_epoch = 0 # Default start epoch

    # Load model if configured to do so
    if config.load_model_for_sampling:
        if os.path.exists(config.model_checkpoint_path):
            print(f"Loading model from {config.model_checkpoint_path}...")
            checkpoint = torch.load(config.model_checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] # The epoch when the model was saved
            print(f"Model loaded. Training will resume from epoch {start_epoch + 1} (or skip if training not desired).")
        else:
            print(f"No checkpoint found at {config.model_checkpoint_path}. Starting training from scratch.")
            start_epoch = 0 # Ensure training starts from 0 if no checkpoint is found
    
    print(f"\nTotal trainable parameters in UNet model: {count_parameters(model):,}") # Print model size

    # Pass image_size to Diffusion constructor
    diffusion = Diffusion(
        timesteps=config.timesteps, 
        beta_start=config.beta_start, 
        beta_end=config.beta_end, 
        img_size=config.image_size, 
        device=config.device) # Instantiate the Diffusion utility

    # 3. Train the model (if not loading for sampling only)
    train_losses, val_losses = [], [] # Initialize loss lists
    if not config.load_model_for_sampling or start_epoch == 0: 
        train_losses, val_losses = train_diffusion_model(model, train_loader, val_loader, diffusion, optimizer, config.epochs, config.device, land_mask, start_epoch)

        # Save model after training if configured
        if config.save_model_after_training:
            print(f"Saving model checkpoint to {config.model_checkpoint_path}...")
            torch.save({
                'epoch': config.epochs, # Save the last epoch value
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, config.model_checkpoint_path)
            print("Model checkpoint saved.")
    else:
        print("Skipping training as 'load_model_for_sampling' is True and a model was loaded.")

    # Plot training and validation losses after training
    if train_losses and val_losses: # Only plot if training actually occurred and losses were collected
        plot_losses(train_losses, val_losses, config.loss_plot_save_path)


    # 4. Prepare Sparse Observations for Conditional Sampling
    img_h, img_w = config.image_size
    sample_shape = (1, config.channels, img_h, img_w)
    
    # Generate a new "true" sample to serve as the ground truth for observations
    # Set num_samples=1 as we only need one for this demonstration
    if config.real_data:
        true_sample, _, _, _ = load_ocean_data(
            config.filepath_t_test,
            config.image_size, config.channels,
            time_range=config.sample_day_range,
            lat_range=config.lat_range,lon_range=config.lon_range,
            filepath_s=config.filepath_s_test,
            variable_names=[config.varname_t, config.varname_s],
            min_val_in=[config.T_range[0],config.S_range[0]],
            max_val_in=[config.T_range[1],config.S_range[1]]
        )
    else:
        true_sample, _ = generate_synthetic_ocean_data(
            num_samples=1,
            image_size=config.image_size,
            channels=config.channels,
            use_salinity=config.use_salinity,
            channel_wise_normalization=config.channel_wise_normalization
        )
    true_sample = true_sample.to(config.device) # Move to device

    # Initialize observation tensors based on the shape of the true_sample
    observations_for_sampling = torch.zeros_like(true_sample)
    observed_mask_for_sampling = torch.zeros_like(true_sample, dtype=torch.bool)

    # Randomly select a few observation points from the true_sample
    num_obs_points = config.observation_samples # Number of random observation points
    
    # Find all valid ocean coordinates
    ocean_coords_h, ocean_coords_w = np.where(land_mask[0, 0].cpu().numpy() == 1)
    
    # Combine into a list of (y, x) tuples
    all_ocean_points = list(zip(ocean_coords_h, ocean_coords_w))

    if len(all_ocean_points) < num_obs_points:
        print(f"Warning: Not enough ocean points ({len(all_ocean_points)}) to sample {num_obs_points} observations. Sampling all available ocean points.")
        sampled_indices = np.arange(len(all_ocean_points))
    else:
        sampled_indices = np.random.choice(len(all_ocean_points), num_obs_points, replace=False)
    
    obs_points_actual = []
    for idx in sampled_indices:
        y, x = all_ocean_points[idx]
        for c in range(config.channels): # Sample for each enabled channel
            val = true_sample[0, c, y, x].item() # Get the true value at this point
            observations_for_sampling[0, c, y, x] = val
            observed_mask_for_sampling[0, c, y, x] = True
            obs_points_actual.append((c, y, x, val)) # Store the actual sampled point

    # print("\nSparse observations defined (sampled from generated true image):")
    # for c, y, x, val in obs_points_actual:
    #     channel_name = "Temperature" if c == 0 else "Salinity"
        # print(f"- {channel_name} at ({y},{x}): {val:.4f}")

    # 5. Perform Conditional Sampling
    generated_samples = sample_conditional(
        model,
        diffusion,
        num_samples=1,
        observations=observations_for_sampling,
        observed_mask=observed_mask_for_sampling,
        land_mask=land_mask,
        channels=config.channels,
        image_size=config.image_size, # Pass the tuple
        device=config.device,
        sampling_method=config.sampling_method, # Use method from config
        observation_fidelity_weight=config.observation_fidelity_weight # Use weight from config
    )

    ds_t_training = xr.open_mfdataset(config.filepath_t)
    da_t_training = ds_t_training[config.varname_t]
    clim_t = da_t_training.groupby('time.dayofyear').mean('time')
    clim_t_pred = (clim_t.isel(
        lat=slice(config.lat_range[0],config.lat_range[1]),
        lon=slice(config.lon_range[0],config.lon_range[1]),
        dayofyear=config.sample_day_range[0]).values-config.T_range[0])/\
            (config.T_range[1]-config.T_range[0])
    print(f'Climatological Temperature Shape: {clim_t_pred.shape}')

    ds_s_training = xr.open_mfdataset(config.filepath_s)
    da_s_training = ds_s_training[config.varname_s]
    clim_s = da_s_training.groupby('time.dayofyear').mean('time')
    clim_s_pred = (clim_s.isel(
        lat=slice(config.lat_range[0],config.lat_range[1]),
        lon=slice(config.lon_range[0],config.lon_range[1]),
        dayofyear=config.sample_day_range[0]).values-config.S_range[0])/\
            (config.S_range[1]-config.S_range[0])

    # 6. Visualize and Save Results
    print("\nVisualizing and saving results...")
    generated_sample_np = generated_samples[0].cpu().numpy()
    true_sample_np = true_sample[0].cpu().numpy() # Convert the true sample to numpy
    land_mask_np = land_mask[0, 0].cpu().numpy()

    # Determine number of subplots dynamically: 2 rows (True, Generated), N channels
    num_cols = config.channels if config.use_salinity else 1
    plt.figure(figsize=(16, 18)) # Adjust figure size

    # Plot True Temperature Map
    plt.subplot(3, 2, 1) 
    masked_true_temp = np.ma.masked_where(land_mask_np == 0, true_sample_np[0])
    plt.imshow(masked_true_temp, cmap='viridis', origin='lower', vmin=0., vmax=1.)
    plt.colorbar(label='Normalized Temperature')
    plt.title('True Temperature Map (Ocean Only)')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')

    plt.subplot(3, 2, 2) 
    # Overlay sampled observation points for temperature on true map
    for c, y, x, val in obs_points_actual:
        if c == 0 and land_mask_np[y, x] == 1: # Only plot if it was on ocean
            plt.scatter(x, y, c=val, cmap='viridis', vmin=0., vmax=1., marker='o', s=5)
            # plt.text(x + 1, y + 1, f'{val:.2f}', color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    plt.colorbar(label='Normalized Temperature')
    plt.title('True Temperature Samples (Ocean Only)')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')

    # Plot Climatology Temperature Map
    plt.subplot(3, 2, 3) # Position below true temp map
    rmse = np.sqrt(np.mean((clim_t_pred-masked_true_temp)**2))
    print(f'Climatological Temperature RMSE: {rmse}')
    plt.imshow(clim_t_pred, cmap='viridis', origin='lower', vmin=0., vmax=1.)
    plt.colorbar(label='Normalized Temperature')
    plt.title(f'Climatological Temperature Map (Ocean Only) RMSE:{rmse:.3f}')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')

    plt.subplot(3, 2, 4) # Position below true temp map
    mae = np.mean(np.abs(clim_t_pred-masked_true_temp))
    print(f'Climatological Temperature MAE: {mae}')
    plt.imshow(clim_t_pred-masked_true_temp, cmap='bwr', origin='lower', vmin=-1, vmax=1.)
    plt.colorbar(label='Normalized Temperature')
    plt.title(f'Climatological Temperature Map (Ocean Only) MAE:{mae:.3f}')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')

    # Plot Generated Temperature Map
    plt.subplot(3, 2, 5) # Position below true temp map
    masked_generated_temp = np.ma.masked_where(land_mask_np == 0, generated_sample_np[0])
    rmse = np.sqrt(np.mean((masked_generated_temp-masked_true_temp)**2))
    print(f'Generated Temperature RMSE: {rmse}')
    plt.imshow(masked_generated_temp, cmap='viridis', origin='lower', vmin=0., vmax=1.)
    plt.colorbar(label='Normalized Temperature')
    plt.title(f'Generated Temperature Map (Ocean Only) RMSE:{rmse:.3f}')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    # # Overlay sampled observation points for temperature on generated map
    # for c, y, x, val in obs_points_actual:
    #     if c == 0 and land_mask_np[y, x] == 1:
    #         plt.scatter(x, y, color='black', marker='o', s=1)
    #         # plt.text(x + 1, y + 1, f'{val:.2f}', color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.subplot(3, 2, 6) # Position below true temp map
    mae = np.mean(np.abs(masked_generated_temp-masked_true_temp))
    print(f'Generated Temperature MAE: {mae}')
    plt.imshow(masked_generated_temp-masked_true_temp, cmap='bwr', origin='lower', vmin=-1., vmax=1.)
    plt.colorbar(label='Normalized Temperature')
    plt.title(f'Generated Temperature Map (Ocean Only) MAE:{mae:.3f}')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    # Overlay sampled observation points for temperature on generated map
    for c, y, x, val in obs_points_actual:
        if c == 0 and land_mask_np[y, x] == 1:
            plt.scatter(x, y, color='black', marker='o', s=1)
            # plt.text(x + 1, y + 1, f'{val:.2f}', color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    # Plot Salinity Maps if enabled
    if config.use_salinity:
        # Plot True Salinity Map
        plt.subplot(3, num_cols, 2) # Position next to true temp map
        masked_true_sal = np.ma.masked_where(land_mask_np == 0, true_sample_np[1])
        plt.imshow(masked_true_sal, cmap='plasma', origin='lower', vmin=0., vmax=1.)
        plt.colorbar(label='Normalized Salinity')
        plt.title('True Salinity Map (Ocean Only)')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        # Overlay sampled observation points for salinity on true map
        for c, y, x, val in obs_points_actual:
            if c == 1 and land_mask_np[y, x] == 1:
                plt.scatter(x, y, c=val, cmap='viridis', vmin=0., vmax=1., marker='o', s=5)
                # plt.text(x + 1, y + 1, f'{val:.2f}', color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='blue', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

        # Plot Climatology Salinity Map
        plt.subplot(3, num_cols, 2 + num_cols) # Position below true sal map
        rmse = np.sqrt(np.mean((clim_s_pred-masked_true_sal)**2))
        plt.imshow(clim_s_pred, cmap='plasma', origin='lower', vmin=0., vmax=1.)
        plt.colorbar(label='Normalized Salinity')
        plt.title(f'Generated Salinity Map (Ocean Only) RMSE:{rmse:.3f}')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')

        # Plot Generated Salinity Map
        plt.subplot(3, num_cols, 2 + num_cols*2) # Position below true sal map
        masked_generated_sal = np.ma.masked_where(land_mask_np == 0, generated_sample_np[1])
        rmse = np.sqrt(np.mean((masked_generated_sal-masked_true_sal)**2))
        plt.imshow(masked_generated_sal, cmap='plasma', origin='lower', vmin=0., vmax=1.)
        plt.colorbar(label='Normalized Salinity')
        plt.title(f'Generated Salinity Map (Ocean Only) RMSE:{rmse:.3f}')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        # Overlay sampled observation points for salinity on generated map
        for c, y, x, val in obs_points_actual:
            if c == 1 and land_mask_np[y, x] == 1:
                plt.scatter(x, y, color='blue', marker='o', s=3)
                # plt.text(x + 1, y + 1, f'{val:.2f}', color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='blue', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    plt.tight_layout()
    plt.savefig(config.plot_save_path) # Save the plot to a file
    plt.show()

    print(f"\nModel training and conditional sampling complete. Check the generated plots (saved to {config.plot_save_path}).")
    print(f"Model checkpoint saved to {config.model_checkpoint_path} if enabled in config.")
    print("The red circles represent temperature observation points.")
    print("The blue X's represent salinity observation points.")
    if not config.use_salinity:
        print("Salinity plots were skipped as 'use_salinity' is set to False in config.")
    print("Land regions (where the mask is 0) are shown as masked areas in the plots, indicating no data.")
