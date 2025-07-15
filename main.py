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
    try:
        config = Config.from_json_file(f"{args.work_path}/{args.config}")
        if os.path.realpath(config.output_dir) != os.path.realpath(args.work_path):
            raise Exception('Working directory does not match case name.')
    except FileNotFoundError as e:
        print(e)
        raise Exception("Please ensure 'config.json' exists")

    print(f"Using device: {config.device}")
    print(f"Model will use {config.channels} channel(s). Salinity included: {config.use_salinity}")
    print(f"Sampling method: {config.sampling_method.upper()}")
    print(f"Observation fidelity weight: {config.observation_fidelity_weight}")
    print(f"Generate training animation: {config.generate_training_animation}")
    print(f"Using day of year embedding: {config.use_dayofyear_embedding} (dim: {config.dayofyear_embedding_dim})")
    print(f"Using CO2 embedding: {config.use_co2_embedding} (dim: {config.co2_embedding_dim})")
    print(f"Using 2D location embedding: {config.use_2d_location_embedding} (channels: {config.location_embedding_channels})")
    print(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
    print(f"Checkpoint save interval: {config.save_interval} epochs")

    # 1. Prepare Data
    if config.real_data:
        data, land_mask, day_of_year_data, co2_data, location_field_data, T_range, S_range = load_ocean_data(
            config.filepath_t,
            config.image_size, config.channels,
            time_range=config.training_day_range,
            time_interval=config.training_day_interval,
            lat_range=config.lat_range,lon_range=config.lon_range,
            use_salinity=config.use_salinity,
            filepath_s=config.filepath_s,
            variable_names=[config.varname_t, config.varname_s],
            T_range=config.T_range,S_range=config.S_range,
            co2_filepath=config.co2_filepath
        )
    else:
        data, land_mask, day_of_year_data, location_field_data = generate_synthetic_ocean_data(
            config.data_points, config.image_size, config.channels, 
            config.use_salinity, config.channel_wise_normalization
        )

    print(f"Data size: {data.shape}")
    print(f"Land mask size: {land_mask.shape}")
    if config.use_dayofyear_embedding: print(f"Day of Year data size: {day_of_year_data.shape}")
    if config.use_co2_embedding: print(f"CO2 data size: {co2_data.shape}")
    if config.use_2d_location_embedding: print(f"2D Location Field data size: {location_field_data.shape}")

    # Create a custom dataset that yields the additional embeddings
    class CustomOceanDataset(torch.utils.data.Dataset):
        def __init__(self, data, dayofyear, location_field): 
            self.data = data
            self.dayofyear = dayofyear
            self.co2 = co2
            self.location_field = location_field

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # Returns data, day_of_year, and location_field for the given index
            return self.data[idx], self.dayofyear[idx], self.co2[idx], self.location_field.squeeze()

    # Initialize CustomOceanDataset with the loaded data
    full_dataset = CustomOceanDataset(data, day_of_year_data, co2_data, location_field_data) 

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
        use_dayofyear_embedding=config.use_dayofyear_embedding,
        dayofyear_embedding_dim=config.dayofyear_embedding_dim,
        use_2d_location_embedding=config.use_2d_location_embedding,
        location_embedding_channels=config.location_embedding_channels
        ).to(config.device) # Instantiate the U-Net model and move to device

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
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

    if config.load_model_for_sampling:
        if latest_checkpoint_path and os.path.exists(latest_checkpoint_path):
            print(f"Loading model from {latest_checkpoint_path} for sampling...")
            checkpoint = torch.load(latest_checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Optimizer state is not needed for sampling, but loaded for consistency
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
            start_epoch = checkpoint['epoch'] # The epoch when the model was saved
            print(f"Model loaded. Sampling will use model from epoch {start_epoch}.")
        else:
            print(f"No checkpoint found at {config.model_checkpoint_dir}. Cannot load model for sampling.")
            # If no model to load for sampling, you might want to exit or proceed with untrained model
            # For now, we'll proceed, but it might lead to poor sampling results.

    elif latest_checkpoint_path and os.path.exists(latest_checkpoint_path): # Resume training
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
        train_losses, val_losses = train_diffusion_model(
            model, train_loader, val_loader, diffusion, optimizer, 
            config.epochs, config.device, land_mask, 
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            start_epoch=start_epoch, 
            use_dayofyear_embedding=config.use_dayofyear_embedding,
            use_2d_location_embedding=config.use_2d_location_embedding,
            save_interval=config.save_interval, 
            checkpoint_dir=config.model_checkpoint_dir, 
            test_id=config.test_id,
            channels=config.channels,
            loss_plot_dir=config.loss_plot_dir
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
    else:
        print("Skipping training as 'load_model_for_sampling' is True and a model was loaded.")

    # Plot training and validation losses after training
    if train_losses and val_losses: # Only plot if training actually occurred and losses were collected
        plot_filename = f"loss_plot_{test_id}_epoch_{config.epochs}.png"
        specific_plot_path = os.path.join(config.plot_save_dir, plot_filename)
        plot_losses(train_losses, val_losses, specific_plot_path)

    # 4. Prepare Sparse Observations for Conditional Sampling
    img_h, img_w = config.image_size
    sample_shape = (1, config.channels, img_h, img_w)
    
    # Load new "true" samples to serve as the ground truth for observations
    for sample_day in config.sample_days:
        if config.real_data:
            true_sample, _, true_day_of_year_single, true_location_field_single, _, _ = load_ocean_data(
                config.filepath_t_test,
                config.image_size, config.channels,
                time_range=[sample_day,sample_day+1,1],
                lat_range=config.lat_range,lon_range=config.lon_range,
                use_salinity=config.use_salinity,
                filepath_s=config.filepath_s_test,
                variable_names=[config.varname_t, config.varname_s],
                T_range=config.T_range,S_range=config.S_range
            )
        else:
            true_sample, _, true_day_of_year_single, true_location_field_single = generate_synthetic_ocean_data(
                num_samples=1,
                image_size=config.image_size,
                channels=config.channels,
                use_salinity=config.use_salinity,
                channel_wise_normalization=config.channel_wise_normalization
            )
        true_sample = true_sample.to(config.device)

        # Extract scalar day of year and the 2D location field
        target_doy = true_day_of_year_single.item() if true_day_of_year_single.numel() > 0 else None
        target_location_field_for_sampling = true_location_field_single[0:1, :, :, :].to(config.device)

        # Initialize observation tensors based on the shape of the true_sample
        observations_for_sampling = torch.zeros_like(true_sample)
        observed_mask_for_sampling = torch.zeros_like(true_sample, dtype=torch.bool)

        # Randomly select a few observation points from the true_sample
        for num_obs_points in config.observation_samples:
            ocean_coords_h, ocean_coords_w = np.where(land_mask[0, 0].cpu().numpy() == 1)
            all_ocean_points = list(zip(ocean_coords_h, ocean_coords_w))

            if len(all_ocean_points) < num_obs_points:
                print(f"Warning: Not enough ocean points ({len(all_ocean_points)}) to sample {num_obs_points} observations. Sampling all available ocean points.")
                sampled_indices = np.arange(len(all_ocean_points))
            else:
                sampled_indices = np.random.choice(len(all_ocean_points), num_obs_points, replace=False)
            
            obs_points_actual = []
            for idx in sampled_indices:
                y, x = all_ocean_points[idx]
                for c in range(config.channels):
                    val = true_sample[0, c, y, x].item()
                    observations_for_sampling[0, c, y, x] = val
                    observed_mask_for_sampling[0, c, y, x] = True
                    obs_points_actual.append((c, y, x, val))

            # 5. Perform Conditional Sampling
            generated_samples = sample_conditional(
                model,
                diffusion,
                num_samples=1,
                observations=observations_for_sampling,
                observed_mask=observed_mask_for_sampling,
                land_mask=land_mask,
                channels=config.channels,
                image_size=config.image_size,
                device=config.device,
                sampling_method=config.sampling_method,
                observation_fidelity_weight=config.observation_fidelity_weight,
                target_dayofyear=target_doy,
                target_location_field=target_location_field_for_sampling,
                config=config # Pass the full config object
            )

            # Get climatological prediction as baseline skills
            ds_t_training = xr.open_mfdataset(config.filepath_t)
            da_t_training = ds_t_training[config.varname_t]
            clim_t = da_t_training.groupby('time.dayofyear').mean('time')
            clim_t_pred = (clim_t.isel(
                lat=slice(config.lat_range[0],config.lat_range[1]),
                lon=slice(config.lon_range[0],config.lon_range[1]),
                dayofyear=sample_day).values-config.T_range[0])/\
                    (config.T_range[1]-config.T_range[0])
            print(f'Climatological Temperature Shape: {clim_t_pred.shape}')
            
            if config.use_salinity: 
                ds_s_training = xr.open_mfdataset(config.filepath_s)
                da_s_training = ds_s_training[config.varname_s]
                clim_s = da_s_training.groupby('time.dayofyear').mean('time')
                clim_s_pred = (clim_s.isel(
                    lat=slice(config.lat_range[0],config.lat_range[1]),
                    lon=slice(config.lon_range[0],config.lon_range[1]),
                    dayofyear=sample_day).values-config.S_range[0])/\
                        (config.S_range[1]-config.S_range[0])

            # 6. Visualize and Save Results
            print("\nVisualizing and saving results...")
            generated_sample_np = generated_samples[0].cpu().numpy()
            true_sample_np = true_sample[0].cpu().numpy() # Convert the true sample to numpy
            land_mask_np = land_mask[0, 0].cpu().numpy()

            plt.figure(figsize=(config.image_size[1]/15, config.image_size[0]/10))

            # Plot True Temperature Map
            plt.subplot(3, 2, 1) 
            masked_true_temp = np.ma.masked_where(land_mask_np == 0, true_sample_np[0])
            std = np.nanstd(masked_true_temp)
            plt.imshow(masked_true_temp, cmap='viridis', origin='lower', vmin=0., vmax=1.)
            plt.colorbar(label='Normalized Temperature')
            plt.title(f'True Temperature Map (std: {std:.3f})')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')

            plt.subplot(3, 2, 2) 
            # Overlay sampled observation points for temperature on true map
            for c, y, x, val in obs_points_actual:
                if c == 0 and land_mask_np[y, x] == 1: # Only plot if it was on ocean
                    plt.scatter(x, y, c=val, cmap='viridis', vmin=0., vmax=1., marker='o', s=5)
                    # plt.text(x + 1, y + 1, f'{val:.2f}', color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
            plt.colorbar(label='Normalized Temperature')
            plt.title('True Temperature Samples')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')

            # Plot Climatology Temperature Map
            plt.subplot(3, 2, 3) # Position below true temp map
            std = np.nanstd(clim_t_pred)
            true_flat=masked_true_temp.flatten()
            clim_flat=clim_t_pred.flatten()
            true_flat_ma=true_flat[~np.isnan(true_flat) & ~np.isnan(clim_flat)]
            clim_flat_ma=clim_flat[~np.isnan(true_flat) & ~np.isnan(clim_flat)]
            corr = np.corrcoef(clim_flat_ma,true_flat_ma)[0,1]
            print(f'Climatological Temperature STD: {std}')
            print(f'Climatological Temperature Correlation: {corr}')
            plt.imshow(clim_t_pred, cmap='viridis', origin='lower', vmin=0., vmax=1.)
            plt.colorbar(label='Normalized Temperature')
            plt.title(f'Climatological Temperature Map (std:{std:.3f}, corr:{corr:.3f})')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')

            plt.subplot(3, 2, 4) # Position below true temp map
            rmse = np.sqrt(np.mean((clim_t_pred-masked_true_temp)**2))
            mae = np.mean(np.abs(clim_t_pred-masked_true_temp))
            print(f'Climatological Temperature RMSE: {rmse}')
            print(f'Climatological Temperature MAE: {mae}')
            plt.imshow(clim_t_pred-masked_true_temp, cmap='bwr', origin='lower', vmin=-0.1, vmax=0.1)
            plt.colorbar(label='Normalized Temperature')
            plt.title(f'Climatological Temperature Map (rmse:{rmse:.3f}, mae:{mae:.3f})')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')

            # Plot Generated Temperature Map
            plt.subplot(3, 2, 5) # Position below true temp map
            masked_generated_temp = np.ma.masked_where(land_mask_np == 0, generated_sample_np[0])
            std = np.nanstd(masked_generated_temp)
            generated_flat=masked_generated_temp.flatten()
            true_flat_ma=true_flat[~np.isnan(true_flat) & ~np.isnan(generated_flat)]
            generated_flat_ma=generated_flat[~np.isnan(true_flat) & ~np.isnan(generated_flat)]
            corr = np.corrcoef(generated_flat_ma,true_flat_ma)[0,1]
            print(f'Generated Temperature STD: {std}')
            print(f'Generated Temperature Correlation: {corr}')
            plt.imshow(masked_generated_temp, cmap='viridis', origin='lower', vmin=0., vmax=1.)
            plt.colorbar(label='Normalized Temperature')
            plt.title(f'Generated Temperature Map (std:{std:.3f}, corr:{corr:.3f})')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')

            plt.subplot(3, 2, 6) # Position below true temp map
            rmse = np.sqrt(np.mean((masked_generated_temp-masked_true_temp)**2))
            mae = np.mean(np.abs(masked_generated_temp-masked_true_temp))
            print(f'Generated Temperature RMSE: {rmse}')
            print(f'Generated Temperature MAE: {mae}')
            plt.imshow(masked_generated_temp-masked_true_temp, cmap='bwr', origin='lower', vmin=-0.1, vmax=0.1)
            plt.colorbar(label='Normalized Temperature')
            plt.title(f'Generated Temperature Map (rmse:{rmse:.3f}, mae:{mae:.3f})')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')
            # Overlay sampled observation points for temperature on generated map
            for c, y, x, val in obs_points_actual:
                if c == 0 and land_mask_np[y, x] == 1:
                    plt.scatter(x, y, color='black', marker='o', s=1)
            
            plt.tight_layout()
            plot_save_path_t = f"{config.sample_plot_dir}/ODA_obs{num_obs_points}_day{sample_day}_T.png"
            plt.savefig(plot_save_path_t) # Save the plot to a file
            plt.show()

            # Plot Salinity Maps if enabled
            if config.use_salinity:
                plt.figure(figsize=(16, 18))

                # Plot True Salinity Map
                plt.subplot(3, 2, 1) # Position next to true temp map
                masked_true_sal = np.ma.masked_where(land_mask_np == 0, true_sample_np[1])
                plt.imshow(masked_true_sal, cmap='plasma', origin='lower', vmin=0., vmax=1.)
                plt.colorbar(label='Normalized Salinity')
                plt.title('True Salinity Map (Ocean Only)')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')

                plt.subplot(3, 2, 2) # Position next to true temp map
                # Overlay sampled observation points for salinity on true map
                for c, y, x, val in obs_points_actual:
                    if c == 1 and land_mask_np[y, x] == 1:
                        plt.scatter(x, y, c=val, cmap='viridis', vmin=0., vmax=1., marker='o', s=5)
                        # plt.text(x + 1, y + 1, f'{val:.2f}', color='white', fontsize=8, ha='left', va='top', bbox=dict(facecolor='blue', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
                plt.colorbar(label='Normalized Salinity')
                plt.title('True Salinity Map (Ocean Only)')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')

                # Plot Climatology Salinity Map
                plt.subplot(3, 2, 3) # Position below true sal map
                rmse = np.sqrt(np.mean((clim_s_pred-masked_true_sal)**2))
                plt.imshow(clim_s_pred, cmap='plasma', origin='lower', vmin=0., vmax=1.)
                plt.colorbar(label='Normalized Salinity')
                plt.title(f'Generated Salinity Map (Ocean Only) RMSE:{rmse:.3f}')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')

                plt.subplot(3, 2, 4) # Position below true sal map
                mae = np.mean(np.abs(clim_s_pred-masked_true_sal))
                plt.imshow(clim_s_pred-masked_true_sal, cmap='bwr', origin='lower', vmin=-0.1, vmax=0.1)
                plt.colorbar(label='Normalized Salinity')
                plt.title(f'Generated Salinity Map (Ocean Only) MAE:{mae:.3f}')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')

                # Plot Generated Salinity Map
                plt.subplot(3, 2, 5) # Position below true sal map
                masked_generated_sal = np.ma.masked_where(land_mask_np == 0, generated_sample_np[1])
                rmse = np.sqrt(np.mean((masked_generated_sal-masked_true_sal)**2))
                plt.imshow(masked_generated_sal, cmap='plasma', origin='lower', vmin=0., vmax=1.)
                plt.colorbar(label='Normalized Salinity')
                plt.title(f'Generated Salinity Map (Ocean Only) RMSE:{rmse:.3f}')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')

                plt.subplot(3, 2, 6) # Position below true sal map
                mae = np.mean(np.abs(masked_generated_sal-masked_true_sal))
                plt.imshow(masked_generated_sal-masked_true_sal, cmap='bwr', origin='lower', vmin=-0.1, vmax=0.1)
                plt.colorbar(label='Normalized Salinity')
                plt.title(f'Generated Salinity Map (Ocean Only) MAE:{mae:.3f}')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')

                plt.tight_layout()
                plot_save_path_s = f"{config.sample_plot_dir}/ODA_obs{num_obs_points}_day{sample_day}_S.png"
                plt.savefig(config.plot_save_path_s) # Save the plot to a file
                plt.show()

    print(f"\nModel training and conditional sampling complete. Check the generated plots (saved to {config.sample_plot_dir}).")
    print(f"Model checkpoint saved to {config.model_checkpoint_dir} if enabled in config.")
    print("The red circles represent temperature observation points.")
    print("The blue X's represent salinity observation points.")
    if not config.use_salinity:
        print("Salinity plots were skipped as 'use_salinity' is set to False in config.")
    print("Land regions (where the mask is 0) are shown as masked areas in the plots, indicating no data.")
