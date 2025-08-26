# --- ensemble_sampling.py ---
# This script generates an ensemble of samples from a trained 3D model.
# REFINED: Added batched sampling loop to conserve GPU memory while improving speed.
import torch
import numpy as np
import xarray as xr
import os
import argparse
import pandas as pd
from tqdm import tqdm
import re
import pickle

# Import components from other project files
from config import Config
from data_utils import load_static_data, load_test_ocean_slice
from unet_model import UNet
from diffusion_process import Diffusion
from sampling_utils import sample_conditional
from observation_utils import map_real_obs_to_grid_3d
from plotting_utils import plot_ensemble_results_3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble sampling for Ocean Diffusion Model.")
    parser.add_argument("--work_path", "-w", type=str, default=".", help="Working directory.")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to the JSON config file.")
    args = parser.parse_args()

    # --- 1. Initialization ---
    config_path = os.path.join(args.work_path, args.config)
    try:
        config = Config.from_json_file(config_path)
    except FileNotFoundError as e:
        print(e); raise

    # Set device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    config.device = device
    print(f"Using device: {config.device}")
    print(f"Ensemble size: {config.ensemble_size}")
    print(f"Sampling batch size: {config.sampling_batch_size}")

    model = UNet(config).to(config.device)
    diffusion = Diffusion(
        timesteps=config.timesteps, beta_start=config.beta_start, 
        beta_end=config.beta_end, device=config.device
    )

    # --- 2. Load Pre-trained Model (ROBUST LOGIC) ---
    state_dict = None
    checkpoint_dir = config.model_checkpoint_dir
    
    if os.path.exists(checkpoint_dir):
        # Find all potential checkpoints and sort them by modification time (newest first)
        all_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)]
        all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # Iterate through checkpoints and try to load them until one succeeds
        for checkpoint_path in all_checkpoints:
            try:
                if checkpoint_path.endswith('.pth'):
                    print(f"Attempting to load .pth checkpoint: {os.path.basename(checkpoint_path)}")
                    checkpoint_data = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
                    state_dict = checkpoint_data['model_state_dict']
                    print("Successfully loaded model state from .pth file.")
                    break # Exit loop on success
                elif os.path.isdir(checkpoint_path) and os.path.basename(checkpoint_path).startswith('epoch_'):
                    print(f"Attempting to load Accelerate checkpoint: {os.path.basename(checkpoint_path)}")
                    model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
                    if not os.path.exists(model_file):
                         model_file = os.path.join(checkpoint_path, "model.safetensors")
                    
                    if os.path.exists(model_file):
                        state_dict = torch.load(model_file, map_location=config.device, weights_only=False)
                        print("Successfully loaded model state from Accelerate checkpoint.")
                        break # Exit loop on success
            except (pickle.UnpicklingError, RuntimeError, EOFError) as e:
                print(f"Warning: Could not load checkpoint {os.path.basename(checkpoint_path)}. It may be corrupted. Trying next...")
                continue # Go to the next checkpoint
    
    if state_dict is None:
        raise FileNotFoundError(f"No valid, uncorrupted model checkpoint found in {checkpoint_dir}.")
    
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")


    z_ds=xr.open_dataset(config.filepath_z_static)
    z_da=z_ds['z']

    # --- 3. Prepare Static Data ---
    print("Loading static data for sampling...")
    land_mask, location_field = load_static_data(config)
    land_mask = land_mask.to(config.device)
    if location_field is not None:
        location_field = location_field.to(config.device)

    # --- 4. Loop Through Sample Days and Generate Ensembles ---
    for year in config.sample_years[:-1]:
        for sample_day in config.sample_days:
            sample_day_str = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(sample_day, unit='d')
            print(f"\n--- Processing Sample Day: {sample_day_str.strftime('%Y-%m-%d')} ---")

            # --- 4a. Prepare Data and Observations ---
            true_sample, true_conditions = load_test_ocean_slice(config, year, sample_day)
            true_sample = true_sample.to(config.device)
            target_location_field = location_field

            clim_ds = xr.open_dataset(config.filepath_t_clim)
            if not ( config.varname_lat=='lat' and config.varname_lon=='lon'):
                clim_ds = clim_ds.rename({config.varname_lat:'lat', config.varname_lon:'lon'})
            clim_da = clim_ds[config.varname_t].isel(
                lat=slice(config.lat_range[0], config.lat_range[1]), 
                lon=slice(config.lon_range[0], config.lon_range[1]))
            clim_pred = (clim_da - config.T_range[0]) / (config.T_range[1]-config.T_range[0])

            if config.use_real_observations:
                observations, observed_mask, obs_points_actual = map_real_obs_to_grid_3d(
                    config, sample_day_str, true_sample.shape
                )
                num_obs_points = len(obs_points_actual)
            else:
                print("Generating synthetic observations by sampling vertical columns...")
                num_profiles = config.observation_samples[0]
                observations = torch.zeros_like(true_sample)
                observed_mask = torch.zeros_like(true_sample, dtype=torch.bool)
                
                surface_mask = land_mask[0, 0, 0]
                ocean_coords_h, ocean_coords_w = torch.where(surface_mask.squeeze() == 1)
                all_ocean_locations_2d = list(zip(ocean_coords_h.cpu().numpy(), ocean_coords_w.cpu().numpy()))

                if not all_ocean_locations_2d:
                    raise ValueError("Land mask is all land. Cannot generate synthetic observations.")

                num_profiles_to_sample = min(len(all_ocean_locations_2d), num_profiles)
                sampled_indices = np.random.choice(len(all_ocean_locations_2d), num_profiles_to_sample, replace=False)
                
                obs_points_actual = []
                for idx in sampled_indices:
                    y, x = all_ocean_locations_2d[idx]
                    for z in range(config.data_shape[0]):
                        if land_mask[0, 0, z, y, x] == 1:
                            for c in range(config.channels):
                                val = true_sample[0, c, z, y, x].item()
                                observations[0, c, z, y, x] = val
                                observed_mask[0, c, z, y, x] = True
                                obs_points_actual.append((c, z, y, x, val))
                
                num_obs_points = len(obs_points_actual)
                print(f"Generated {num_obs_points} observation points from {num_profiles_to_sample} vertical profiles.")

            # --- 4b. Generate Ensemble in Batches ---
            print(f"Generating ensemble of size {config.ensemble_size} with {num_obs_points} observations...")
            ensemble_members = []
            batch_size = config.sampling_batch_size
            num_generated = 0
            
            with tqdm(total=config.ensemble_size, desc="Generating ensemble members") as pbar:
                while num_generated < config.ensemble_size:
                    current_batch_size = min(batch_size, config.ensemble_size - num_generated)
                    
                    # Generate one batch of samples at a time
                    generated_batch = sample_conditional(
                        model, diffusion, config,
                        observations=observations.to(device), 
                        observed_mask=observed_mask.to(device),
                        land_mask=land_mask,
                        target_conditions=true_conditions,
                        target_location_field=target_location_field,
                        num_samples=current_batch_size
                    )
                    
                    # Move samples to CPU and add to the list
                    for i in range(generated_batch.shape[0]):
                        ensemble_members.append(generated_batch[i].cpu())
                    
                    num_generated += current_batch_size
                    pbar.update(current_batch_size)


            # --- 4c. Analyze and Visualize Ensemble ---
            if ensemble_members:
                # Stack the collected CPU tensors for analysis
                ensemble_tensor = torch.stack(ensemble_members)
                ensemble_mean = torch.mean(ensemble_tensor, dim=0)
                ensemble_spread = torch.std(ensemble_tensor, dim=0)

                # Loop through specified depth levels and plot each one
                for depth_idx in config.plot_depth_levels:
                    plot_ensemble_results_3d(
                        ensemble_mean, ensemble_spread, true_sample.squeeze(0).cpu(), clim_pred, 
                        obs_points_actual, land_mask[0, 0, depth_idx].cpu().numpy(),
                        config, sample_day_str, num_obs_points,
                        depth_level=depth_idx, depth=z_da[depth_idx].values
                    )
            else:
                print("Ensemble generation failed.")

    print("\nEnsemble sampling and analysis complete.")
