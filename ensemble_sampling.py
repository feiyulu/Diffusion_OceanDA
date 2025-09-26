# --- ensemble_sampling.py ---
# This script generates an ensemble of samples from a trained 3D model.
import torch
import numpy as np
import xarray as xr
import os
import argparse
import pandas as pd
from tqdm import tqdm
import re
import pickle

from config import Config
from data_utils import load_static_data, load_test_ocean_slice, load_real_observations, load_2d_prior_slice
from unet_model import UNet
from diffusion_process import Diffusion
from sampling_utils import sample_conditional
from observation_utils import create_observation_tensors
from plotting_utils import plot_ensemble_results_3d, plot_loaded_observations

if __name__ == "__main__":
    # --- 1. Setup and Configuration ---
    parser = argparse.ArgumentParser(description="Ensemble sampling for Ocean Diffusion Model.")
    parser.add_argument("--work_path", "-w", type=str, default=".", help="Working directory.")
    parser.add_argument("--config", "-c", type=str, default="config.json", help="Path to the JSON config file.")
    args = parser.parse_args()

    config_path = os.path.join(args.work_path, args.config)
    try:
        config = Config.from_json_file(config_path)
    except FileNotFoundError as e:
        print(e); raise

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    config.device = device
    print(f"Using device: {config.device}")
    print(f"Ensemble size: {config.ensemble_size}")
    print(f"Sampling batch size: {config.sampling_batch_size}")

    model = UNet(config).to(config.device)
    diffusion = Diffusion(timesteps=config.timesteps, beta_start=config.beta_start, beta_end=config.beta_end, device=config.device)

    state_dict = None
    checkpoint_dir = config.model_checkpoint_dir
    if os.path.exists(checkpoint_dir):
        # Find the latest checkpoint directory or .pth file
        all_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('epoch_') or f.endswith('.pth')]
        if not all_checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        if latest_checkpoint.endswith('.pth'):
            checkpoint_data = torch.load(latest_checkpoint, map_location=config.device)
            state_dict = checkpoint_data.get('model_state_dict', checkpoint_data)
        elif os.path.isdir(latest_checkpoint):
            model_file = os.path.join(latest_checkpoint, "pytorch_model.bin")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location=config.device)

    if state_dict is None: raise FileNotFoundError(f"No valid model checkpoint found in {checkpoint_dir}.")
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")


    z_ds=xr.open_dataset(config.filepath_z_static)
    z_da=z_ds['z']

    # --- 2. Load Static Data ---
    print("Loading static data for sampling...")
    land_mask, location_field, area_weights = load_static_data(config)
    land_mask = land_mask.to(config.device)
    if location_field is not None:
        location_field = location_field.to(config.device)

    for year in config.sample_years[:-1]:
        # --- 3. Main Loop: Process each specified sample day ---
        for sample_day in config.sample_days:
            sample_day_datetime = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(sample_day, unit='d')
            print(f"\n--- Processing Sample Day: {sample_day_datetime.strftime('%Y-%m-%d')} ---")

            # Load the ground truth data for this day to use for observation generation and verification.
            true_sample, true_conditions = load_test_ocean_slice(config, year, sample_day)
            true_sample = true_sample.to(config.device)

            # --- NEW: Load previous states' surface data ---
            prev_state_surface = None
            if config.previous_states:
                print("Loading previous states for sequential input...")
                prev_state_surfaces = []
                for state_config in config.previous_states:
                    lag_days = state_config.get("lag_days", 1)
                    channels_to_use = state_config.get("channels", [0])
                    prev_day = sample_day - lag_days
                    if prev_day < 0:
                        print(f"Warning: Skipping previous state for lag {lag_days} as it falls before the start of the year.")
                        continue
                    
                    print(f"  - Loading state from {lag_days} day(s) ago (day {prev_day}), channels {channels_to_use}...")
                    prev_sample, _ = load_test_ocean_slice(config, year, prev_day)
                    # Extract surface layer (depth=0) for the specified channels.
                    # load_test_ocean_slice returns a shape of (1, C, D, H, W), so we index at batch 0.
                    surface_slice = prev_sample[0, channels_to_use, 0, :, :].to(config.device)
                    # Add a batch dimension back for concatenation.
                    surface_slice = surface_slice.unsqueeze(0)
                    prev_state_surfaces.append(surface_slice)
                
                if prev_state_surfaces:
                    prev_state_surface = torch.cat(prev_state_surfaces, dim=1) # Concatenate along the channel dimension
                    print(f"Previous state surface shape: {prev_state_surface.shape}")

            # --- NEW: Load 2D prior fields ---
            prior_2d_data = None
            if config.prior_2d_fields:
                print("Loading 2D prior fields...")
                prior_2d_list = []
                for field_config in config.prior_2d_fields:
                    lag_days = field_config.get("lag_days", 1)
                    prev_day = sample_day - lag_days
                    prev_time_coord = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(prev_day, unit='d')
                    prior_slice = load_2d_prior_slice(config, prev_time_coord)
                    if prior_slice is not None:
                        prior_2d_list.append(prior_slice.unsqueeze(0)) # Add batch dim
                if prior_2d_list:
                    prior_2d_data = torch.cat(prior_2d_list, dim=1).to(config.device)

            target_location_field = location_field

            clim_ds = xr.open_dataset(config.filepath_t_clim, decode_times=False)
            if not ( config.varname_lat=='lat' and config.varname_lon=='lon'):
                clim_ds = clim_ds.rename({config.varname_lat:'lat', config.varname_lon:'lon'})
            clim_da = clim_ds[config.varname_t].isel(
                lat=slice(config.lat_range[0], config.lat_range[1]), 
                lon=slice(config.lon_range[0], config.lon_range[1]))
            clim_pred = (clim_da - config.T_range[0]) / (config.T_range[1]-config.T_range[0])


            # --- 4. Observation Handling ---
            # Based on the config, load real observations or generate synthetic ones.
            use_real_obs = any(s.get("enabled", False) and s.get("type", "").startswith("real") for s in config.observation_sources)

            if use_real_obs:
                print("Loading real observations...")
                observations, observed_mask, guidance_strength = load_real_observations(
                    config, sample_day_datetime
                )
                print(f"Loaded {torch.sum(observed_mask).item()} real observation points.")

            else:
                print("Generating synthetic observations...")
                observations, observed_mask, guidance_strength = create_observation_tensors(
                    config, sample_day_datetime, true_sample, land_mask
                )

            # Calculate the number of observations that will be used for guidance, for logging and plotting.
            # --- Calculate the number of observation points that will actually be used ---
            obs_counts = {}
            total_used_obs_points = 0
            for source in config.observation_sources:
                if not source.get("enabled", False):
                    continue
                
                # Create a mask specific to this source's strength to count its points
                source_strength = source.get("guidance_strength", 1.0)
                source_mask = (observed_mask.bool()) & (guidance_strength == source_strength)

                source_type = source.get("type", "")
                if "profiles" in source_type or "argo" in source_type:
                    # For profile data, count the number of unique horizontal locations (profiles).
                    # Project the 3D mask to a 2D horizontal plane.
                    horizontal_profile_mask = torch.any(source_mask, dim=2).squeeze(0).squeeze(0)
                    num_profiles = torch.sum(horizontal_profile_mask).item()
                    obs_counts[source['name']] = num_profiles
                    total_used_obs_points += num_profiles # Add profile count for consistency in total
                else:
                    # For gridded data (like SST), the number of points in the mask is the final count,
                    # as subsampling was already handled during creation.
                    used_points = torch.sum(source_mask).item()
                    obs_counts[source['name']] = used_points
                    total_used_obs_points += used_points
            
            obs_count_str = "_".join([f"{name}{count}" for name, count in obs_counts.items()])
            print(f"Total observation points to be used for guidance: {total_used_obs_points} ({obs_count_str})")

            # --- 5. Ensemble Generation ---
            print(f"\nGenerating ensemble of size {config.ensemble_size}...")
            ensemble_members = []
            batch_size = config.sampling_batch_size
            num_generated = 0

            while num_generated < config.ensemble_size:
                current_batch_size = min(batch_size, config.ensemble_size - num_generated)
                
                generated_batch = sample_conditional(
                    model, diffusion, config,
                    observations=observations.to(device), 
                    observed_mask=observed_mask.to(device),
                    guidance_strength_mask=guidance_strength.to(device),
                    land_mask=land_mask,
                    target_conditions=true_conditions, target_location_field=target_location_field,
                    prev_state_surface=prev_state_surface,
                    prior_2d_fields=prior_2d_data,
                    num_samples=current_batch_size)
                
                for i in range(generated_batch.shape[0]):
                    ensemble_members.append(generated_batch[i].cpu())
                num_generated += current_batch_size

            # --- 6. Analysis and Plotting ---
            if ensemble_members:
                ensemble_tensor = torch.stack(ensemble_members)
                ensemble_mean = torch.mean(ensemble_tensor, dim=0)
                ensemble_spread = torch.std(ensemble_tensor, dim=0)

                area_weights_np = area_weights.squeeze(0).cpu().numpy()
                
                observed_mask_np = observed_mask.squeeze(0).cpu().numpy()

                for depth_idx in config.plot_depth_levels:
                    plot_ensemble_results_3d(
                        ensemble_mean, ensemble_spread, true_sample.squeeze(0).cpu(), clim_pred, 
                        observed_mask_np, land_mask[0, 0, depth_idx].cpu().numpy(), 
                        area_weights_np, 
                        config, sample_day_datetime, total_used_obs_points, obs_count_str,
                        depth_level=depth_idx, depth=z_da[depth_idx].values
                    )
            else:
                print("Ensemble generation failed.")

    print("\nEnsemble sampling and analysis complete.")