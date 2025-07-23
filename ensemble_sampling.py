# --- ensemble_sampling.py ---
# This script generates an ensemble of samples from a trained 3D model.
import torch
import numpy as np
import xarray as xr
import os
import argparse
import pandas as pd
from tqdm import tqdm

# Import components from other project files
from config import Config
from data_utils import load_ocean_data
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

    print(f"Using device: {config.device}")
    print(f"Ensemble size: {config.ensemble_size}")

    model = UNet(config).to(config.device)
    diffusion = Diffusion(
        timesteps=config.timesteps, beta_start=config.beta_start, 
        beta_end=config.beta_end, device=config.device
    )

    # --- 2. Load Pre-trained Model ---
    latest_checkpoint_path = None
    if os.path.exists(config.model_checkpoint_dir):
        checkpoints = [f for f in os.listdir(config.model_checkpoint_dir) if f.startswith(f"ODA_ch{config.channels}_{config.test_id}") and f.endswith(".pth")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
            latest_checkpoint_path = os.path.join(config.model_checkpoint_dir, checkpoints[-1])
    
    if not latest_checkpoint_path:
        raise FileNotFoundError(f"No model checkpoint found in {config.model_checkpoint_dir}.")
    
    print(f"Loading model from {latest_checkpoint_path} for sampling...")
    checkpoint = torch.load(latest_checkpoint_path, map_location=config.device)
    # Handle DataParallel wrapper if present
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"Model from epoch {checkpoint.get('epoch', 'N/A')} loaded successfully.")

    # --- 3. Loop Through Sample Days and Generate Ensembles ---
    for sample_day in config.sample_days:
        sample_day_str = pd.to_datetime(f"{config.sample_years[0]}-01-01") + pd.to_timedelta(sample_day, unit='d')
        print(f"\n--- Processing Sample Day: {sample_day_str.strftime('%Y-%m-%d')} ---")

        # --- 3a. Prepare Data and Observations ---
        true_sample, land_mask, true_conditions, true_location_field, _, _ = load_ocean_data(
            config, time_range=[sample_day], is_test_data=True
        )
        true_sample = true_sample.to(config.device)
        land_mask = land_mask.to(config.device)
        target_location_field = true_location_field.to(config.device)

        clim_pred = torch.zeros_like(true_sample.squeeze(0))

        if config.use_real_observations:
            observations, observed_mask, obs_points_actual = map_real_obs_to_grid_3d(
                config, sample_day_str, true_sample.shape
            )
            num_obs_points = len(obs_points_actual)
        else:
            # --- NEW: Sample full vertical columns at random 2D locations ---
            print("Generating synthetic observations by sampling vertical columns...")
            num_profiles = config.observation_samples[0]
            observations = torch.zeros_like(true_sample)
            observed_mask = torch.zeros_like(true_sample, dtype=torch.bool)
            
            # Find valid 2D ocean locations from the surface mask
            surface_mask = land_mask[0, 0, 0]
            ocean_coords_h, ocean_coords_w = torch.where(surface_mask == 1)
            all_ocean_locations_2d = list(zip(ocean_coords_h.cpu().numpy(), ocean_coords_w.cpu().numpy()))

            if not all_ocean_locations_2d:
                raise ValueError("Land mask is all land. Cannot generate synthetic observations.")

            num_profiles_to_sample = min(len(all_ocean_locations_2d), num_profiles)
            sampled_indices = np.random.choice(len(all_ocean_locations_2d), num_profiles_to_sample, replace=False)
            
            obs_points_actual = []
            for idx in sampled_indices:
                y, x = all_ocean_locations_2d[idx]
                # For each chosen (y, x), sample the entire water column
                for z in range(config.data_shape[0]):
                    # Check if this specific 3D point is in the ocean
                    if land_mask[0, 0, z, y, x] == 1:
                        for c in range(config.channels):
                            val = true_sample[0, c, z, y, x].item()
                            observations[0, c, z, y, x] = val
                            observed_mask[0, c, z, y, x] = True
                            obs_points_actual.append((c, z, y, x, val))
            
            num_obs_points = len(obs_points_actual)
            print(f"Generated {num_obs_points} observation points from {num_profiles_to_sample} vertical profiles.")

        # --- 3b. Generate Ensemble ---
        print(f"Generating ensemble of size {config.ensemble_size} with {num_obs_points} observations...")
        generated_samples = sample_conditional(
            model, diffusion, config,
            observations=observations, 
            observed_mask=observed_mask,
            land_mask=land_mask,
            target_conditions=true_conditions,
            target_location_field=target_location_field
        )
        ensemble_members = [s for s in generated_samples]

        # --- 3c. Analyze and Visualize Ensemble ---
        if ensemble_members:
            ensemble_tensor = torch.stack(ensemble_members)
            ensemble_mean = torch.mean(ensemble_tensor, dim=0)
            ensemble_spread = torch.std(ensemble_tensor, dim=0)

            # Loop through specified depth levels and plot each one
            for depth_idx in config.plot_depth_levels:
                plot_ensemble_results_3d(
                    ensemble_mean, ensemble_spread, true_sample.squeeze(0), clim_pred, 
                    obs_points_actual, land_mask[0, 0, depth_idx].cpu().numpy(),
                    config, sample_day_str, num_obs_points,
                    depth_level=depth_idx
                )
        else:
            print("Ensemble generation failed.")

    print("\nEnsemble sampling and analysis complete.")
