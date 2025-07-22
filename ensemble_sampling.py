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
from plotting_utils import plot_ensemble_results_3d_surface

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

    # FIX: Initialize model and diffusion process correctly
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
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model from epoch {checkpoint.get('epoch', 'N/A')} loaded successfully.")

    # --- 3. Loop Through Sample Days and Generate Ensembles ---
    for year in range(config.sample_years[0], config.sample_years[1] + 1):
        for day_of_year in config.sample_days[0]:
            sample_day_str = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(day_of_year, unit='d')
            print(f"\n--- Processing Sample Day: {sample_day_str.strftime('%Y-%m-%d')} ---")

            # --- 3a. Prepare Data and Observations ---
            true_sample, land_mask, true_conditions, true_location_field, _, _ = load_ocean_data(
                config, time_range=[sample_day_str.strftime('%Y-%m-%d')], is_test_data=True
            )
            true_sample = true_sample.to(config.device)
            land_mask = land_mask.to(config.device)
            target_location_field = true_location_field.to(config.device)

            # Get climatological prediction as a baseline
            # (Climatology logic can be complex, this is a simplified placeholder)
            clim_pred = torch.zeros_like(true_sample.squeeze(0)) # Placeholder

            if config.use_real_observations:
                observations, observed_mask, obs_points_actual = map_real_obs_to_grid_3d(
                    config, sample_day_str, true_sample.shape
                )
                num_obs_points = len(obs_points_actual)
            else:
                # Generate synthetic observations from the 3D ground truth
                print("Generating synthetic observations by sampling from 3D ground truth...")
                num_obs_points = config.observation_samples[0]
                observations = torch.zeros_like(true_sample)
                observed_mask = torch.zeros_like(true_sample, dtype=torch.bool)
                
                # Find valid ocean coordinates in 3D
                ocean_coords_d, ocean_coords_h, ocean_coords_w = torch.where(land_mask[0, 0] == 1)
                all_ocean_points = list(zip(ocean_coords_d.cpu().numpy(), ocean_coords_h.cpu().numpy(), ocean_coords_w.cpu().numpy()))

                if not all_ocean_points:
                    raise ValueError("Land mask is all land. Cannot generate synthetic observations.")

                num_obs_to_sample = min(len(all_ocean_points), num_obs_points)
                sampled_indices = np.random.choice(len(all_ocean_points), num_obs_to_sample, replace=False)
                
                obs_points_actual = []
                for idx in sampled_indices:
                    z, y, x = all_ocean_points[idx]
                    for c in range(config.channels):
                        val = true_sample[0, c, z, y, x].item()
                        observations[0, c, z, y, x] = val
                        observed_mask[0, c, z, y, x] = True
                        obs_points_actual.append((c, z, y, x, val))
                print(f"Generated {len(obs_points_actual)} synthetic 3D observation points.")

            # --- 3b. Generate Ensemble ---
            ensemble_members = []
            print(f"Generating ensemble of size {config.ensemble_size} with {num_obs_points} observations...")
            
            # FIX: Call the refactored sample_conditional function with correct arguments
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

                # FIX: Use a plotting function designed for the surface of 3D data
                plot_ensemble_results_3d_surface(
                    ensemble_mean, ensemble_spread, true_sample.squeeze(0), clim_pred, 
                    obs_points_actual, land_mask.squeeze(0).squeeze(0).cpu().numpy(), 
                    config, sample_day_str, num_obs_points
                )
            else:
                print("Ensemble generation failed.")

    print("\nEnsemble sampling and analysis complete.")

