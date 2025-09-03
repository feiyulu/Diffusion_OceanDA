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
from data_utils import load_static_data, load_test_ocean_slice
from unet_model import UNet
from diffusion_process import Diffusion
from sampling_utils import sample_conditional
from observation_utils import create_observation_tensors
from plotting_utils import plot_ensemble_results_3d

if __name__ == "__main__":
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
        all_checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)]
        all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        for checkpoint_path in all_checkpoints:
            try:
                if checkpoint_path.endswith('.pth'):
                    checkpoint_data = torch.load(checkpoint_path, map_location=config.device)
                    state_dict = checkpoint_data['model_state_dict']
                    break
                elif os.path.isdir(checkpoint_path) and os.path.basename(checkpoint_path).startswith('epoch_'):
                    model_file = os.path.join(checkpoint_path, "pytorch_model.bin")
                    if os.path.exists(model_file):
                        state_dict = torch.load(model_file, map_location=config.device)
                        break
            except Exception as e:
                print(f"Warning: Could not load checkpoint {os.path.basename(checkpoint_path)}. Error: {e}")
                continue
    if state_dict is None: raise FileNotFoundError(f"No valid model checkpoint found in {checkpoint_dir}.")
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")


    z_ds=xr.open_dataset(config.filepath_z_static)
    z_da=z_ds['z']

    print("Loading static data for sampling...")
    land_mask, location_field, area_weights = load_static_data(config)
    land_mask = land_mask.to(config.device)
    if location_field is not None:
        location_field = location_field.to(config.device)

    for year in config.sample_years[:-1]:
        for sample_day in config.sample_days:
            sample_day_str = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(sample_day, unit='d')
            print(f"\n--- Processing Sample Day: {sample_day_str.strftime('%Y-%m-%d')} ---")

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


            # --- Generate observations using the flexible dispatcher ---
            observations, observed_mask, guidance_strength = create_observation_tensors(
                config, sample_day_str, true_sample, land_mask
            )
            num_obs_points = torch.sum(observed_mask).item()

            print(f"Generating ensemble of size {config.ensemble_size}...")
            ensemble_members = []
            batch_size = config.sampling_batch_size
            num_generated = 0
            
            with tqdm(total=config.ensemble_size, desc="Generating ensemble members") as pbar:
                while num_generated < config.ensemble_size:
                    current_batch_size = min(batch_size, config.ensemble_size - num_generated)
                    
                    generated_batch = sample_conditional(
                        model, diffusion, config,
                        observations=observations.to(device), 
                        observed_mask=observed_mask.to(device),
                        guidance_strength_mask=guidance_strength.to(device),
                        land_mask=land_mask,
                        target_conditions=true_conditions,
                        target_location_field=target_location_field,
                        num_samples=current_batch_size
                    )
                    
                    for i in range(generated_batch.shape[0]):
                        ensemble_members.append(generated_batch[i].cpu())
                    
                    num_generated += current_batch_size
                    pbar.update(current_batch_size)

            if ensemble_members:
                ensemble_tensor = torch.stack(ensemble_members)
                ensemble_mean = torch.mean(ensemble_tensor, dim=0)
                ensemble_spread = torch.std(ensemble_tensor, dim=0)

                area_weights_np = area_weights.squeeze(0).cpu().numpy()
                
                observed_mask_np = observed_mask.squeeze(0).cpu().numpy()

                for depth_idx in config.plot_depth_levels:
                    plot_ensemble_results_3d(
                        ensemble_mean, ensemble_spread, true_sample.squeeze(0).cpu(), clim_pred, 
                        observed_mask_np,
                        land_mask[0, 0, depth_idx].cpu().numpy(),
                        area_weights_np, config, sample_day_str, num_obs_points,
                        depth_level=depth_idx, depth=z_da[depth_idx].values
                    )
            else:
                print("Ensemble generation failed.")

    print("\nEnsemble sampling and analysis complete.")