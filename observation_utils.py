# --- observation_utils.py ---
# This file contains a flexible framework for processing and assimilating
# multiple types of observations into the model's grid.

import torch
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree

# --- Main Dispatcher ---
def create_observation_tensors(config, sample_day_datetime, true_sample, land_mask):
    """
    Acts as a dispatcher to generate observation tensors from various synthetic sources.
    Main dispatcher for creating observation tensors from multiple sources.
    Reads the `observation_sources` from the config and calls the appropriate handler for each.
    """
    _, C, D, H, W = true_sample.shape
    
    # Initialize combined tensors
    combined_observations = torch.zeros_like(true_sample)
    combined_mask = torch.zeros_like(true_sample, dtype=torch.bool)
    combined_guidance_strength = torch.zeros_like(true_sample)
    
    total_obs_points = 0

    # Loop through all observation sources defined in the configuration.
    for source in config.observation_sources:
        if not source.get("enabled", False):
            continue

        print(f"--- Processing observation source: {source['name']} ---")
        obs_type = source['type'].lower()
        
        obs_values, obs_mask = None, None

        # Call the appropriate handler based on the source type.
        if obs_type == "synthetic_profiles":
            obs_values, obs_mask = process_synthetic_profiles(source, true_sample, land_mask)
        elif obs_type == "synthetic_surface":
            obs_values, obs_mask = process_synthetic_surface(source, true_sample, land_mask)
        # Future real data handlers would go here
        # elif obs_type == "real_argo":
        #     obs_values, obs_mask = process_real_argo(source, config, sample_day_datetime)
        else:
            print(f"Warning: Unknown observation type '{source['type']}'. Skipping.")
            continue
            
        if obs_values is not None:
            # Add the processed observations to the combined tensors
            valid_mask = obs_mask.bool()
            combined_observations[valid_mask] = obs_values[valid_mask]
            combined_mask[valid_mask] = True
            combined_guidance_strength[valid_mask] = source.get("guidance_strength", 1.0)
            
            # --- Improved Diagnostic Logging ---
            total_points_in_source = torch.sum(valid_mask).item()
            total_obs_points += total_points_in_source

            if "profiles" in obs_type:
                # For profiles, count the number of unique horizontal locations
                horizontal_mask = torch.any(valid_mask, dim=2).squeeze(0).squeeze(0)
                num_profiles = torch.sum(horizontal_mask).item()
                print(f"Added {num_profiles} profiles ({total_points_in_source} total points) from {source['name']}.")
            else:
                # For other types (like SST), just report the points
                print(f"Added {total_points_in_source} observation points from {source['name']}.")

    print(f"\nTotal observation points created: {total_obs_points}")
    return combined_observations, combined_mask, combined_guidance_strength


# --- Observation Type Handlers ---

def process_synthetic_profiles(source_config, true_sample, land_mask):
    """
    Creates synthetic Argo-like profiles by selecting random ocean columns from the ground truth data.
    Generates sparse vertical profile observations by sampling columns from the ground truth.
    """
    print(f"Generating {source_config['num_profiles']} synthetic profiles...")
    _, C, D, H, W = true_sample.shape
    observations = torch.zeros_like(true_sample)
    observed_mask = torch.zeros_like(true_sample, dtype=torch.bool)

    # Find all possible ocean surface locations to sample from
    surface_mask = land_mask[0, 0, 0].cpu().numpy()
    ocean_coords_y, ocean_coords_x = np.where(surface_mask == 1)
    
    if len(ocean_coords_y) == 0:
        print("Warning: No ocean points found in mask. Cannot generate profiles.")
        return None, None

    num_profiles_to_sample = min(len(ocean_coords_y), source_config['num_profiles'])
    sampled_indices = np.random.choice(len(ocean_coords_y), num_profiles_to_sample, replace=False)

    for idx in sampled_indices:
        y, x = ocean_coords_y[idx], ocean_coords_x[idx]
        # Copy the entire vertical column from the true sample
        observations[0, :, :, y, x] = true_sample[0, :, :, y, x]
        observed_mask[0, :, :, y, x] = True

    return observations, observed_mask


def process_synthetic_surface(source_config, true_sample, land_mask):
    """
    Creates synthetic sea-surface observations (like SST) from the top layer of the ground truth.
    It can also subsample the data to simulate sparse observations.
    Generates dense surface observations from the ground truth.
    """
    target_channel = source_config.get("target_channel", 0)
    print(f"Generating synthetic surface data for channel {target_channel}...")
    
    observations = torch.zeros_like(true_sample)
    observed_mask = torch.zeros_like(true_sample, dtype=torch.bool)
    
    # Get the surface layer (z=0) from the true sample for the specified channel
    surface_data = true_sample[0, target_channel, 0, :, :].clone()
    
    # Apply the land mask
    surface_mask = land_mask[0, 0, 0, :, :].bool().cpu().numpy()
    valid_y, valid_x = np.where(surface_mask)
    
    # --- Handle subsampling ---
    subsample_frac = source_config.get("subsample_fraction")
    if subsample_frac is not None and subsample_frac < 1.0:
        num_to_sample = int(len(valid_y) * subsample_frac)
        sampled_indices = np.random.choice(len(valid_y), num_to_sample, replace=False)
        final_y = valid_y[sampled_indices]
        final_x = valid_x[sampled_indices]
    else:
        final_y, final_x = valid_y, valid_x

    # Create a new mask with only the subsampled points
    final_mask = torch.zeros_like(surface_data, dtype=torch.bool)
    final_mask[final_y, final_x] = True
    
    observations[0, target_channel, 0, :, :][final_mask] = surface_data[final_mask]
    observed_mask[0, target_channel, 0, :, :][final_mask] = True

    return observations, observed_mask
