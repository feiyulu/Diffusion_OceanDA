# --- observation_utils.py ---
# This file contains utility functions for processing and assimilating
# irregularly spaced observations into the model's 3D grid.

import torch
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree

def create_observation_tensors(config, sample_day_datetime, true_sample_shape):
    """
    Main dispatcher for creating observation tensors.
    It can generate synthetic observations or process real sparse data.
    """
    if config.use_real_observations:
        print(f"Using real observations with '{config.observation_operator}' operator.")
        if config.observation_operator == "nearest_neighbor":
            return map_real_obs_to_grid_3d(config, sample_day_datetime, true_sample_shape)
        else:
            raise ValueError(f"Unknown observation operator: {config.observation_operator}")
    else:
        print("Generating synthetic observations from ground truth.")
        # This part needs the ground truth data, so it's better handled in the main sampling script.
        # This function will now focus only on real observations.
        return None, None, None


def map_real_obs_to_grid_3d(config, sample_day_datetime, true_sample_shape):
    """
    Loads sparse observational data and maps it to the model's 3D grid.
    This function now works with 3D data, assuming observations are at the surface (z=0).
    """
    print("Loading and processing sparse observations for 3D grid...")
    
    # 1. Load the pre-processed observation file
    year = sample_day_datetime.year
    obs_path = config.observation_path_template.format(year=year)
    try:
        obs_ds = xr.open_dataset(obs_path)
    except FileNotFoundError:
        print(f"Warning: Observation file not found at {obs_path}. Returning empty observations.")
        return torch.zeros(true_sample_shape), torch.zeros(true_sample_shape, dtype=torch.bool), []

    # 2. Filter observations within the specified time window
    time_window = pd.Timedelta(days=config.observation_time_window_days)
    start_time = sample_day_datetime - time_window
    end_time = sample_day_datetime + time_window
    obs_in_window = obs_ds.sel(time=slice(start_time, end_time))
    
    if obs_in_window.sizes['profile'] == 0:
        print("Warning: No observations found within the time window. Returning empty observations.")
        return torch.zeros(true_sample_shape), torch.zeros(true_sample_shape, dtype=torch.bool), []
    
    # 3. Load the model's static grid to get geolat/geolon
    static_ds = xr.open_dataset(config.filepath_static)
    model_lats = static_ds['geolat'].isel(yh=slice(config.lat_range[0], config.lat_range[1]), xh=slice(config.lon_range[0], config.lon_range[1])).values
    model_lons = static_ds['geolon'].isel(yh=slice(config.lat_range[0], config.lat_range[1]), xh=slice(config.lon_range[0], config.lon_range[1])).values
    
    # 4. Create a KD-Tree for efficient nearest-neighbor search on the 2D horizontal grid
    grid_points_2d = np.vstack([model_lats.ravel(), model_lons.ravel()]).T
    kdtree = cKDTree(grid_points_2d)

    # 5. Initialize output tensors with the correct 5D shape
    # Shape: (N, C, D, H, W)
    observations_tensor = torch.zeros(true_sample_shape)
    observed_mask_tensor = torch.zeros(true_sample_shape, dtype=torch.bool)
    obs_points_for_plotting = []

    # 6. Map each observation to the nearest model grid cell at the surface (z=0)
    for i in range(len(obs_in_window.profile)):
        obs_lat = obs_in_window.lat[i].item()
        obs_lon = obs_in_window.lon[i].item()
        
        # Find nearest (y, x) grid index
        _, nearest_idx = kdtree.query([obs_lat, obs_lon])
        y, x = np.unravel_index(nearest_idx, model_lats.shape)
        
        # Assume all observations are for the surface layer (depth index z=0)
        z = 0

        # Process Temperature (channel 0)
        # We take the shallowest observation value
        temp_val = obs_in_window.T[i, 0].item() 
        if not np.isnan(temp_val):
            norm_temp = (temp_val - config.T_range[0]) / (config.T_range[1] - config.T_range[0])
            observations_tensor[0, 0, z, y, x] = norm_temp
            observed_mask_tensor[0, 0, z, y, x] = True
            # Store as (channel, z, y, x, value) for plotting
            obs_points_for_plotting.append((0, z, y, x, norm_temp))

        # Process Salinity (channel 1) if enabled
        if config.use_salinity:
            salt_val = obs_in_window.S[i, 0].item()
            if not np.isnan(salt_val):
                norm_salt = (salt_val - config.S_range[0]) / (config.S_range[1] - config.S_range[0])
                observations_tensor[0, 1, z, y, x] = norm_salt
                observed_mask_tensor[0, 1, z, y, x] = True
                obs_points_for_plotting.append((1, z, y, x, norm_salt))

    num_obs = len(obs_points_for_plotting)
    print(f"Successfully created observation tensor with {num_obs} points at the surface layer.")
    return observations_tensor, observed_mask_tensor, obs_points_for_plotting
