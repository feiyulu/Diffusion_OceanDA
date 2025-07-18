# --- observation_utils.py ---
# This file contains utility functions for processing and assimilating
# irregularly spaced observations into the model's grid.

import torch
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree

def create_observation_tensors_from_sparse_data(config, sample_day_datetime, true_sample_shape):
    """
    Main dispatcher function for creating observation tensors from real sparse data.
    It reads the configuration and calls the appropriate observation operator.

    Args:
        config (Config): The main configuration object.
        sample_day_datetime (pd.Timestamp): The timestamp for the current sample day.
        true_sample_shape (tuple): The shape of the ground truth tensor.

    Returns:
        tuple: (observations_tensor, observed_mask_tensor, obs_points_for_plotting)
    """
    operator_type = config.observation_operator
    print(f"Using real observations with '{operator_type}' operator.")

    if operator_type == "nearest_neighbor":
        return map_obs_to_grid_nearest_neighbor(config, sample_day_datetime, true_sample_shape)
    else:
        raise ValueError(f"Unknown observation operator: {operator_type}")


def map_obs_to_grid_nearest_neighbor(config, sample_day_datetime, true_sample_shape):
    """
    Loads sparse observational data, finds the nearest model grid points using a KD-Tree,
    and creates observation and mask tensors by assigning each observation to a single grid cell.
    """
    print("Loading and processing sparse observations via nearest-neighbor mapping...")
    
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
    obs_in_window = obs_ds.where(
        (obs_ds.time >= start_time) & (obs_ds.time <= end_time), drop=True
    )
    if len(obs_in_window.profile) == 0:
        print("Warning: No observations found within the time window. Returning empty observations.")
        return torch.zeros(true_sample_shape), torch.zeros(true_sample_shape, dtype=torch.bool), []
    
    # 3. Load the model's static grid to get geolat/geolon
    static_ds = xr.open_dataset(config.filepath_static)
    model_lats = static_ds['geolat'].isel(yh=slice(config.lat_range[0], config.lat_range[1]), xh=slice(config.lon_range[0], config.lon_range[1])).values
    model_lons = static_ds['geolon'].isel(yh=slice(config.lat_range[0], config.lat_range[1]), xh=slice(config.lon_range[0], config.lon_range[1])).values
    
    # 4. Create a KD-Tree for efficient nearest-neighbor search
    grid_points = np.vstack([model_lats.ravel(), model_lons.ravel()]).T
    kdtree = cKDTree(grid_points)

    # 5. Initialize output tensors
    observations_tensor = torch.zeros(true_sample_shape)
    observed_mask_tensor = torch.zeros(true_sample_shape, dtype=torch.bool)
    obs_points_for_plotting = []

    # 6. Map each observation to the nearest model grid cell
    for i in range(len(obs_in_window.profile)):
        obs_lat = obs_in_window.lat[i].item()
        obs_lon = obs_in_window.lon[i].item()
        
        _, nearest_idx = kdtree.query([obs_lat, obs_lon])
        y, x = np.unravel_index(nearest_idx, model_lats.shape)

        temp_val = obs_in_window.T[i, 0].item()
        if not np.isnan(temp_val):
            norm_temp = (temp_val - config.T_range[0]) / (config.T_range[1] - config.T_range[0])
            observations_tensor[0, 0, y, x] = norm_temp
            observed_mask_tensor[0, 0, y, x] = True
            obs_points_for_plotting.append((0, y, x, norm_temp))

        if config.use_salinity:
            salt_val = obs_in_window.S[i, 0].item()
            if not np.isnan(salt_val):
                norm_salt = (salt_val - config.S_range[0]) / (config.S_range[1] - config.S_range[0])
                observations_tensor[0, 1, y, x] = norm_salt
                observed_mask_tensor[0, 1, y, x] = True
                obs_points_for_plotting.append((1, y, x, norm_salt))

    num_obs = len(obs_points_for_plotting)
    print(f"Successfully created observation tensor with {num_obs} points.")
    return observations_tensor, observed_mask_tensor, obs_points_for_plotting
