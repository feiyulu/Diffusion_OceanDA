# --- data_utils.py ---
# This file handles the loading and preprocessing of ocean data from NetCDF files.
import numpy as np
import torch
import xarray as xr
import pandas as pd
import datetime
import cftime
import os.path
from scipy.spatial import cKDTree


def get_time_coordinates(config):
    """
    Scans the NetCDF files to get a list of all available time coordinates.
    """
    filepath_t = config.filepath_t
    print(f"Scanning time coordinates from: {filepath_t}")
    with xr.open_mfdataset(filepath_t, combine='by_coords', decode_cf=True, chunks={}) as ds:
        time_coords = ds.sel(time=slice(config.training_day_range[0], config.training_day_range[1])).time
        return time_coords[::config.training_day_interval].values

def load_single_ocean_slice(config, time_coord, return_doy=False):
    """
    Loads and processes a single time slice of ocean data.
    """
    filepath_t = config.filepath_t
    filepath_s = config.filepath_s if config.use_salinity else None
    data_vars = []
    
    def _load_and_process_slice(filepaths, varname, time_coord, min_val, max_val):
        with xr.open_mfdataset(filepaths, combine='by_coords', decode_cf=True, chunks={'time': 1}) as ds:
            # Use a more robust check for renaming coordinates
            if config.varname_lat and config.varname_lat != 'lat':
                ds = ds.rename({config.varname_lat:'lat'})
            if config.varname_lon and config.varname_lon != 'lon':
                ds = ds.rename({config.varname_lon:'lon'})
            
            da_sliced = ds[varname].sel(time=time_coord, method='nearest').isel(
                z=slice(config.depth_range[0], config.depth_range[1]),
                lat=slice(config.lat_range[0], config.lat_range[1]), 
                lon=slice(config.lon_range[0], config.lon_range[1])
            ).load()
        
        data_np = da_sliced.values.astype(np.float32)
        data_np[np.isnan(data_np)] = 0.0
        normalized_data_np = (data_np - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else np.full_like(data_np, 0.5)
        return normalized_data_np

    temp_np = _load_and_process_slice(filepath_t, config.varname_t, time_coord, config.T_range[0], config.T_range[1])
    data_vars.append(temp_np)

    if config.use_salinity:
        sal_np = _load_and_process_slice(filepath_s, config.varname_s, time_coord, config.S_range[0], config.S_range[1])
        data_vars.append(sal_np)

    data_tensor = torch.tensor(np.stack(data_vars, axis=0), dtype=torch.float32)
    
    conditional_data = {}
    if config.conditioning_configs:
        if 'dayofyear' in config.conditioning_configs or return_doy:
            day_of_year = time_coord.timetuple().tm_yday
            conditional_data['dayofyear'] = torch.tensor(day_of_year, dtype=torch.long)
        
        if 'co2' in config.conditioning_configs and config.co2_filepath:
            with xr.open_dataset(config.co2_filepath) as co2_ds:
                pandas_time = pd.to_datetime(str(time_coord))
                co2_da = co2_ds[config.co2_varname].sel(time=pandas_time, method='nearest').load()
                co2_normalized = (co2_da.values - config.co2_range[0]) / (config.co2_range[1] - config.co2_range[0])
                conditional_data['co2'] = torch.tensor(co2_normalized, dtype=torch.float32)

    return data_tensor, conditional_data

def load_real_observations(config, target_time_pd):
    """
    Loads and processes real-world observations for a specific day.
    It handles different data sources like SST and Argo, projecting them onto the model's grid.
    Loads real observations (e.g., SST, Argo) for a specific time,
    projects them onto the model grid, and returns them as tensors.

    Args:
        config (Config): The experiment configuration.
        target_time_pd (pd.Timestamp): The timestamp for which to load observations.

    Returns:
        A tuple of tensors: (observations, observed_mask, guidance_strength_mask)
    """
    print(f"Loading real observations for {target_time_pd.strftime('%Y-%m-%d')}...")
    C, D, H, W = config.channels, *config.data_shape
    device = config.device

    # --- Grid setup for projection ---
    # Use a KD-tree for fast nearest-neighbor lookup of sparse observations.
    with xr.open_dataset(config.filepath_static) as static_ds:
        if not (config.varname_lat=='lat' and config.varname_lon=='lon'):
            static_ds = static_ds.rename({config.varname_lat:'lat', config.varname_lon:'lon'})
        
        model_lat_2d = static_ds['geolat'].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values
        model_lon_2d = static_ds['geolon'].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values

    grid_points = np.vstack([model_lat_2d.ravel(), model_lon_2d.ravel()]).T
    kdtree = cKDTree(grid_points)

    def find_nearest_grid_cell(obs_lat, obs_lon):
        """A helper function to project a lat/lon point to the nearest model grid index."""
        """Finds the nearest model grid cell (y, x) indices for a given lat/lon."""
        # Ensure longitude is in the model grid's range (e.g., 0-360)
        obs_lon = obs_lon % 360 if np.min(model_lon_2d) >= 0 else obs_lon
        _dist, idx = kdtree.query([obs_lat, obs_lon], k=1)
        return np.unravel_index(idx, model_lat_2d.shape)

    # Initialize empty tensors
    observations = torch.zeros((1, C, D, H, W), device=device)
    observed_mask = torch.zeros((1, C, D, H, W), dtype=torch.bool, device=device)
    guidance_strength_mask = torch.zeros((1, C, D, H, W), device=device)

    # --- Loop through observation sources ---
    for source in config.observation_sources:
        if not source.get("enabled", False):
            continue

        source_type = source.get("type")
        guidance_strength = source.get("guidance_strength", 1.0)

        # Handler for gridded sea-surface temperature data.
        if source_type == "real_sst":
            print(f"  - Processing source: {source['name']} (SST)")
            try:
                sst_filepath = source["filepath_template"].format(year=target_time_pd.year)
                with xr.open_dataset(sst_filepath) as ds:
                    calendar = ds.time.encoding.get('calendar', 'standard')
                    target_cftime = cftime.datetime(target_time_pd.year, target_time_pd.month, target_time_pd.day, calendar=calendar)

                    sst_da_sliced = ds['sst'].sel(time=target_cftime, method='nearest').isel(
                        lat=slice(config.lat_range[0], config.lat_range[1]),
                        lon=slice(config.lon_range[0], config.lon_range[1])
                    ).load()
                    
                    sst_np = sst_da_sliced.values
                    valid_mask = (sst_np > -1e30)
                    
                    sst_normalized = (sst_np[valid_mask] - config.T_range[0]) / (config.T_range[1] - config.T_range[0])
                    
                    channel_idx = source.get("target_channel", 0)
                    depth_idx = 0  # SST is at the surface
                    observations[0, channel_idx, depth_idx, :, :][valid_mask] = torch.from_numpy(sst_normalized).to(device)
                    observed_mask[0, channel_idx, depth_idx, :, :][valid_mask] = True
                    guidance_strength_mask[0, channel_idx, depth_idx, :, :][valid_mask] = guidance_strength
                    print(f"    ...found {np.sum(valid_mask)} valid SST observations on the regridded file.")

            except Exception as e:
                print(f"    ...could not process SST source {source['name']}. Error: {e}")

        # Handler for sparse, vertical Argo profile data.
        elif source_type == "real_argo":
            print(f"  - Processing source: {source['name']} (Argo)")
            try:
                argo_filepath = source["filepath_template"].format(year=target_time_pd.year)
                with xr.open_dataset(argo_filepath) as argo_ds:
                    time_window_days = source.get("time_window_days", 1)
                    start_time = target_time_pd - pd.to_timedelta(time_window_days / 2, 'd')
                    end_time = target_time_pd + pd.to_timedelta(time_window_days / 2, 'd')
                    
                    time_mask = (argo_ds['time'].values >= np.datetime64(start_time)) & (argo_ds['time'].values <= np.datetime64(end_time))
                    profiles_in_window = argo_ds.isel(profile=time_mask)
                    print(f"    ...found {len(profiles_in_window['profile'])} Argo profiles in time window.")

                    for i in range(len(profiles_in_window['profile'])):
                        profile = profiles_in_window.isel(profile=i)
                        lat_idx, lon_idx = find_nearest_grid_cell(profile['lat'].item(), profile['lon'].item())
                        
                        # Temperature
                        T_profile = profile['T'].values
                        valid_T = ~np.isnan(T_profile)
                        if np.any(valid_T):
                            norm_T = (T_profile[valid_T] - config.T_range[0]) / (config.T_range[1] - config.T_range[0])
                            depth_indices = np.arange(len(T_profile))[valid_T]
                            
                            safe_mask = depth_indices < D
                            if np.any(safe_mask):
                                observations[0, 0, depth_indices[safe_mask], lat_idx, lon_idx] = torch.from_numpy(norm_T[safe_mask]).to(device)
                                observed_mask[0, 0, depth_indices[safe_mask], lat_idx, lon_idx] = True
                                guidance_strength_mask[0, 0, depth_indices[safe_mask], lat_idx, lon_idx] = guidance_strength
            except Exception as e:
                print(f"    ...could not process Argo source {source['name']}. Error: {e}")

    return observations, observed_mask, guidance_strength_mask

def load_static_data(config):
    """
    Loads time-invariant data associated with the model grid.
    Loads static data: land mask, location embeddings, and area weights.
    """
    print("Loading static data (mask, location embeddings, area weights)...")
    with xr.open_dataset(config.filepath_static) as static_ds:
        if not (config.varname_lat=='lat' and config.varname_lon=='lon'):
            static_ds = static_ds.rename({config.varname_lat:'lat', config.varname_lon:'lon'})
            
        mask_da = static_ds[config.mask_varname].isel(
            lat=slice(config.lat_range[0], config.lat_range[1]), 
            lon=slice(config.lon_range[0], config.lon_range[1])
        )
        if 'z' in mask_da.dims:
            mask_da = mask_da.isel(z=slice(config.depth_range[0], config.depth_range[1]))
        
        static_mask_np = mask_da.values.astype(np.float32)
        if len(static_mask_np.shape) == 2:
            # If the mask is 2D, it's a surface mask. We need to create a 3D mask.
            static_mask_np = np.expand_dims(static_mask_np, axis=0).repeat(config.data_shape[0], axis=0)
        # --- Correct 3D Mask Generation using Bathymetry ---
        # Check if the loaded mask is 2D (a surface mask).
        if len(mask_da.shape) == 2:
            print("Generating 3D mask from 2D surface mask and ocean depth...")
            if not config.filepath_z_static or not os.path.exists(config.filepath_z_static):
                raise FileNotFoundError(f"filepath_z_static is not defined or file not found. Please point it to your z-levels file (e.g., z25.nc).")

            # Load the vertical grid coordinates (depth of each layer center).
            with xr.open_dataset(config.filepath_z_static) as z_ds:
                z_levels = z_ds['z'].isel(z=slice(config.depth_range[0], config.depth_range[1])).values
            
            # Load the ocean depth (bathymetry) data.
            depth_ocean = static_ds['depth_ocean'].isel(
                lat=slice(config.lat_range[0], config.lat_range[1]),
                lon=slice(config.lon_range[0], config.lon_range[1])
            ).values

            # Create the 3D mask by comparing layer depth to ocean depth.
            # A cell is 'ocean' (1) if its depth is less than the seafloor depth.
            static_mask_np = (z_levels[:, np.newaxis, np.newaxis] < depth_ocean[np.newaxis, :, :]).astype(np.float32) * mask_da.values
        else:
            # If the mask is already 3D, use it directly.
            print("Using pre-existing 3D mask from file.")
            static_mask_np = mask_da.values.astype(np.float32)

        land_mask_tensor = torch.tensor(static_mask_np[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32)
        
        # --- Load Area Weights for Loss Calculation ---
        # These weights account for the varying size of grid cells in area-based metrics.
        area_weights_tensor = None
        if config.area_weight_varname:
            print(f"Loading area weights from variable: {config.area_weight_varname}")
            area_da = static_ds[config.area_weight_varname].isel(
                lat=slice(config.lat_range[0], config.lat_range[1]), 
                lon=slice(config.lon_range[0], config.lon_range[1])
            )
            area_np = area_da.values.astype(np.float32)
            # Normalize the weights
            total_ocean_area = np.sum(area_np[static_mask_np[0] == 1])
            if total_ocean_area > 0:
                area_np /= total_ocean_area
            area_weights_tensor = torch.tensor(area_np, dtype=torch.float32).unsqueeze(0)
            print(f"Area weights loaded with shape: {area_weights_tensor.shape}")

        location_field_tensor = None
        # --- Generate Location Embeddings ---
        # These provide the 2D U-Net with spatial context (e.g., latitude, longitude).
        if config.location_embedding_channels > 0:
            print(f"Generating location embeddings for: {config.location_embedding_types}")
            location_channels = []
            
            lat_grid = static_ds['geolat'].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values
            lon_grid = static_ds['geolon'].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values

            if "lon_cyclical" in config.location_embedding_types:
                lon_rad = np.deg2rad(lon_grid)
                location_channels.append(np.sin(lon_rad).astype(np.float32))
                location_channels.append(np.cos(lon_rad).astype(np.float32))
            if "cos_lat" in config.location_embedding_types:
                lat_rad = np.deg2rad(lat_grid)
                location_channels.append(np.cos(lat_rad).astype(np.float32))
            if "coriolis" in config.location_embedding_types:
                omega = 7.2921e-5
                lat_rad = np.deg2rad(lat_grid)
                coriolis_f = 2 * omega * np.sin(lat_rad)
                location_channels.append((coriolis_f / (2 * omega)).astype(np.float32))
            if "depth_ocean" in config.location_embedding_types:
                depth_ocean_data = static_ds['depth_ocean'].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values
                # Replace NaNs (representing land) with 0 before normalization
                np.nan_to_num(depth_ocean_data, copy=False, nan=0.0)
                normalized_depth = (depth_ocean_data / 6000.0).astype(np.float32)
                location_channels.append(normalized_depth)
            
            if location_channels:
                location_field_single_sample = np.stack(location_channels, axis=0)
                location_field_tensor = torch.tensor(location_field_single_sample, dtype=torch.float32).unsqueeze(0)
                print(f"Location embedding created with shape: {location_field_tensor.shape}")

    return land_mask_tensor, location_field_tensor, area_weights_tensor

def load_test_ocean_slice(config, year, day_of_year):
    """
    Loads a single day of data from the test set, used as the "ground truth" for sampling experiments.
    Loads a single time slice of test data.
    """
    target_time_pd = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(day_of_year, unit='d')
    filepath_t = config.filepath_t_test
    filepath_s = config.filepath_s_test if config.use_salinity else None

    def _load_and_process_slice(filepaths, varname, target_time, min_val, max_val):
        with xr.open_mfdataset(filepaths, combine='by_coords', decode_cf=True, chunks={'time': 1}) as ds:
            if config.varname_lat and config.varname_lat != 'lat':
                ds = ds.rename({config.varname_lat:'lat'})
            if config.varname_lon and config.varname_lon != 'lon':
                ds = ds.rename({config.varname_lon:'lon'})
            calendar = ds.time.encoding.get('calendar', 'standard')
            target_cftime = cftime.datetime(target_time.year, target_time.month, target_time.day, calendar=calendar)
            
            da_sliced = ds[varname].sel(time=target_cftime, method='nearest').isel(
                z=slice(config.depth_range[0], config.depth_range[1]),
                lat=slice(config.lat_range[0], config.lat_range[1]),
                lon=slice(config.lon_range[0], config.lon_range[1])
            ).load()
            
        data_np = da_sliced.values.astype(np.float32)
        data_np[np.isnan(data_np)] = 0.0
        normalized_data_np = (data_np - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else np.full_like(data_np, 0.5)
        return normalized_data_np

    temp_np = _load_and_process_slice(filepath_t, config.varname_t, target_time_pd, config.T_range[0], config.T_range[1])
    data_vars = [temp_np]
    if config.use_salinity:
        sal_np = _load_and_process_slice(filepath_s, config.varname_s, target_time_pd, config.S_range[0], config.S_range[1])
        data_vars.append(sal_np)
    
    data_tensor = torch.tensor(np.stack(data_vars, axis=0), dtype=torch.float32).unsqueeze(0)

    conditional_data = {}
    if config.conditioning_configs:
        if 'dayofyear' in config.conditioning_configs:
            conditional_data['dayofyear'] = torch.tensor(target_time_pd.dayofyear, dtype=torch.long)
        if 'co2' in config.conditioning_configs and config.co2_filepath:
            with xr.open_dataset(config.co2_filepath) as co2_ds:
                co2_da = co2_ds[config.co2_varname].sel(time=target_time_pd, method='nearest').load()
                co2_normalized = (co2_da.values - config.co2_range[0]) / (config.co2_range[1] - config.co2_range[0])
                conditional_data['co2'] = torch.tensor(co2_normalized, dtype=torch.float32)

    return data_tensor, conditional_data
