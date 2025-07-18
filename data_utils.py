# --- data_utils.py ---
# This file handles the loading and preprocessing of ocean data from NetCDF files.
import numpy as np
import torch
import xarray as xr
import pandas as pd

def load_ocean_data(config, time_range, is_test_data=False): 
    """
    Loads ocean data from xarray/NetCDF files, processes it, and prepares it for the model.
    This version is refactored to accept a single config object for most parameters.

    Args:
        config (Config): The main configuration object for the project.
        time_range (list): [start_time, end_time] or [select_time] for the time dimension.
        is_test_data (bool): If True, loads test data filepaths from config. Otherwise, loads training filepaths.

    Returns:
        tuple: (torch.Tensor, torch.Tensor, dict, torch.Tensor, list, list) -
               data_tensor, land_mask_tensor, conditional_data_dict, location_field_tensor, actual_T_range, actual_S_range.
    """
    data_d, data_h, data_w = config.data_shape
    print(f"Loading 3D ocean data of size {data_d}x{data_h}x{data_w} with {config.channels} channel(s)...")

    # Select filepaths based on whether this is for training or testing/validation
    filepath_t = config.filepath_t_test if is_test_data else config.filepath_t
    filepath_s = config.filepath_s_test if is_test_data and config.use_salinity else config.filepath_s

    data_vars = []
    actual_T_range = []
    actual_S_range = []
    conditional_data = {}

    # --- Inner Helper Function for Data Loading and Processing ---
    def _load_and_process_variable(filepaths, varname, mask_static, min_val_in=None, max_val_in=None):
        """Loads, slices, normalizes, and masks a single data variable."""
        # Load dataset, handling multiple files if necessary
        ds = xr.open_mfdataset(filepaths, combine='by_coords', decode_cf=True)
        if not ( config.varname_lat=='lat' and config.varname_lon=='lon'):
            ds = ds.rename({config.varname_lat:'lat',config.varname_lon:'lon'})
        da = ds[varname]

        # Slice data based on the provided time, latitude, and longitude ranges
        if len(time_range)==2:
            da_sliced = da.sel(time=slice(time_range[0], time_range[1], config.training_day_interval)).isel(
                z=slice(config.depth_range[0], config.depth_range[1]),
                lat=slice(config.lat_range[0], config.lat_range[1]), 
                lon=slice(config.lon_range[0], config.lon_range[1]))
        elif len(time_range)==1:
            da_sliced = da.isel(time=time_range).isel(
                z=slice(config.depth_range[0], config.depth_range[1]),
                lat=slice(config.lat_range[0], config.lat_range[1]), 
                lon=slice(config.lon_range[0], config.lon_range[1]))
        
        # Convert to numpy array and create a mask from NaNs
        data_np = da_sliced.values.astype(np.float32)
        mask_var = mask_static if mask_static is not None else ~np.isnan(data_np).all(axis=0)
        data_np[np.isnan(data_np)] = 0.0

        # Normalize the data to a [0, 1] range
        min_val, max_val = (min_val_in, max_val_in) if min_val_in and max_val_in else (data_np.min(), data_np.max())
        normalized_data_np = (data_np - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else np.full_like(data_np, 0.5)
        
        # Apply the land/ocean mask
        normalized_data_np *= mask_var[None, :, :]

        return normalized_data_np, mask_var, [min_val, max_val], da_sliced.time

    # --- Main Loading Logic ---
    # Load a static land mask if provided in the config
    mask_ds = xr.open_dataset(config.filepath_mask) if hasattr(config, 'filepath_mask') and config.filepath_mask else None
    static_mask_np = mask_ds[config.mask_varname].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values if mask_ds else None

    # Load Temperature Data (always the first channel)
    temp_np, land_mask_temp, actual_T_range, time_coords = _load_and_process_variable(
        filepath_t, config.varname_t, static_mask_np, config.T_range[0] if config.T_range else None, config.T_range[1] if config.T_range else None
    )
    data_vars.append(temp_np)

    # Load Salinity Data if enabled
    land_mask_sal = land_mask_temp
    if config.use_salinity:
        sal_np, land_mask_sal, actual_S_range, _ = _load_and_process_variable(
            filepath_s, config.varname_s, static_mask_np, config.S_range[0] if config.S_range else None, config.S_range[1] if config.S_range else None
        )
        data_vars.append(sal_np)

    # --- Load Scalar/Temporal Conditional Data ---
    if config.conditioning_configs and 'dayofyear' in config.conditioning_configs:
        day_of_year_list = [t.dt.dayofyear.item() for t in time_coords]
        conditional_data['dayofyear'] = torch.tensor(day_of_year_list, dtype=torch.long)
        
    if config.conditioning_configs and 'co2' in config.conditioning_configs and config.co2_filepath:
        co2_ds = xr.open_dataset(config.co2_filepath)
        if len(time_range)==2:
            co2_da = co2_ds[config.co2_varname].sel(time=slice(time_range[0], time_range[1], config.training_day_interval))
        elif len(time_range)==1:
            co2_da = co2_ds[config.co2_varname].isel(time=time_range)
        co2_normalized = (co2_da.values - config.co2_range[0]) / (config.co2_range[1] - config.co2_range[0])
        conditional_data['co2'] = torch.tensor(co2_normalized, dtype=torch.float32)

    location_field_tensor = None
    if config.location_embedding_channels > 0:
        print(f"Generating location embeddings for: {config.location_embedding_types}")
        location_channels = []
        
        # # Load static grid file which may contain geolat, geolon, depth, etc.
        # static_ds = xr.open_dataset(config.filepath_static)
        # lat_grid = static_ds['geolat'].isel(yh=slice(config.lat_range[0], config.lat_range[1]), xh=slice(config.lon_range[0], config.lon_range[1])).values
        # lon_grid = static_ds['geolon'].isel(yh=slice(config.lat_range[0], config.lat_range[1]), xh=slice(config.lon_range[0], config.lon_range[1])).values

        # This embedding provides the model with explicit coordinate information.
        ds_ref = xr.open_mfdataset(filepath_t[0])
        if not ( config.varname_lat=='lat' and config.varname_lon=='lon'):
            ds_ref = ds_ref.rename({config.varname_lat:'lat',config.varname_lon:'lon'})
        actual_lats = ds_ref.lat.isel(lat=slice(config.lat_range[0], config.lat_range[1])).values
        actual_lons = ds_ref.lon.isel(lon=slice(config.lon_range[0], config.lon_range[1])).values
        lat_grid, lon_grid = np.meshgrid(actual_lats, actual_lons, indexing='ij')

        # Build the embedding tensor channel by channel based on the config list
        if "lat" in config.location_embedding_types:
            location_channels.append((lat_grid / 90.0).astype(np.float32))

        if "lon" in config.location_embedding_types:
            location_channels.append(((lon_grid / 180.0) - 1.0).astype(np.float32))

        if "lon_cyclical" in config.location_embedding_types:
            lon_rad = (lon_grid / 360.0) * (2 * np.pi)
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

        if "ocean_depth" in config.location_embedding_types:
            depth = static_ds['wet'].isel(yh=slice(config.lat_range[0], config.lat_range[1]), xh=slice(config.lon_range[0], config.lon_range[1])).values
            # Normalize by a typical max depth
            normalized_depth = (depth / 6000.0).astype(np.float32)
            location_channels.append(normalized_depth)
        
        # Stack the generated channels into a single tensor
        location_field_single_sample = np.stack(location_channels, axis=0)
        location_field_tensor = torch.tensor(location_field_single_sample, dtype=torch.float32).unsqueeze(0)
        print(f"Location embedding created with shape: {location_field_tensor.shape}")

    # Stack the channels into a single tensor
    location_field_single_sample = np.stack(location_channels, axis=0)
    location_field_tensor = torch.tensor(location_field_single_sample, dtype=torch.float32).unsqueeze(0)
    print(f"Location embedding created with shape: {location_field_tensor.shape}")

    # --- Finalize and Return ---
    # Stack the data variables (e.g., Temp, Salinity) into a single tensor
    data_tensor = torch.tensor(np.stack(data_vars, axis=1), dtype=torch.float32)
    
    # Combine the masks from all variables to create a final land mask
    combined_mask_np = (land_mask_temp * land_mask_sal).astype(np.float32)
    land_mask_tensor = torch.tensor(combined_mask_np[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32)
    
    # Ensure all data points on land are zeroed out
    full_mask_expanded = land_mask_tensor.repeat(1, data_tensor.shape[1], 1, 1, 1)
    data_tensor = data_tensor * full_mask_expanded

    print(f"Loaded and normalized data shape: {data_tensor.shape}")

    return (data_tensor, land_mask_tensor, conditional_data, 
            location_field_tensor, actual_T_range, actual_S_range)
