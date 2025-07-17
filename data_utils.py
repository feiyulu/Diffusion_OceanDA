# --- data_utils.py ---
# This file handles synthetic data generation and provides a placeholder for loading real data.
import numpy as np
import torch
import xarray as xr # xarray is commonly used for scientific datasets like ocean data
import pandas as pd

def load_ocean_data(
    filepath_t, 
    image_size, 
    channels, 
    time_range=["20130101","20131231"],
    time_interval=1,
    lat_range=[-90,90], 
    lon_range=[0,360],
    use_salinity=False,
    filepath_mask=None,
    mask_varname=None, 
    filepath_s=None, 
    variable_names=['sst', 'sss'], 
    T_range=None, 
    S_range=None,
    co2_filepath=None,
    co2_varname="co2",
    co2_range=[320,450],
    conditioning_configs=None,
    use_coriolis_embedding=False): 
    """
    Loads ocean data from xarray/NetCDF files for temperature and optionally salinity,
    applies spatial and temporal slicing, combines into channels, and normalizes.
    It now handles separate normalization ranges for temperature and salinity,
    and extracts day of year and mean lat/lon for each sample.

    Args:
        filepath_t (list or str): Path(s) to the data file(s) for temperature (e.g., NetCDF).
        image_size (tuple): The expected spatial dimensions (height, width).
        channels (int): Number of channels (1 for Temp, 2 for Temp+Salinity).
        time_range (list): [start_time, end_time] or [select_time] for time dimension.
        time_interval (int): slice interval
        lat_range (list): [start_idx, end_idx] for latitude dimension (indices).
        lon_range (list): [start_idx, end_idx] for longitude dimension (indices).
        filepath_mask (list or str, optional): Path to the land mask file
        filepath_s (list or str, optional): Path(s) to the data file(s) for salinity. Required if channels is 2.
        variable_names (list): List of variable names to load as channels, e.g., ['sst', 'sss'].
        T_range (list, optional): List of minimum values for normalization for temperature.
        S_range (list, optional): List of maximum values for normalization for salinity.
        co2_filepath (str, optional): Path to CO2 data file.
        co2_varname (str): Variable name for CO2 in the file.
        co2_range (list): Min/max values for CO2 normalization.
        conditioning_configs (dict): Dictionary from config specifying which conditioners to load.

    Returns:
        tuple: (torch.Tensor, torch.Tensor, dict, torch.Tensor, list, list) -
               data_tensor, land_mask_tensor, conditional_data_dict, location_field_tensor, actual_T_range, actual_S_range.
               The mask identifies where data is valid (1) vs. NaN/land (0).
    """
    img_h, img_w = image_size
    print(f"Loading real ocean data of size {img_h}x{img_w} with {channels} channel(s)...")

    data_vars = []
    actual_T_range = []
    actual_S_range = []
    conditional_data = {}

    # This inner function is a refactoring of the original loading logic
    def _load_and_process_variable(filepaths, varname, mask_static, min_val_in=None, max_val_in=None):
        ds = xr.open_mfdataset(filepaths, combine='by_coords', decode_cf=True)
        da = ds[varname]

        if len(time_range)==2:
            da_sliced = da.sel(time=slice(time_range[0], time_range[1],time_interval)).isel(
                lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))
        elif len(time_range)==1:
            da_sliced = da.isel(time=time_range).isel(
                lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1]))
        
        data_np = da_sliced.values.astype(np.float32)
        mask_var = mask_static if mask_static is not None else ~np.isnan(data_np).all(axis=0)
        data_np[np.isnan(data_np)] = 0.0

        min_val, max_val = (min_val_in, max_val_in) if min_val_in and max_val_in else (data_np.min(), data_np.max())
        
        normalized_data_np = (data_np - min_val) / (max_val - min_val) if (max_val - min_val) > 1e-6 else np.full_like(data_np, 0.5)
        normalized_data_np *= mask_var[None, :, :]

        return normalized_data_np, mask_var, [min_val, max_val], da_sliced.time

    # --- Main Loading Logic ---
    # Load a static mask first if available
    mask_ds = xr.open_dataset(filepath_mask) if filepath_mask else None
    static_mask_np = mask_ds[mask_varname].isel(lat=slice(lat_range[0], lat_range[1]), lon=slice(lon_range[0], lon_range[1])).values if mask_ds else None

    # Load Temperature Data
    temp_np, land_mask_temp, actual_T_range, time_coords = _load_and_process_variable(
        filepath_t, variable_names[0], static_mask_np, T_range[0] if T_range else None, T_range[1] if T_range else None
    )
    data_vars.append(temp_np)

    # Load Salinity Data if required
    land_mask_sal = land_mask_temp
    if use_salinity:
        sal_np, land_mask_sal, actual_S_range, _ = _load_and_process_variable(
            filepath_s, variable_names[1], static_mask_np, S_range[0] if S_range else None, S_range[1] if S_range else None
        )
        data_vars.append(sal_np)

    # --- Load Conditional Data based on config ---
    if conditioning_configs and 'dayofyear' in conditioning_configs:
        day_of_year_list = [t.dt.dayofyear.item() for t in time_coords]
        conditional_data['dayofyear'] = torch.tensor(day_of_year_list, dtype=torch.long)
        print(f"Loaded 'dayofyear' conditional data, shape: {conditional_data['dayofyear'].shape}")
        
    if conditioning_configs and 'co2' in conditioning_configs and co2_filepath:
        co2_ds = xr.open_dataset(co2_filepath)
        if len(time_range)==2:
            co2_da = co2_ds[co2_varname].sel(time=slice(time_range[0], time_range[1],time_interval))
        elif len(time_range)==1:
            co2_da = co2_ds[co2_varname].isel(time=time_range)
        co2_normalized = (co2_da.values - co2_range[0]) / (co2_range[1] - co2_range[0])
        conditional_data['co2'] = torch.tensor(co2_normalized, dtype=torch.float32)
        print(f"Loaded 'co2' conditional data, shape: {conditional_data['co2'].shape}")

    # --- Load Spatial/Location Embedding ---
    ds_ref = xr.open_mfdataset(filepath_t, combine='by_coords')
    actual_lats = ds_ref.lat.isel(lat=slice(lat_range[0], lat_range[1])).values
    actual_lons = ds_ref.lon.isel(lon=slice(lon_range[0], lon_range[1])).values
    lat_grid, lon_grid = np.meshgrid(actual_lats, actual_lons, indexing='ij')

    normalized_lat_grid = (lat_grid / 90.0).astype(np.float32)
    normalized_lon_grid = ((lon_grid / 180.0) - 1.0).astype(np.float32)

    location_channels = [normalized_lat_grid, normalized_lon_grid]

    if use_coriolis_embedding:
        print("Calculating and adding Coriolis parameter to location embedding...")
        omega = 7.2921e-5  # Earth's rotation rate in rad/s
        lat_rad = np.deg2rad(lat_grid)
        coriolis_f = 2 * omega * np.sin(lat_rad)
        
        # Normalize by the maximum possible value of f (at the pole) to scale to [-1, 1]
        normalized_coriolis = (coriolis_f / (2 * omega)).astype(np.float32)
        location_channels.append(normalized_coriolis)

    location_field_single_sample = np.stack(location_channels, axis=0)
    location_field_tensor = torch.tensor(location_field_single_sample, dtype=torch.float32).unsqueeze(0)
    print(f"Location embedding data shape: {location_field_tensor.shape}")

    # --- Finalize and Return ---
    data_tensor = torch.tensor(np.stack(data_vars, axis=1), dtype=torch.float32)
    combined_land_mask_np = (land_mask_temp * land_mask_sal).astype(np.float32)
    land_mask_tensor = torch.tensor(combined_land_mask_np[None, None, :, :], dtype=torch.float32)
    
    # Ensure data is zeroed out on land
    full_mask_expanded = land_mask_tensor.repeat(data_tensor.shape[0], data_tensor.shape[1], 1, 1)
    data_tensor = data_tensor * full_mask_expanded

    print(f"Loaded and normalized data shape: {data_tensor.shape}")
    print(f"Combined Land mask shape: {land_mask_tensor.shape}")

    return (data_tensor, land_mask_tensor, conditional_data, 
            location_field_tensor, actual_T_range, actual_S_range)
