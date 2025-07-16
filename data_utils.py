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
    co2_range=[320,450]):
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
        min_val_in (list, optional): List of minimum values for normalization for each channel [min_temp, min_sal].
        max_val_in (list, optional): List of maximum values for normalization for each channel [max_temp, max_sal].
        co2_filepath:
        co2_varname:
        co2_range:

    Returns:
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, list) -
               data tensor, land_mask_tensor, day_of_year_tensor, location_field_tensor, actual min_vals, actual max_vals. 
               location_field_tensor
               The mask identifies where data is valid (1) vs. NaN/land (0).
    """
    img_h, img_w = image_size
    print(f"Loading real ocean data of size {img_h}x{img_w} with {channels} channel(s)...")

    data_vars = []
    actual_T_range = []
    actual_S_range = []
    day_of_year_list = []
    co2_daily_list = []
    location_field_list = []

    def _load_and_process_data(
        filepaths, varname, 
        filepath_mask=None, mask_varname=None, 
        min_val_in=None, max_val_in=None):

        ds = xr.open_mfdataset(filepaths, combine='by_coords', decode_cf=True)
        da = ds[varname]

        # Slice time, lat, lon
        if len(time_range)==2:
            da_sliced = da.sel(
                time=slice(time_range[0], time_range[1], time_interval)).isel(
                    lat=slice(lat_range[0], lat_range[1]),
                    lon=slice(lon_range[0], lon_range[1]))
        elif len(time_range)==1:
            da_sliced = da.isel(time=time_range).isel(
                lat=slice(lat_range[0], lat_range[1]),lon=slice(lon_range[0], lon_range[1]))

        # Get actual lat/lon values for the sliced range
        actual_lats = da.lat.isel(lat=slice(lat_range[0], lat_range[1])).values
        actual_lons = da.lon.isel(lon=slice(lon_range[0], lon_range[1])).values
        
        # NEW: Create 2D latitude and longitude fields
        lat_grid, lon_grid = np.meshgrid(actual_lats, actual_lons, indexing='ij')
        
        # Normalize lat/lon grids to [-1, 1] for stable input to model
        normalized_lat_grid = (lat_grid - (-90)) / (90 - (-90)) * 2 - 1
        normalized_lon_grid = (lon_grid - (0)) / (360 - (0)) * 2 - 1
        
        # Stack into a (2, H, W) tensor
        location_field_single_sample = np.stack([normalized_lat_grid, normalized_lon_grid], axis=0).astype(np.float32)
        location_field_list.append(location_field_single_sample)

        for t_idx in range(da_sliced.sizes['time']):
            doy = da_sliced.isel(time=t_idx).time.dt.dayofyear.item()
            day_of_year_list.append(doy)
            # location_field_list.append(location_field_single_sample)

        # Convert to numpy and handle NaNs for masking
        data_np = da_sliced.values.astype(np.float32)
        
        # Load land mask or calculate land mask based on NaNs in this specific variable
        if filepath_mask and mask_var:
            mask_ds = xr.open_dataset(filepath_mask)
            mask_static = mask_ds[mask_varname].isel(
                lat=slice(lat_range[0], lat_range[1]),
                lon=slice(lon_range[0], lon_range[1])).values
            mask_var = mask_static.unsqueeze(0).repeat(data_np.shape[0],1,1)
        else:
            mask_var = ~np.isnan(data_np).all(axis=0)
            mask_var = mask_var.astype(np.float32)

        data_np[np.isnan(data_np)] = 0.0 

        if min_val_in and max_val_in:
            min_val = min_val_in
            max_val = max_val_in
        else:
            min_val = data_np.min()
            max_val = data_np.max()

        normalized_data_np = np.zeros_like(data_np)
        if (max_val - min_val) > 1e-6:
            normalized_data_np = (data_np - min_val) / (max_val - min_val)
        else:
            normalized_data_np = np.full_like(data_np, 0.5)

        normalized_data_np *= mask_var[None, :, :]

        return normalized_data_np, mask_var, min_val, max_val

    # Load Temperature Data
    if T_range:
        temp_np, land_mask_temp, min_t, max_t = _load_and_process_data(
            filepath_t, variable_names[0], filepath_mask, mask_varname,
            T_range[0], T_range[1])
    else:
        temp_np, land_mask_temp, min_t, max_t = _load_and_process_data(
            filepath_t, variable_names[0], filepath_mask, mask_varname)

    data_vars.append(temp_np)
    actual_T_range.append(min_t)
    actual_T_range.append(max_t)

    # Load Salinity Data if required
    land_mask_sal = land_mask_temp
    if channels == 2 and filepath_s is not None and len(variable_names) > 1:
        if min_val_in and max_val_in:
            sal_np, land_mask_sal, min_s, max_s = _load_and_process_data(
                filepath_s, variable_names[1], filepath_mask, mask_varname,
                min_val_in[1], max_val_in[1])
        else:
            sal_np, land_mask_sal, min_s, max_s = _load_and_process_data(
                filepath_s, variable_names[1], filepath_mask, mask_varname,)
        data_vars.append(sal_np)
        actual_S_range.append(min_s)
        actual_S_range.append(max_s)
        
    elif channels == 2 and (filepath_s is None or len(variable_names) <= 1):
        print("Warning: Configured for 2 channels (Temp+Salinity) but salinity file or varname is missing. Proceeding with 1 channel (Temperature only).")
        channels = 1

    co2_ds = xr.open_dataset(co2_filepath)
    co2_da = co2_ds[co2_varname]
    if len(time_range)==2:
        co2_sliced = co2_da.sel(time=slice(time_range[0], time_range[1], time_interval))
    elif len(time_range)==1:
        co2_sliced = co2_da.isel(time=time_range)
        
    co2_normalized = (co2_sliced-co2_range[0])/(co2_range[1]-co2_range[0])
    data_tensor = torch.tensor(np.stack(data_vars, axis=1), dtype=torch.float32)

    combined_land_mask_np = (land_mask_temp * land_mask_sal).astype(np.float32)
    land_mask_tensor = torch.tensor(combined_land_mask_np[None, None, :, :], dtype=torch.float32)

    full_mask_expanded = land_mask_tensor.repeat(data_tensor.shape[0], data_tensor.shape[1], 1, 1)
    data_tensor = data_tensor * full_mask_expanded

    print(f"Loaded and normalized data shape: {data_tensor.shape}")
    print(f"Combined Land mask shape: {land_mask_tensor.shape}")

    day_of_year_tensor = torch.tensor(day_of_year_list, dtype=torch.long)
    co2_daily_tensor = torch.tensor(co2_normalized.values, dtype=torch.long)
    # Convert list of 2D location fields to a single tensor (N, 2, H, W)
    location_field_tensor = torch.tensor(np.stack(location_field_list, axis=0), dtype=torch.float32) 
    
    print(f"Dayofyear embedding data shape: {day_of_year_tensor.shape}")
    print(f"CO2 embedding data shape: {co2_daily_tensor.shape}")
    print(f"Location embedding data shape: {location_field_tensor.shape}")

    return data_tensor, land_mask_tensor, \
        day_of_year_tensor, co2_daily_tensor, location_field_tensor, \
            actual_T_range, actual_S_range