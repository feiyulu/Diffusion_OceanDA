# --- data_utils.py ---
# This file handles synthetic data generation and provides a placeholder for loading real data.
import numpy as np
import torch
import xarray as xr # xarray is commonly used for scientific datasets like ocean data

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
        time_range (list): [start_time, end_time] for time dimension.
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

        ds = xr.open_mfdataset(filepaths, combine='by_coords')
        da = ds[varname]

        # Slice time, lat, lon
        da_sliced = da.sel(
            time=slice(time_range[0], time_range[1], time_interval)
            ).isel(
                lat=slice(lat_range[0], lat_range[1]),
                lon=slice(lon_range[0], lon_range[1]))
        
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
    co2_sliced = co2_da.sel(time=slice(time_range[0], time_range[1], time_interval))
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

def generate_synthetic_ocean_data(num_samples, image_size, channels, use_salinity, channel_wise_normalization=False):
    """
    Generates synthetic ocean temperature and optionally salinity data.
    This function simulates a 2D ocean grid with some patterns and a land mask.
    It now supports non-square image dimensions and performs global normalization.
    Enhanced with more prominent periodic patterns for visualization.

    Args:
        num_samples (int): Number of synthetic data samples to generate.
        image_size (tuple): The spatial dimensions (height, width) of the grid.
        channels (int): Number of channels (1 for Temp, 2 for Temp+Salinity).
        use_salinity (bool): If True, also generates salinity data.
        channel_wise_normalization (bool): If False, normalize all channels together based on global min/max.

    Returns:
        tuple: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -
               data tensor, land_mask_tensor, day_of_year_tensor,
               location_field_tensor: location_field_tensor instead of lat_lon_tensor
    """
    img_h, img_w = image_size
    print(f"Generating synthetic ocean data of size {img_h}x{img_w} with {channels} channel(s)...")
    
    raw_data = []
    # Create a simple land mask (e.g., a square "island" in the middle)
    mask_np = np.ones((img_h, img_w), dtype=np.float32)
    
    # Adjust land mask to be a rectangular area relative to the new dimensions
    land_h_start = img_h // 4
    land_h_end = img_h - img_h // 4
    land_w_start = img_w // 4
    land_w_end = img_w - img_w // 4
    mask_np[land_h_start:land_h_end, land_w_start:land_w_end] = 0.0 # Set a central rectangular area to land
    
    land_mask_tensor = torch.tensor(mask_np[None, None, :, :], dtype=torch.float32) # (1, 1, H, W)

    x_coords = np.linspace(0, 4 * np.pi, img_w) 
    y_coords = np.linspace(0, 4 * np.pi, img_h) 

    # Define coordinates for a different scale waveform
    x_coords_fine = np.linspace(0, 8 * np.pi, img_w) # Higher frequency
    y_coords_fine = np.linspace(0, 8 * np.pi, img_h) # Higher frequency

    day_of_year_list = []
    # NEW: Prepare for 2D location field
    lat_field = np.linspace(-1, 1, img_h)[:, None].repeat(img_w, axis=1) # Latitude varies by row
    lon_field = np.linspace(-1, 1, img_w)[None, :].repeat(img_h, axis=0) # Longitude varies by column
    location_field_per_sample = np.stack([lat_field, lon_field], axis=0) # Shape (2, H, W)
    location_field_tensor = torch.tensor(location_field_per_sample[None, :, :, :], dtype=torch.float32) # Shape (1, 2, H, W), will be repeated

    for _ in range(num_samples):
        # Add a random phase shift for each sample to create movement
        phase_shift_x = np.random.uniform(-2 * np.pi, 2 * np.pi)
        phase_shift_y = np.random.uniform(-2 * np.pi, 2 * np.pi)

        X_shifted, Y_shifted = np.meshgrid(x_coords + phase_shift_x, y_coords + phase_shift_y)
        X_fine_shifted, Y_fine_shifted = np.meshgrid(x_coords_fine + phase_shift_x, y_coords_fine + phase_shift_y)

        # Temperature pattern with more distinct waves
        temp_base = np.random.rand(img_h, img_w) * 0.1 # Reduced base randomness
        
        # Larger scale waveform (amplitude reverted to 0.4 for stability)
        temp_pattern1_large = np.sin(Y_shifted * 1.5 + X_shifted * 0.8) * 0.4 
        # Smaller scale waveform (amplitude reduced from 0.2 to 0.1 for smoothness)
        temp_pattern2_fine = np.cos(Y_fine_shifted * 0.5 - X_fine_shifted * 2) * 0.1 
        
        temp = temp_base + temp_pattern1_large + temp_pattern2_fine
        temp = (temp - temp.min()) / (temp.max() - temp.min()) # Local normalization for pattern visibility
        temp = temp * 0.8 + 0.1 # Scale to 0.1-0.9 range
        temp = temp * mask_np # Apply land mask to temperature

        if use_salinity:
            # Salinity pattern with different distinct waves
            sal_base = np.random.rand(img_h, img_w) * 0.1
            
            # Larger scale waveform (amplitude reverted to 0.4 for stability)
            sal_pattern1_large = np.cos(Y_shifted * 2.5 + X_shifted * 1.2) * 0.4 
            # Smaller scale waveform (amplitude reduced from 0.2 to 0.1 for smoothness)
            sal_pattern2_fine = np.sin(Y_fine_shifted * 0.7 - X_fine_shifted * 3) * 0.1 
            
            sal = sal_base + sal_pattern1_large + sal_pattern2_fine
            sal = (sal - sal.min()) / (sal.max() - sal.min()) # Local normalization for pattern visibility
            sal = sal * 0.8 + 0.1 # Scale to 0.1-0.9 range
            sal = sal * mask_np # Apply land mask to salinity
            sample = np.stack([temp, sal], axis=0) # Stack both channels
        else:
            sample = temp[None, :, :] # Add channel dimension for temperature only

        raw_data.append(sample)

        # NEW: Dummy day of year generation for synthetic data
        day_of_year_list.append(np.random.randint(1, 367)) # Random day of year

    raw_data_tensor = torch.tensor(np.array(raw_data), dtype=torch.float32) # (N, C, H, W)

    # --- Global Normalization (applied over all samples and valid regions) ---
    normalized_data_tensor = torch.zeros_like(raw_data_tensor)
    
    # Expand land_mask_tensor to match raw_data_tensor's batch and channel dimensions for masking
    full_mask = land_mask_tensor.repeat(num_samples, channels, 1, 1)

    if channel_wise_normalization:
        print("Performing channel-wise global normalization...")
        for c in range(channels):
            # Only consider valid ocean points for min/max calculation
            valid_channel_data = raw_data_tensor[:, c, :, :][full_mask[:, c, :, :] == 1]
            if valid_channel_data.numel() > 0: # Check if there's any valid data
                min_val = valid_channel_data.min()
                max_val = valid_channel_data.max()
                if (max_val - min_val) > 1e-6:
                    normalized_data_tensor[:, c, :, :][full_mask[:, c, :, :] == 1] = \
                        (valid_channel_data - min_val) / (max_val - min_val)
                else:
                    normalized_data_tensor[:, c, :, :][full_mask[:, c, :, :] == 0] = 0.0 # Set land to 0
                    normalized_data_tensor[:, c, :, :][full_mask[:, c, :, :] == 1] = 0.5 # Default if all valid values are the same
            else: # No valid data in this channel
                normalized_data_tensor[:, c, :, :] = 0.0 # Set entire channel to 0 if no valid data
            # Ensure land points remain zero
            normalized_data_tensor[:, c, :, :] *= full_mask[:, c, :, :]
    else:
        print("Performing global normalization across all channels...")
        # Flatten and select only valid ocean points from all channels combined
        valid_overall_data = raw_data_tensor[full_mask == 1]
        if valid_overall_data.numel() > 0:
            min_val = valid_overall_data.min()
            max_val = valid_overall_data.max()
            if (max_val - min_val) > 1e-6:
                normalized_data_tensor[full_mask == 1] = (valid_overall_data - min_val) / (max_val - min_val)
            else:
                normalized_data_tensor[full_mask == 0] = 0.0 # Set land to 0
                normalized_data_tensor[full_mask == 1] = 0.5
        else: # No valid data
            normalized_data_tensor[:] = 0.0 # Set entire tensor to 0
        # Ensure land points remain zero
        normalized_data_tensor *= full_mask

    print(f"Normalized data shape: {normalized_data_tensor.shape}")
    print(f"Land mask shape: {land_mask_tensor.shape}")
    
    day_of_year_tensor = torch.tensor(day_of_year_list, dtype=torch.long) # Day of year as long (categorical or integer)
    # Repeat location_field_tensor for each sample in the batch
    # location_field_tensor_batch = location_field_tensor.repeat(num_samples, 1, 1, 1)

    return normalized_data_tensor, land_mask_tensor, day_of_year_tensor, location_field_tensor