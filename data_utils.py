# --- data_utils.py ---
# This file handles synthetic data generation and provides a placeholder for loading real data.
import numpy as np
import torch
import xarray as xr # xarray is commonly used for scientific datasets like ocean data
# from config import config # Already imported in the main script, but would be needed if run stand-alone

def load_ocean_data(
    filepath_t, image_size, channels, 
    time_range=[0,5000,1],lat_range=[-90,90], lon_range=[0,360], lat_name='lat', lon_name='lon',
    filepath_s=None, variable_names=['sst', 'sss'], channel_wise_normalization=True,
    max_val_in=[],min_val_in=[]):
    """
    Args:
        filepath_t/s (str): Path to the data files.
        image_size (tuple): The expected spatial dimensions (height, width).
        channels (int): Number of channels (1 for Temp, 2 for Temp+Salinity).
        variable_names (list): List of variable names to load as channels.
        channel_wise_normalization (bool): If True, normalize each channel independently.
                                           If False, normalize all channels together based on global min/max.
        max_val_in, min_val_in: preset normalization
    Returns:
        tuple: (torch.Tensor, torch.Tensor) - data tensor and a land_mask_tensor.
               The mask identifies where data is valid (1) vs. NaN/land (0).
    """
    img_h, img_w = image_size
    print(f"Loading data from {filepath_t} (placeholder - assuming {img_h}x{img_w} numpy/xarray data).")
    
    data_tensor = None
    land_mask_tensor = None

    ds_t = xr.open_mfdataset(filepath_t)
    print(f"Successfully opened Temp dataset from {filepath_t}")
    
    # Determine which variables to load based on the `channels` configuration
    loaded_vars = []
    if channels >= 1: # Always load temperature if at least 1 channel
        temp_data_np = ds_t[variable_names[0]]\
            [time_range[0]:time_range[1]:time_range[2],lat_range[0]:lat_range[1],\
                lon_range[0]:lon_range[1]].values # Shape: (time, y, x)
        print(f"Temp data shape{temp_data_np.shape}")
        print(ds_t[variable_names[0]]\
            [time_range[0]:time_range[1],lat_range[0]:lat_range[1],lon_range[0]:lon_range[1]])
        loaded_vars.append(temp_data_np)
    if channels == 2 and len(variable_names) > 1: # Load salinity if 2 channels and var name exists
        ds_s = xr.open_mfdataset(filepath_s)
        sal_data_np = ds_s[variable_names[1]]\
            [time_range[0]:time_range[1]:time_range[2],lat_range[0]:lat_range[1],\
                lon_range[0]:lon_range[1]].values # Shape: (time, y, x)
        loaded_vars.append(sal_data_np)
    else:
        print("Warning: Salinity variable not loaded or not configured for 2 channels.")

    if not loaded_vars:
        raise ValueError("No data variables could be loaded based on configuration.")

    # Create a raw data array with channels: (samples, channels, H, W)
    raw_data_np = np.stack(loaded_vars, axis=1) # Stack along new channel dim

    # Create a land mask: typically based on NaNs in the data or a dedicated mask variable
    # If NaN means land:
    if channels == 2:
        land_mask_np = ~np.isnan(raw_data_np[:, 0, :, :]) & ~np.isnan(raw_data_np[:, 1, :, :])
    else:
        land_mask_np = ~np.isnan(raw_data_np[:, 0, :, :]) # Assuming NaNs are consistent across channels
    land_mask_np = land_mask_np[:,None,:,:].astype(np.float32) # Convert bool to float for the mask
    print(f"Land mask shape{land_mask_np.shape}")
    # Fill NaNs with 0 for model input for the raw data
    raw_data_np[np.isnan(raw_data_np)] = 0.0

    raw_data_tensor = torch.tensor(raw_data_np, dtype=torch.float32)
    land_mask_tensor_for_norm = torch.tensor(land_mask_np, dtype=torch.float32) # May have N samples or 1
    print(f"Land mask tensor shape{land_mask_tensor_for_norm.shape}")

    # --- Apply Global Normalization to Real Data ---
    normalized_data_tensor = torch.zeros_like(raw_data_tensor)
    print(f"Raw data tensor shape{raw_data_tensor.shape}")
    # Ensure full_mask matches the batch and channel dimensions of raw_data_tensor
    full_mask = land_mask_tensor_for_norm.repeat(
        raw_data_tensor.shape[0] // land_mask_tensor_for_norm.shape[0], raw_data_tensor.shape[1], 1, 1)
    print(f"Full land mask shape{full_mask.shape}")

    if channel_wise_normalization:
        min_val_out=[]
        max_val_out=[]
        for c in range(raw_data_tensor.shape[1]): # Iterate over actual loaded channels
            valid_channel_data = raw_data_tensor[:, c, :, :][full_mask[:, c, :, :] == 1]
            if valid_channel_data.numel() > 0:
                if max_val_in and min_val_in:
                    min_val = min_val_in[c]
                    max_val = max_val_in[c]
                else:
                    min_val = valid_channel_data.min()
                    max_val = valid_channel_data.max()
                    min_val_out.append(min_val)
                    max_val_out.append(max_val)
                if (max_val - min_val) > 1e-6:
                    normalized_data_tensor[:, c, :, :][full_mask[:, c, :, :] == 1] = \
                        (valid_channel_data - min_val) / (max_val - min_val)
                else:
                    normalized_data_tensor[:, c, :, :][full_mask[:, c, :, :] == 0] = 0.0 # Set land to 0
                    normalized_data_tensor[:, c, :, :][full_mask[:, c, :, :] == 1] = 0.5
            else: # No valid data in this channel
                normalized_data_tensor[:, c, :, :] = 0.0 # Set entire channel to 0 if no valid data
            normalized_data_tensor[:, c, :, :] *= full_mask[:, c, :, :]
    else:
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
        normalized_data_tensor *= full_mask
    
    data_tensor = normalized_data_tensor
    
    # For consistency with synthetic data and UNet's expectation,
    # ensure a single (1, 1, H, W) land mask is returned if it's static.
    if land_mask_tensor_for_norm.shape[0] > 1: # If mask has a batch dimension, take the first one or average
        land_mask_tensor = land_mask_tensor_for_norm[0:1, :, :, :]
    else:
        land_mask_tensor = land_mask_tensor_for_norm
    
    # Ensure land_mask_tensor always has a single batch dimension (1, 1, H, W) for consistency with UNet input.
    if land_mask_tensor.shape[0] > 1:
        land_mask_tensor = land_mask_tensor[0:1, :, :, :] # Take the first mask if batch dim exists

    return data_tensor, land_mask_tensor, min_val_out, max_val_out


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
        channel_wise_normalization (bool): If True, normalize each channel independently.
                                           If False, normalize all channels together based on global min/max.

    Returns:
        torch.Tensor: A tensor of synthetic ocean data, shape (num_samples, channels, H, W).
                      Land points are set to 0 in this tensor.
        torch.Tensor: A tensor representing the land/ocean mask, shape (1, 1, H, W).
                      Value 1.0 for ocean (valid data), 0.0 for land (no data).
    """
    img_h, img_w = image_size
    print(f"Generating synthetic ocean data of size {img_h}x{img_w} with {channels} channel(s)...")
    
    raw_data = []
    # Create a simple land mask (e.g., a square "island" in the middle)
    mask_np = np.ones((img_h, img_w), dtype=np.float32)
    
    # Adjust land mask to be a rectangular area relative to the new dimensions
    land_h_start = 3 * img_h // 8
    land_h_end = img_h - 3 * img_h // 8
    land_w_start = 2 * img_w // 8
    land_w_end = img_w - 4 * img_w // 8
    mask_np[land_h_start:land_h_end, land_w_start:land_w_end] = 0.0 # Set a central rectangular area to land
    
    land_mask_tensor = torch.tensor(mask_np[None, None, :, :], dtype=torch.float32) # (1, 1, H, W)

    x_coords = np.linspace(0, 2 * np.pi, img_w) 
    y_coords = np.linspace(0, 2 * np.pi, img_h) 

    # Define coordinates for a different scale waveform
    x_coords_fine = np.linspace(0, 5 * np.pi, img_w) # Higher frequency
    y_coords_fine = np.linspace(0, 5 * np.pi, img_h) # Higher frequency


    for _ in range(num_samples):
        # Add a random phase shift for each sample to create movement
        phase_shift_x = np.pi # 1. * np.random.uniform(-2 * np.pi, 2 * np.pi)
        phase_shift_y = -np.pi # 1. * np.random.uniform(-2 * np.pi, 2 * np.pi)

        X_shifted, Y_shifted = np.meshgrid(x_coords + phase_shift_x, y_coords + phase_shift_y)
        X_fine_shifted, Y_fine_shifted = np.meshgrid(x_coords_fine + phase_shift_x, y_coords_fine + phase_shift_y)

        # Temperature pattern with more distinct waves
        temp_base = np.random.rand(img_h, img_w) * 0.1 # Reduced base randomness
        
        # Larger scale waveform (amplitude reverted to 0.4 for stability)
        temp_pattern1_large = np.sin(Y_shifted * 1.5 + X_shifted * 0.8) * 0.1 
        # Smaller scale waveform (amplitude reduced from 0.2 to 0.1 for smoothness)
        temp_pattern2_fine = np.cos(Y_fine_shifted * 0.5 - X_fine_shifted * 2) * 0.2 
        
        temp = temp_base + temp_pattern1_large + temp_pattern2_fine
        temp = (temp - temp.min()) / (temp.max() - temp.min()) # Local normalization for pattern visibility
        temp = temp * 0.8 + 0.1 # Scale to 0.1-0.9 range
        temp = temp * mask_np # Apply land mask to temperature

        if use_salinity:
            # Salinity pattern with different distinct waves
            sal_base = np.random.rand(img_h, img_w) * 0.1
            
            # Larger scale waveform (amplitude reverted to 0.4 for stability)
            sal_pattern1_large = np.cos(Y_shifted * 2.5 + X_shifted * 1.2) * 0.2 
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
    return normalized_data_tensor, land_mask_tensor