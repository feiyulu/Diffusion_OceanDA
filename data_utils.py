# --- data_utils.py ---
# This file handles the loading and preprocessing of ocean data from NetCDF files.
import numpy as np
import torch
import xarray as xr
import pandas as pd
import datetime
import cftime

def get_time_coordinates(config):
    """
    Scans the NetCDF files to get a list of all available time coordinates.
    This is used to determine the length of the dataset without loading everything into RAM.
    """
    filepath_t = config.filepath_t
    print(f"Scanning time coordinates from: {filepath_t}")
    # Use chunks={} to prevent loading data variables into memory during scan
    with xr.open_mfdataset(filepath_t, combine='by_coords', decode_cf=True, chunks={}) as ds:
        time_coords = ds.sel(time=slice(config.training_day_range[0], config.training_day_range[1])).time
        return time_coords[::config.training_day_interval].values

def load_single_ocean_slice(config, time_coord):
    """
    Loads and processes a single time slice of ocean data.
    This is called by the Dataset's __getitem__ method for lazy loading.
    """
    filepath_t = config.filepath_t
    filepath_s = config.filepath_s if config.use_salinity else None
    
    data_vars = []
    
    def _load_and_process_slice(filepaths, varname, time_coord, min_val, max_val):
        with xr.open_mfdataset(filepaths, combine='by_coords', decode_cf=True, chunks={'time': 1}) as ds:
            if not (config.varname_lat=='lat' and config.varname_lon=='lon'):
                ds = ds.rename({config.varname_lat:'lat', config.varname_lon:'lon'})
            
            # Select the single time slice and load it into memory
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
        if 'dayofyear' in config.conditioning_configs:
            # Use .timetuple().tm_yday to correctly get the day of the year from a cftime object
            day_of_year = time_coord.timetuple().tm_yday
            conditional_data['dayofyear'] = torch.tensor(day_of_year, dtype=torch.long)
        
        if 'co2' in config.conditioning_configs and config.co2_filepath:
            with xr.open_dataset(config.co2_filepath) as co2_ds:
                # Convert the cftime object to a standard pandas Timestamp
                # before using it for selection to avoid calendar mismatch errors.
                pandas_time = pd.to_datetime(str(time_coord))
                co2_da = co2_ds[config.co2_varname].sel(time=pandas_time, method='nearest').load()
                co2_normalized = (co2_da.values - config.co2_range[0]) / (config.co2_range[1] - config.co2_range[0])
                conditional_data['co2'] = torch.tensor(co2_normalized, dtype=torch.float32)

    return data_tensor, conditional_data

def load_static_data(config):
    """
    Loads the static (non-time-varying) data like the land mask and location embeddings.
    This data is small enough to be kept in memory for the entire training run.
    """
    print("Loading static data (mask and location embeddings)...")
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
            static_mask_np = np.expand_dims(static_mask_np, axis=0).repeat(config.data_shape[0], axis=0)

        land_mask_tensor = torch.tensor(static_mask_np[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32)
        
        location_field_tensor = None
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
            if "ocean_depth" in config.location_embedding_types:
                depth = static_ds['depth_ocean'].isel(lat=slice(config.lat_range[0], config.lat_range[1]), lon=slice(config.lon_range[0], config.lon_range[1])).values
                normalized_depth = (depth / 6000.0).astype(np.float32)
                location_channels.append(normalized_depth)
            
            if location_channels:
                location_field_single_sample = np.stack(location_channels, axis=0)
                location_field_tensor = torch.tensor(location_field_single_sample, dtype=torch.float32).unsqueeze(0)
                print(f"Location embedding created with shape: {location_field_tensor.shape}")

    return land_mask_tensor, location_field_tensor

# REFINED: Function to load a single slice of the test data for sampling
def load_test_ocean_slice(config, year, day_of_year):
    """
    Loads a single time slice of test data for a specific year and day of the year.
    """
    # Create a pandas Timestamp for easy calculation
    target_time_pd = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(day_of_year, unit='d')
    
    filepath_t = config.filepath_t_test
    filepath_s = config.filepath_s_test if config.use_salinity else None

    def _load_and_process_slice(filepaths, varname, target_time, min_val, max_val):
        with xr.open_mfdataset(filepaths, combine='by_coords', decode_cf=True, chunks={'time': 1}) as ds:
            if not (config.varname_lat=='lat' and config.varname_lon=='lon'):
                ds = ds.rename({config.varname_lat:'lat', config.varname_lon:'lon'})

            # 1. Get the calendar type from the dataset
            calendar = ds.time.encoding.get('calendar', 'standard')
            # 2. Create a cftime object for the target time using the dataset's calendar
            target_cftime = cftime.datetime(target_time.year, target_time.month, target_time.day, calendar=calendar)
            # 3. Manually find the index of the nearest time
            time_diffs = np.abs(ds.time - target_cftime)
            nearest_time_index = time_diffs.argmin().item()
            
            # 4. Select data using the integer index (.isel)
            da_sliced = ds[varname].isel(
                time=nearest_time_index,
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
            # Use the pandas object to get the day of the year
            conditional_data['dayofyear'] = torch.tensor(target_time_pd.dayofyear, dtype=torch.long)
        if 'co2' in config.conditioning_configs and config.co2_filepath:
            with xr.open_dataset(config.co2_filepath) as co2_ds:
                # The pandas object is suitable for the CO2 data which likely uses a standard calendar
                co2_da = co2_ds[config.co2_varname].sel(time=target_time_pd, method='nearest').load()
                co2_normalized = (co2_da.values - config.co2_range[0]) / (config.co2_range[1] - config.co2_range[0])
                conditional_data['co2'] = torch.tensor(co2_normalized, dtype=torch.float32)

    return data_tensor, conditional_data
