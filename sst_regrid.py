# --- sst_regrid.py ---
# This script performs an offline regridding of SST data from a regular
# 1x1 degree grid to the model's native irregular grid. This is a one-time
# preprocessing step to make the observation data directly compatible with
# the model's state for faster loading during data assimilation experiments.

import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
import os

# --- Configuration ---
# Define base directory and file paths. Adjust these as needed.
data_dir = '/scratch/cimes/feiyul/Ocean_Data'
model_static_path = f'{data_dir}/model_data/M9/ocean_z.static.nc'

# Define the range of years to process
start_year = 2003
end_year = 2023 # Set the range for all years you need

# --- 1. Load Model's Target Grid ---
print("Loading model's target grid definition...")
with xr.open_dataset(model_static_path) as static_ds:
    # These are the target coordinates we want to interpolate onto.
    target_lat = static_ds['geolat'].values
    target_lon = static_ds['geolon'].values
    # We need the original coordinate dimensions for creating the new DataArray
    model_dims = static_ds['geolat'].dims
    model_coords = {'lat': static_ds['yh'], 'lon': static_ds['xh']}

print(f"Target grid shape: {target_lat.shape}")

# --- 2. Main Processing Loop ---
print(f"\nStarting to regrid SST data from {start_year} to {end_year}...")
for year in range(start_year, end_year + 1):
    print(f"--- Processing year: {year} ---")
    
    sst_raw_path = f'{data_dir}/obs_data/sst/sst.day.{year}.1x1.nc'
    output_path = f'{data_dir}/obs_data/sst/sst.day.{year}.regridded.nc'

    if not os.path.exists(sst_raw_path):
        print(f"Warning: Raw SST file not found, skipping: {sst_raw_path}")
        continue

    with xr.open_dataset(sst_raw_path) as sst_ds:
        # Prepare source grid and data
        source_lat = sst_ds['lat'].values
        source_lon = sst_ds['lon'].values
        source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat)
        
        # Flatten the source grid for griddata
        source_points = np.vstack((source_lat_grid.ravel(), source_lon_grid.ravel())).T
        
        regridded_sst_list = []
        # Iterate through each day in the SST file
        for t_idx in tqdm(range(len(sst_ds['time'])), desc=f"Regridding {year}"):
            sst_slice = sst_ds['sst'].isel(time=t_idx).values
            
            # Use scipy.interpolate.griddata for regridding. 'nearest' is robust.
            regridded_data = griddata(source_points, sst_slice.ravel(), (target_lat, target_lon), method='nearest')
            regridded_sst_list.append(regridded_data)

        # Create a new xarray.DataArray with the regridded data
        regridded_da = xr.DataArray(
            data=np.array(regridded_sst_list),
            dims=('time', *model_dims),
            coords={'time': sst_ds['time'], **model_coords},
            name='sst'
        )
        
        print(f"Saving regridded data to {output_path}...")
        regridded_da.to_netcdf(output_path)

print("\nSST regridding complete.")