# --- model_vertical_coarsen.py ---
# This script performs offline preprocessing of 3D model output data.
# Its primary function is to take daily model data on a fine vertical grid,
# apply a vertical coarsening scheme to reduce the number of depth levels,
# and save the result as a new NetCDF file. This prepares the data for
# more efficient loading and use in 3D data assimilation experiments.

import xarray as xr
import data_types  # Assumes data_types.py is in the same directory or Python path
import numpy as np

# --- Configuration ---
# Define base directory for data and specific variable names.
data_dir = '/scratch/cimes/feiyul/Ocean_Data'
T_varname = 'thetao_prior_z'  # 3D potential temperature
S_varname = 'so_prior_z'      # 3D practical salinity
SST_obs_varname = 'sst'       # Sea Surface Temperature (observation)
SSS_obs_varname = 'sss'       # Sea Surface Salinity (observation)

# --- Grid Loading and Preprocessing ---
# Load static grid information which includes coordinates and land masks.
# This model uses a tripolar grid, indicated by 'geolat'/'geolon' variables.
print("Loading and processing static grid information...")
ocean_static_ds = xr.open_dataset(f'{data_dir}/model_data/M9/ocean_z.static.nc')

# The model's native longitudes might be on a -180 to 180 range.
# We convert them to a 0 to 360 range for consistency.
ocean_static_ds['xh'] = ocean_static_ds['xh'].where(ocean_static_ds['xh'] > 0, ocean_static_ds['xh'] + 360)
ocean_static_ds['xq'] = ocean_static_ds['xq'].where(ocean_static_ds['xq'] > 0, ocean_static_ds['xq'] + 360)

# Roll the coordinates to shift the grid. This is often done to move the "dateline"
# or center the grid on a specific ocean basin (e.g., the Pacific).
ocean_static_ds_rolled = ocean_static_ds.roll(xh=60, roll_coords=True).roll(xq=60, roll_coords=True)
geolat = ocean_static_ds_rolled.geolat
geolon = ocean_static_ds_rolled.geolon

# Load basin codes, which can be used for regional analysis (e.g., masking specific oceans).
# 1: Southern Ocean, 2: Atlantic, 3: Pacific, 4: Arctic, 5: Indian, 6: Mediterranean
basin_ds = xr.open_dataset(f'{data_dir}/model_data/M9/basin.nc')
basin_code = basin_ds['basin']

# --- Vertical Grid Definition ---
# Load the model's native vertical grid definition (layer thicknesses).
print("Defining vertical grid and coarsening scheme...")
model_dz_ds = xr.open_dataset(f'{data_dir}/model_data/M9/vgrid_75_2m.nc')
model_dz = model_dz_ds.dz  # Thickness of each fine layer

# Define the coarsening scheme. Here, we are combining every 3 fine layers into 1 coarse layer.
# `range(0, len(model_dz)+1, 3)` creates the boundaries for the new coarse layers.
combine_levels = range(0, len(model_dz) + 1, 3)

# Use the utility function from data_types to calculate the new coarse grid.
model_dz_coarse, model_z_coarse = data_types.depth_vertical_coarsen(model_dz, combine_levels)
print(f"Original vertical levels: {len(model_dz)}. Coarsened vertical levels: {len(model_dz_coarse)}.")

# --- Main Processing Loop ---
# Loop through the specified year(s) to process the data.
for year in range(2021, 2025):
    print(f"\nProcessing data for year: {year}...")

    # Load the 3D temperature data for the entire year.
    # Using open_mfdataset is good practice for handling multiple files per year if needed.
    print(f"Loading raw 3D temperature data...")
    T_ds = xr.open_mfdataset(f'{data_dir}/model_data/M9/ocean_daily.{year}0101-{year}1231.{T_varname}.nc')
    
    # Apply the same longitude conversion and roll as the static grid file to ensure alignment.
    T_ds['xh'] = T_ds['xh'].where(T_ds['xh'] > 0, T_ds['xh'] + 360)
    T_ds_rolled = T_ds.roll(xh=60, roll_coords=True)
    T_var = T_ds_rolled[T_varname]

    # Perform the vertical coarsening using the dedicated function.
    # This function calculates a weighted average of the fine layers to create the coarse layers.
    print("Applying vertical coarsening...")
    T_var_coarse = data_types.variable_vertical_coarsen(T_var, model_dz, combine_levels)

    # Create a new xarray Dataset with the coarsened variable.
    T_ds_coarse = xr.Dataset({'T': T_var_coarse.astype('float32')})

    # Save the processed, coarsened data to a new NetCDF file.
    output_path = f'{data_dir}/model_data/M9/T.{year}.nc'
    print(f"Saving coarsened data to {output_path}...")
    T_ds_coarse.to_netcdf(output_path)
    print("Processing complete for the year.")
