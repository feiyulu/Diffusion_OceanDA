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
SPEAR_exp ='M9'
T_varname = 'thetao_prior_z'  # 3D potential temperature
S_varname = 'so_prior_z'      # 3D practical salinity
SSH_varname = 'SSH_prior'     # 2D Sea Surface Height
SST_obs_varname = 'sst'       # Sea Surface Temperature (observation)
SSS_obs_varname = 'sss'       # Sea Surface Salinity (observation)

# --- Grid Loading and Preprocessing ---
# Load static grid information which includes coordinates and land masks.
# This model uses a tripolar grid, indicated by 'geolat'/'geolon' variables.
print("Loading and processing static grid information...")
ocean_static_ds = xr.open_dataset(f'{data_dir}/model_data/{SPEAR_exp}/ocean_z.static.nc')

# The model's native longitudes might be on a -180 to 180 range.
# We convert them to a 0 to 360 range for consistency.
ocean_static_ds['xh'] = ocean_static_ds['xh'].where(ocean_static_ds['xh'] > 0, ocean_static_ds['xh'] + 360)
ocean_static_ds['xq'] = ocean_static_ds['xq'].where(ocean_static_ds['xq'] > 0, ocean_static_ds['xq'] + 360)

# Roll the coordinates to shift the grid. This is often done to move the "dateline"
# or center the grid on a specific ocean basin (e.g., the Pacific).
ocean_static_ds_rolled = ocean_static_ds.roll(xh=60, roll_coords=True).roll(xq=60, roll_coords=True)
geolat = ocean_static_ds_rolled.geolat
geolon = ocean_static_ds_rolled.geolon
depth_ocean = ocean_static_ds_rolled.depth_ocean

# Load basin codes, which can be used for regional analysis (e.g., masking specific oceans).
# 1: Southern Ocean, 2: Atlantic, 3: Pacific, 4: Arctic, 5: Indian, 6: Mediterranean
basin_ds = xr.open_dataset(f'{data_dir}/model_data/M9/basin.nc')
basin_code = basin_ds['basin']

# --- Vertical Grid Definition ---
# Load the model's native vertical grid definition (layer thicknesses).
print("Defining vertical grid and coarsening scheme...")
model_dz_ds = xr.open_dataset(f'{data_dir}/model_data/{SPEAR_exp}/vgrid_75_2m.nc')
model_dz = model_dz_ds.dz  # Thickness of each fine layer

# Define the coarsening scheme. Here, we are combining every 3 fine layers into 1 coarse layer.
# `range(0, len(model_dz)+1, 3)` creates the boundaries for the new coarse layers.
combine_levels = range(0, len(model_dz) + 1, 3)

# Use the utility function from data_types to calculate the new coarse grid.
model_dz_coarse, model_z_coarse = data_types.depth_vertical_coarsen(model_dz, combine_levels)
print(f"Original vertical levels: {len(model_dz)}. Coarsened vertical levels: {len(model_dz_coarse)}.")

# --- Main Processing Loop for 3D Variables ---
# Loop through the specified year(s) to process the data.
variables_to_process = {
    # 'S': S_varname
}

for year in range(2003, 2012):
    for var_symbol, var_name in variables_to_process.items():
        print(f"\n--- Processing {var_symbol} for year: {year} ---")

        # Load the 3D data for the entire year.
        print(f"Loading raw 3D {var_symbol} data ({var_name})...")
        try:
            ds = xr.open_mfdataset(f'{data_dir}/model_data/{SPEAR_exp}/ocean_daily.{year}0101-{year}1231.{var_name}.nc')
        except FileNotFoundError:
            print(f"Warning: Data file for {var_name} in year {year} not found. Skipping.")
            continue
        
        # Apply the same longitude conversion and roll as the static grid file to ensure alignment.
        ds['xh'] = ds['xh'].where(ds['xh'] > 0, ds['xh'] + 360)
        ds_rolled = ds.roll(xh=60, roll_coords=True)
        data_var = ds_rolled[var_name]

        print(f"Applying vertical coarsening to {var_symbol}...")
        var_coarse = data_types.variable_vertical_coarsen(data_var, model_dz, combine_levels, depth_ocean)
        ds_coarse = xr.Dataset({var_symbol: var_coarse.astype('float32')})
        output_path = f'{data_dir}/model_data/{SPEAR_exp}/{var_symbol}.{year}.nc'
        print(f"Saving coarsened data to {output_path}...")
        ds_coarse.to_netcdf(output_path, unlimited_dims=["time"])
        print(f"Processing complete for {var_symbol} in {year}.")

# --- Main Processing Loop for 2D Variables ---
# This loop handles variables that only need horizontal rolling, not vertical coarsening.
variables_2d_to_process = {
    'SSH': SSH_varname
}

for year in range(2003, 2025):
    for var_symbol, var_name in variables_2d_to_process.items():
        print(f"\n--- Processing 2D variable {var_symbol} for year: {year} ---")

        # Load the 2D data for the entire year.
        print(f"Loading raw 2D {var_symbol} data ({var_name})...")
        try:
            ds = xr.open_dataset(f'{data_dir}/model_data/{SPEAR_exp}/ocean_daily.{year}0101-{year}1231.{var_name}.nc')
        except FileNotFoundError:
            print(f"Warning: Data file for {var_name} in year {year} not found. Skipping.")
            continue

        # Apply the same longitude conversion and roll as the static grid file.
        ds['xh'] = ds['xh'].where(ds['xh'] > 0, ds['xh'] + 360)
        ds_rolled = ds.roll(xh=60, roll_coords=True)
        
        output_path = f'{data_dir}/model_data/{SPEAR_exp}/{var_symbol}.{year}.nc'
        print(f"Saving rolled data to {output_path}...")
        # Save the entire rolled dataset, which now contains the correctly rolled variable.
        ds_rolled.to_netcdf(output_path, unlimited_dims=["time"])
        print(f"Processing complete for {var_symbol} in {year}.")
