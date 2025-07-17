# --- argo_vertical_coarsen.py ---
# This script performs offline preprocessing of raw Argo float data.
# Its purpose is to read raw, high-resolution Argo profiles for specified years,
# interpolate them onto a consistent, coarsened vertical grid derived from a
# model, and save the processed data as new, analysis-ready NetCDF files.
# This is a one-time data preparation step.

import xarray as xr
import data_types  # Assumes data_types.py is in the same directory or Python path
import numpy as np

# --- Configuration ---
# Define the base directory where ocean data is stored.
data_dir = '/scratch/cimes/feiyul/Ocean_Data'
# data_dir = '/collab1/data_untrusted/Feiyu.Lu/Ocean_Data' # Alternative path

# --- 1. Define the Target Vertical Grid ---
# The goal is to interpolate sparse Argo observations onto a grid that is
# consistent with the model's vertical resolution.

print("Defining target vertical grid for interpolation...")
# Load the model's native vertical grid definition (layer thicknesses).
model_dz_ds = xr.open_dataset(f'{data_dir}/model_data/M9/vgrid_75_2m.nc')
model_dz = model_dz_ds.dz

# Define the coarsening scheme. Here, we combine every 3 fine layers into 1 coarse layer.
# `range(0, len(model_dz)+1, 3)` creates the boundaries for the new coarse layers.
combine_levels = range(0, len(model_dz) + 1, 3)

# Use the utility function from data_types to calculate the new coarse grid.
model_dz_coarse, model_z_coarse = data_types.depth_vertical_coarsen(model_dz, combine_levels)

# For Argo data, we are typically only interested in the upper ocean (e.g., < 2000m).
# We create a target grid specifically for Argo interpolation.
model_z_argo = model_z_coarse[model_z_coarse < 2000]
print(f"Target Argo interpolation grid defined with {len(model_z_argo)} levels up to {model_z_argo.max():.2f}m.")

# --- 2. Main Processing Loop ---
# Loop through the specified years to process the raw Argo data files.
# The loop runs in reverse chronological order.
print("\nStarting to process raw Argo data files...")
for year in range(2024, 2002, -1):
    print(f"--- Processing year: {year} ---")
    
    # Construct the path to the raw Argo data file for the current year.
    argo_raw_path = f'{data_dir}/obs_data/argo/argo_{year}.nc'
    
    try:
        # Load the raw Argo dataset for the year.
        argo_obs_ds = xr.open_dataset(argo_raw_path)
        
        # Use the `argo` class from our data_types library to handle the complex loading,
        # quality control, and interpolation of the profiles onto our target grid.
        print(f"Loading and interpolating profiles from {argo_raw_path}...")
        argo_obs_interp = data_types.argo(
            argo_obs_ds,
            interp=True,                 # Enable interpolation
            target_depth=model_z_argo,   # Provide the target grid
            year=year                    # Filter for the specific year
        )
        print(f"Found and processed {len(argo_obs_interp)} valid profiles for {year}.")

        # Convert the processed list of profile objects into a clean xarray Dataset.
        ds_processed = argo_obs_interp.convert_interpolated_to_dataset()

        # Define the output path for the new, processed NetCDF file.
        output_path = f'{data_dir}/obs_data/argo/argo_{year}_interp{len(combine_levels)-1}.nc'
        
        # Save the processed data.
        print(f"Saving processed data to {output_path}...")
        ds_processed.to_netcdf(output_path)
        
    except FileNotFoundError:
        print(f"Warning: Raw Argo data file not found for year {year}. Skipping.")
    except Exception as e:
        print(f"An error occurred while processing year {year}: {e}")

print("\nArgo data preprocessing complete.")