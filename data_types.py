# --- data_types.py ---
# This file defines classes and functions for handling and processing
# oceanographic data, with a focus on Argo float profiles.
import numpy as np
import xarray as xr
import scipy.interpolate as interpolate
from typing import List, Tuple, Optional

class argo_float:
    """Class to hold a single Argo float profile."""
    def __init__(self, time: np.datetime64, lat: float, lon: float, depth: np.ndarray, 
                 T: np.ndarray, S: np.ndarray, meta_only: bool = False):
        """
        Initializes an Argo float profile.

        Args:
            time (np.datetime64): Timestamp of the profile.
            lat (float): Latitude of the profile.
            lon (float): Longitude of the profile.
            depth (np.ndarray): Array of depth levels.
            T (np.ndarray): Array of temperature values.
            S (np.ndarray): Array of salinity values.
            meta_only (bool): If True, only metadata is stored.
        """
        if not meta_only and (len(depth) != len(T) or len(depth) != len(S)):
            raise ValueError('Argo profile data arrays must have the same length as depth array.')
        
        self.time = time
        self.lat = lat
        self.lon = lon
        self.min_depth = depth[0] if len(depth) > 0 else np.nan
        self.max_depth = depth[-1] if len(depth) > 0 else np.nan
        self.levels = len(depth)
        
        if not meta_only:
            self.depth = depth
            self.T = T
            self.S = S

    def get_levels(self) -> int:
        """Returns the number of depth levels in the profile."""
        return self.levels

class argo:
    """A collection of Argo float profiles, with methods for loading, filtering, and processing."""
    def __init__(
            self, dataset: xr.Dataset, nread: Optional[int] = None, 
            meta_only: bool = False, nstart: int = 0,
            T_name: str = 'temp', S_name: str = 'salt', 
            depth_name: str = 'depth', station_index: str = 'station_index', 
            depth_index: str = 'depth_index', time_name: str = 'time',
            lon_name: str = 'longitude', lat_name: str = 'latitude', link_name: str = 'link',
            interp: bool = False, 
            target_depth: Optional[np.ndarray] = None, year: Optional[int] = None):
        """
        Initializes an Argo float dataset from an xarray Dataset.

        Args:
            dataset (xr.Dataset): The input dataset containing Argo data.
            nread (int, optional): The number of profiles to read. Defaults to all.
            meta_only (bool): If True, load only metadata, not T/S/depth values.
            nstart (int): The starting index of profiles to read.
            interp (bool): If True, interpolate profiles to a target depth grid.
            target_depth (np.ndarray, optional): The target depth grid for interpolation.
            year (int, optional): If provided, only load profiles from this year.
        """
        links_all = dataset[link_name].values
        nobs = len(dataset[station_index])
        len_chunk = len(dataset[depth_index])

        dataset.load()

        self.interp = interp
        if self.interp:
            if target_depth is None:
                raise ValueError("target_depth must be provided if interp is True.")
            self.model_z = target_depth[target_depth < 2000]
            self.model_levels = len(self.model_z)

        time_start = np.datetime64(f'{year}-01-01') if year else np.datetime64('1900-01-01')
        time_end = np.datetime64(f'{year+1}-01-01') if year else np.datetime64('2100-01-01')
            
        i = nstart
        self.profiles: List[argo_float] = []
        nread = nobs if (nread is None or nread + nstart > nobs) else nread

        while (i < nread + nstart and i < nobs):
            time = dataset[time_name][i].values
            if time_start <= time < time_end:
                lat = dataset[lat_name][i].values
                lon = dataset[lon_name][i].values

                depth_vals = dataset[depth_name][i, :len_chunk].data
                T_vals = [] if meta_only else dataset[T_name][i, :len_chunk].data
                S_vals = [] if meta_only else dataset[S_name][i, :len_chunk].data

                # Follow linked profiles to handle chunked data
                current_link = i
                while links_all[current_link]:
                    next_idx = current_link + 1
                    depth_vals = np.concatenate([depth_vals, dataset[depth_name][next_idx, :len_chunk].data])
                    if not meta_only:
                        T_vals = np.concatenate([T_vals, dataset[T_name][next_idx, :len_chunk].data])
                        S_vals = np.concatenate([S_vals, dataset[S_name][next_idx, :len_chunk].data])
                    current_link = next_idx
                
                i = current_link # Move main index past the linked profiles

                # Quality control and filtering
                valid_mask = (depth_vals > 0) & (depth_vals < 6500) & \
                    ~np.isnan(T_vals) & ~np.isnan(S_vals)
                depth_clean, T_clean, S_clean = \
                    depth_vals[valid_mask], T_vals[valid_mask], S_vals[valid_mask]

                if len(depth_clean) > 5 and (depth_clean[0] < 30 or depth_clean[-1] < 2000):
                    if self.interp:
                        T_new, S_new = argo_vertical_coarsen(depth_clean, T_clean, S_clean, self.model_z)
                        self.profiles.append(argo_float(time, lat, lon, self.model_z, T_new, S_new, meta_only))
                    else:
                        self.profiles.append(argo_float(time, lat, lon, depth_clean, T_clean, S_clean, meta_only))
            else:
                # Skip profile and its linked parts if outside time range
                current_link = i
                while links_all[current_link]:
                    current_link += 1
                i = current_link

            i += 1

        # Pre-calculate summary statistics for faster filtering
        self.min_depth = np.array([p.min_depth for p in self.profiles])
        self.max_depth = np.array([p.max_depth for p in self.profiles])
        self.levels = np.array([p.levels for p in self.profiles])
        self.lat = np.array([p.lat for p in self.profiles])
        self.lon = np.array([p.lon for p in self.profiles])
        self.time = np.array([p.time for p in self.profiles])

    def convert_interpolated_to_dataset(self) -> xr.Dataset:
        """Converts the collection of interpolated profiles into a single xarray Dataset."""
        if not self.interp:
            raise Exception("This Argo dataset is not interpolated and cannot be converted.")

        nprofiles = len(self)
        T_data = np.full((nprofiles, self.model_levels), np.nan, dtype=np.float32)
        S_data = np.full((nprofiles, self.model_levels), np.nan, dtype=np.float32)

        for i, profile in enumerate(self.profiles):
            T_data[i, :] = profile.T
            S_data[i, :] = profile.S

        ds = xr.Dataset(
            {
                "T": (("profile", "z"), T_data),
                "S": (("profile", "z"), S_data),
                "lat": (("profile",), self.lat),
                "lon": (("profile",), self.lon),
                "time": (("profile",), self.time),
            },
            coords={"profile": np.arange(nprofiles), "z": self.model_z},
        )
        return ds

    def __len__(self) -> int:
        return len(self.profiles)

    def get_profiles_by_all(self, start_time, end_time,
                            lat_min=-90, lat_max=90, 
                            lon_min=0., lon_max=360.,
                            depth_min=0., depth_max=10000.):
        """Get profiles by time, region, and depth."""
        return [i for i, (time, lat, lon, depth) in 
                enumerate(zip(self.time, self.lat, self.lon, self.max_depth))
                if start_time <= time <= end_time and
                lat_min <= lat <= lat_max and
                lon_min <= lon <= lon_max and
                depth_min <= depth <= depth_max]

def argo_vertical_coarsen(depth: np.ndarray, T: np.ndarray, S: np.ndarray, target_depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates Argo T/S profiles to a target vertical grid using cubic splines.

    Args:
        depth (np.ndarray): Original depth levels of the Argo profile.
        T (np.ndarray): Original temperature values.
        S (np.ndarray): Original salinity values.
        target_depth (np.ndarray): The target depth grid for interpolation.

    Returns:
        A tuple containing (interpolated Temperature, interpolated Salinity).
    """
    # Use cubic spline interpolation for smoother results. Extrapolate=False returns NaNs outside the original depth range.
    spl_T = interpolate.CubicSpline(depth, T, extrapolate=False)
    T_target = spl_T(target_depth)
    spl_S = interpolate.CubicSpline(depth, S, extrapolate=False)
    S_target = spl_S(target_depth)

    return T_target.astype(np.float32), S_target.astype(np.float32)

def depth_vertical_coarsen(model_dz: xr.DataArray, combine_levels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarsens a model's vertical grid definition.

    Args:
        model_dz (xr.DataArray): DataArray of layer thicknesses.
        combine_levels (List[int]): List of indices defining the boundaries of the new coarse layers.

    Returns:
        A tuple containing (coarse layer thicknesses, coarse layer center depths).
    """
    if combine_levels[-1] > len(model_dz):
        raise ValueError('Combine levels exceed depth levels')

    model_zi = np.zeros(len(model_dz) + 1)
    model_zi[1:] = np.cumsum(model_dz.values)

    model_zi_coarse = model_zi[combine_levels]
    model_dz_coarse = np.diff(model_zi_coarse)
    model_z_coarse = model_zi_coarse[:-1] + model_dz_coarse / 2

    return model_dz_coarse, model_z_coarse

def variable_vertical_coarsen(
    Var: xr.DataArray,
    model_dz: xr.DataArray,
    combine_levels: List[int],
    ocean_depth: Optional[xr.DataArray] = None,
    time_name: str = 'time',
    x_name: str = 'xh',
    y_name: str = 'yh',
    z_name: str = 'z'
) -> xr.DataArray:
    """
    Coarsens a data variable on a model's vertical grid by performing a weighted average.
    If ocean_depth is provided, it masks levels that are below the sea floor with NaNs.
    """
    if combine_levels[-1] > len(model_dz):
        raise ValueError('Combine levels exceed depth levels')
    if len(Var[z_name]) != len(model_dz):
        raise ValueError('Variable vertical levels do not match model_dz levels')

    model_dz_coarse, model_z_coarse = depth_vertical_coarsen(model_dz, combine_levels)

    coarse_shape = (len(Var[time_name]), len(model_dz_coarse), len(Var[y_name]), len(Var[x_name]))
    Var_data_coarse = np.full(coarse_shape, np.nan, dtype=np.float32)
    
    # Pre-calculate the weighted average for all cells first.
    for i in range(len(model_dz_coarse)):
        var_slice = Var[:, combine_levels[i]:combine_levels[i+1], :, :]
        dz_slice = model_dz[combine_levels[i]:combine_levels[i+1]]
        
        var_sum = (var_slice * dz_slice.data[None, :, None, None]).sum(dim=z_name, skipna=True)
        # Calculate weights only for non-NaN cells in the variable slice
        weights_sum = dz_slice.sum()
        weights_sum = weights_sum.where(weights_sum > 0)
        
        Var_data_coarse[:, i, :, :] = (var_sum / weights_sum).values

    Var_coarse = xr.DataArray(
        Var_data_coarse,
        coords={
            time_name: Var[time_name],
            z_name: model_z_coarse,
            y_name: Var[y_name],
            x_name: Var[x_name]
        },
        dims=Var.dims
    ).astype('float32')

    # If ocean_depth is provided, create a mask based on fine-level depths.
    if ocean_depth is not None:
        # Get the depth of the bottom of each fine layer
        model_zi_fine = np.cumsum(model_dz.values)
        
        # Create a 3D mask for the original fine grid. True if cell is below the ocean floor.
        fine_grid_is_bottom = model_zi_fine[:, None, None] > ocean_depth.values[None, :, :]

        # Loop through the coarse levels to create the final mask
        for i in range(len(model_dz_coarse)):
            # Check if ANY fine level within this coarse bin hits the bottom
            fine_levels_in_coarse_bin = fine_grid_is_bottom[combine_levels[i]:combine_levels[i+1], :, :]
            coarse_cell_hits_bottom = np.any(fine_levels_in_coarse_bin, axis=0) # Shape: (y, x)

            # Where the coarse cell hits the bottom, set its value to NaN
            Var_coarse[:, i, :, :] = Var_coarse[:, i, :, :].where(~coarse_cell_hits_bottom)

    return Var_coarse