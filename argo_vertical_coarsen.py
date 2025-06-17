import xarray as xr
import data_types
import numpy as np

data_dir = '/scratch/cimes/feiyul/Ocean_Data'
# data_dir = '/collab1/data_untrusted/Feiyu.Lu/Ocean_Data'

model_dz_ds = xr.open_dataset(f'{data_dir}/model_data/M9/vgrid_75_2m.nc')
model_dz = model_dz_ds.dz
model_zi = np.zeros(len(model_dz)+1)
for i in range(1,len(model_dz)+1):
    model_zi[i] = model_dz[0:i].sum().values

combine_levels = range(0,len(model_dz)+1,3)
model_dz_coarse, model_z_coarse = data_types.depth_vertical_coarsen(model_dz,combine_levels)
model_z_argo = model_z_coarse[model_z_coarse<2000]
print(model_z_argo)

for year in range(2024,2002,-1):
    argo_obs_ds = xr.open_dataset(f'{data_dir}/obs_data/argo/argo_{year}.nc')
    argo_obs_interp = data_types.argo(argo_obs_ds,interp=True,target_depth=model_z_argo,year=year)
    print(year,len(argo_obs_interp))

    ds=argo_obs_interp.convert_interpolated_to_dataset()
    ds.to_netcdf(f'{data_dir}/obs_data/argo/argo_{year}_interp{len(combine_levels)-1}.nc')