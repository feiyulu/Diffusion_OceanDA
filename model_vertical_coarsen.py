import xarray as xr
import data_types
import numpy as np

data_dir = '/scratch/cimes/feiyul/Ocean_Data'
# data_dir = '/collab1/data_untrusted/Feiyu.Lu/Ocean_Data'
T_varname = 'thetao_prior_z'
S_varname = 'so_prior_z'
SST_obs_varname = 'sst'
SSS_obs_varname = 'sss'

# Static grid information
# Tripolar grid is indicated by geolat/geolon
ocean_static_ds = xr.open_dataset(f'{data_dir}/model_data/M9/ocean_z.static.nc')
ocean_static_ds['xh']=ocean_static_ds['xh'].where(ocean_static_ds['xh']>0,ocean_static_ds['xh']+360)
ocean_static_ds['xq']=ocean_static_ds['xq'].where(ocean_static_ds['xq']>0,ocean_static_ds['xq']+360)
ocean_static_ds_rolled = ocean_static_ds.roll(xh=60,roll_coords=True).roll(xq=60,roll_coords=True)
geolat= ocean_static_ds_rolled.geolat
geolon= ocean_static_ds_rolled.geolon

# Basin code in addition to land mask from static file
# 1: SO, 2: Atlantic, 3: Pacific, 4: Arctic, 5: Indian, 6: Med, 7...
basin_ds = xr.open_dataset(f'{data_dir}/model_data/M9/basin.nc')
basin_code = basin_ds['basin']

model_dz_ds = xr.open_dataset(f'{data_dir}/model_data/M9/vgrid_75_2m.nc')
model_dz = model_dz_ds.dz
model_z = np.zeros(len(model_dz))
model_z[0] = model_dz[0]/2 
for i in range(1,len(model_dz)):
    model_z[i] = model_z[i-1] + (model_dz[i-1]+model_dz[i])/2
model_zi = np.zeros(len(model_dz)+1)
for i in range(1,len(model_dz)+1):
    model_zi[i] = model_dz[0:i].sum().values

combine_levels = range(0,len(model_dz)+1,3)
model_dz_coarse, model_z_coarse = data_types.depth_vertical_coarsen(model_dz,combine_levels)

for year in range(2020,2021):
    T_ds = xr.open_mfdataset(f'{data_dir}/model_data/M9/ocean_daily.{year}0101-{year}1231.{T_varname}.nc')
    T_ds['xh']=T_ds['xh'].where(T_ds['xh']>0,T_ds['xh']+360)
    T_ds_rolled = T_ds.roll(xh=60,roll_coords=True)
    T_var = T_ds_rolled[T_varname]

    T_var_coarse = data_types.variable_vertical_coarsen(T_var,model_dz,combine_levels)

    T_ds_coarse = xr.Dataset(dict(T=T_var_coarse.astype('float32')))
    T_ds_coarse.to_netcdf(f'{data_dir}/model_data/M9/T.{year}.nc')