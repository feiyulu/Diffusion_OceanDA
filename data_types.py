import numpy as np
import xarray as xr
import scipy.interpolate as interpolate

class argo:
    def __init__(
            self,dataset,nread=None,meta_only=False,nstart=0,
            T_name='temp',S_name='salt',depth_name='depth',
            station_index='station_index',depth_index='depth_index',time_name='time',
            lon_name='longitude',lat_name='latitude',link_name='link',
            interp=False,target_depth=None,year=None):
        """Initialize an Argo float dataset."""
        
        links_all=dataset[link_name]
        nobs = len(dataset[station_index])
        len_chunk = len(dataset[depth_index])

        dataset.load()

        if interp:
            model_z = target_depth[target_depth<2000]
            self.model_z = model_z
            self.model_levels = len(model_z)
            self.interp=interp

        if year:
            time_start = np.datetime64(f'{year}-01-01T00:00:00')
            time_end = np.datetime64(f'{year+1}-01-01T00:00:00')
        else:
            time_start = np.datetime64('1900-01-01T00:00:00')
            time_end = np.datetime64('2100-01-01T00:00:00')
            
        i = nstart
        self.profiles = []
        nread = nobs if (nread is None or nread + nstart > nobs) else nread

        while (i < nread + nstart and i < nobs):
            time = dataset[time_name][i].values
            if (time_start < time < time_end):
                depth = []
                T = []
                S = []
                lat = dataset[lat_name][i].values
                lon = dataset[lon_name][i].values

                depth=dataset[depth_name][i,0:len_chunk].data
                if meta_only:
                    T=[]
                    S=[]
                else:
                    T=dataset[T_name][i,0:len_chunk].data
                    S=dataset[S_name][i,0:len_chunk].data

                ### link chuncked profiles together
                while (links_all[i]):
                    # if links_all[i+1]:
                    depth=np.concatenate([depth,dataset[depth_name][i+1,0:len_chunk].data])
                    if not meta_only:
                        T=np.concatenate([T,dataset[T_name][i+1,0:len_chunk].data])
                        S=np.concatenate([S,dataset[S_name][i+1,0:len_chunk].data])
                    i += 1

                if (len(depth)!=len(T) or len(depth)!=len(S)):
                    raise Exception('Argo profile levels do not match')

                ### Remove depths outside of the 0-6500m range (missing/fill data problem)
                depth_mask=depth[(depth>0) & (depth<6500) & ~np.isnan(T) & ~np.isnan(S)]
                T_mask=T[(depth>0) & (depth<6500) & ~np.isnan(T) & ~np.isnan(S)]
                S_mask=S[(depth>0) & (depth<6500) & ~np.isnan(T) & ~np.isnan(S)]

                ### Remove DeepArgo profiles and profiles that didn't reach close to surface
                if len(depth_mask)>5:
                    if depth_mask[0]<30 or depth_mask[-1]<2000:
                        ### Need to subsample(thinning) the profiles with too many levels
                        if interp:
                            T_new,S_new = argo_vertical_coarsen(depth_mask,T_mask,S_mask,model_z)
                            self.profiles.append(argo_float(time,lat,lon,model_z,T_new,S_new,meta_only))
                        else:
                            self.profiles.append(argo_float(time,lat,lon,depth_mask,T_mask,S_mask,meta_only))
            else:
                while (links_all[i]):
                    i += 1

            i += 1

        ### Get some summary statistics for easy diagnostics
        self.min_depth = np.array([profile.min_depth for profile in self.profiles])
        self.max_depth = np.array([profile.max_depth for profile in self.profiles])
        self.levels = np.array([profile.levels for profile in self.profiles])
        self.lat = np.array([profile.lat for profile in self.profiles])
        self.lon = np.array([profile.lon for profile in self.profiles])
        self.time = np.array([profile.time for profile in self.profiles])

    def convert_interpolated_to_dataset(self):
        if not self.interp:
            raise Exception("This Argo dataset is not interpolated")

        nprofiles = len(self)
        profile_index = range(nprofiles)
        levels = self.model_levels
        z = self.model_z

        T_data = np.empty((nprofiles,levels))
        S_data = np.empty((nprofiles,levels))

        for i in range(nprofiles):
            T_data[i,:] = self.profiles[i].T
            S_data[i,:] = self.profiles[i].S

        T_var = xr.DataArray(T_data,coords=[profile_index,z],dims=['profile','z'])
        S_var = xr.DataArray(S_data,coords=[profile_index,z],dims=['profile','z'])

        ds = xr.Dataset(dict(T=T_var,S=S_var))

        return ds

    def __len__(self):
        return len(self.profiles)
    
    def get_profiles_by_period(self,start_time,end_time):
        """Get profiles by time period."""
        return [i for i,time in enumerate(self.time) if start_time <= time <= end_time]

    def get_profiles_by_region(self,lat_min,lat_max,lon_min,lon_max):
        """Get profiles by geographic region."""
        return [i for i,(lat,lon) in enumerate(zip(self.lat,self.lon)) 
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max]
    
    def get_profiles_by_max_depth(self, depth_min, depth_max):
        """Get profiles by maximum depth."""
        return [i for i, depth in enumerate(self.max_depth)
                if depth_min <= depth <= depth_max]

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
    
class argo_float:
    """Class to hold a single Argo float profile."""
    def __init__(self,time,lat,lon,depth,T,S,meta_only=False):
        """Initialize an Argo float profile."""
        if (len(depth)!=len(T) or len(depth)!=len(S)):
            raise Exception('Argo profile levels do not match')
        self.time = time
        self.lat = lat
        self.lon = lon
        self.min_depth = depth[0]
        self.max_depth = depth[-1]
        self.levels = len(depth)
        if not meta_only:
            self.depth = depth
            self.T = T
            self.S = S

    def get_levels(self):
        return len(self.depth)

def depth_vertical_coarsen(model_dz,combine_levels):

    if combine_levels[-1] > len(model_dz):
        raise Exception('Combine levels exceed depth levels')

    model_zi = np.zeros(len(model_dz)+1)
    for i in range(1,len(model_dz)+1):
        model_zi[i] = model_dz[0:i].sum().values

    model_z_coarse = np.zeros(len(combine_levels)-1)
    model_dz_coarse = np.zeros(len(combine_levels)-1)
    model_zi_coarse = np.zeros(len(combine_levels))
    for i in range(1,len(combine_levels)):
        model_dz_coarse[i-1] = model_dz[combine_levels[i-1]:combine_levels[i]].sum().data
        model_zi_coarse[i] = model_zi[combine_levels[i]]
        model_z_coarse[i-1] = (model_zi_coarse[i]+model_zi_coarse[i-1])/2

    return model_dz_coarse,model_z_coarse

def variable_vertical_coarsen(Var,model_dz,combine_levels,time_name='time',x_name='xh',y_name='yh',z_name='z'):

    if combine_levels[-1] > len(model_dz):
        raise Exception('Combine levels exceed depth levels')
    if len(Var[z_name]) != len(model_dz):
        raise Exception('vertical levels not matched')
 
    model_z = np.zeros(len(model_dz))
    model_z[0] = model_dz[0]/2 
    for i in range(1,len(model_dz)):
        model_z[i] = model_z[i-1] + (model_dz[i-1]+model_dz[i])/2
    model_zi = np.zeros(len(model_dz)+1)
    for i in range(1,len(model_dz)+1):
        model_zi[i] = model_dz[0:i].sum().values
  
    model_z_coarse = np.zeros(len(combine_levels)-1)
    model_dz_coarse = np.zeros(len(combine_levels)-1)
    model_zi_coarse = np.zeros(len(combine_levels))
    for i in range(1,len(combine_levels)):
        model_dz_coarse[i-1] = model_dz[combine_levels[i-1]:combine_levels[i]].sum().data
        model_zi_coarse[i] = model_zi[combine_levels[i]]
        model_z_coarse[i-1] = (model_zi_coarse[i]+model_zi_coarse[i-1])/2

    Var_data_coarse = np.empty((len(Var[time_name]),len(model_dz_coarse),len(Var[y_name]),len(Var[x_name])))
    Var_weighted = Var*model_dz.data[None,:,None,None]
    for i in range(len(model_dz_coarse)):
        Var_data_coarse[:,i,:,:] = \
            Var_weighted[:,combine_levels[i]:combine_levels[i+1],:,:].sum(dim=z_name,skipna=True)/\
            model_dz_coarse[i]

    Var_coarse = xr.DataArray(
        Var_data_coarse,
        coords=[Var[time_name],model_z_coarse,Var[y_name],Var[x_name]],
        dims=Var.dims).astype('float32')

    return Var_coarse

def argo_vertical_coarsen(depth,T,S,target_depth):

    T_target = np.empty(len(target_depth))
    S_target = np.empty(len(target_depth))

    # T_target = np.interp(target_depth,depth,T,left=np.nan,right=np.nan)
    # S_target = np.interp(target_depth,depth,S,left=np.nan,right=np.nan)

    spl_T = interpolate.CubicSpline(depth,T,extrapolate=False)
    T_target = spl_T(target_depth)
    spl_S = interpolate.CubicSpline(depth,S,extrapolate=False)
    S_target = spl_S(target_depth)

    return T_target, S_target