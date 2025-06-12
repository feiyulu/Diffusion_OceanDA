import numpy as np

class argo:
    def __init__(
            self,dataset,nread=None,meta_only=False,nstart=0,
            T_name='temp',S_name='salt',depth_name='depth',
            station_index='station_index',depth_index='depth_index',time_name='time',
            lon_name='longitude',lat_name='latitude',link_name='link'):
        """Initialize an Argo float dataset."""
        
        links_all=dataset[link_name]
        nobs = len(dataset[station_index])
        len_chunk = len(dataset[depth_index])

        dataset.load()

        i = nstart
        self.profiles = []
        nread = nobs if (nread is None or nread + nstart > nobs) else nread

        while (i < nread + nstart and i < nobs):
            depth = []
            T = []
            S = []
            lat = dataset[lat_name][i].values
            lon = dataset[lon_name][i].values
            time = dataset[time_name][i].values

            depth=dataset[depth_name][i,0:len_chunk].data
            if meta_only:
                T=[]
                S=[]
            else:
                T=dataset[T_name][i,0:len_chunk].data
                S=dataset[S_name][i,0:len_chunk].data

            # link chuncked profiles together
            while (links_all[i]):
                if links_all[i+1]:
                    depth=np.concatenate([depth,dataset[depth_name][i+1,0:len_chunk].data])
                    if not meta_only:
                        T=np.concatenate([T,dataset[T_name][i+1,0:len_chunk].data])
                        S=np.concatenate([S,dataset[S_name][i+1,0:len_chunk].data])
                else:
                    depth=np.concatenate(
                        [depth,dataset[depth_name][i+1,0:len_chunk].dropna(dim=depth_index).data])
                    if not meta_only:
                        T=np.concatenate(
                            [T,dataset[T_name][i+1,0:len_chunk].dropna(dim=depth_index).data])
                        S=np.concatenate(
                            [S,dataset[S_name][i+1,0:len_chunk].dropna(dim=depth_index).data])
                i += 1

            # Need to subsample(thinning) the profiles with too many levels
            if depth[0]<100:
                # if len(depth) > 100 or (depth[-1]-depth[0])/len(depth):
                #     depth_new,T_new,S_new = argo_vertical_coarsen(depth,T,S,model_dz)
                self.profiles.append(argo_float(time,lat,lon,depth,T,S,meta_only))
            i += 1

        self.min_depth = np.array([profile.min_depth for profile in self.profiles])
        self.max_depth = np.array([profile.max_depth for profile in self.profiles])
        self.levels = np.array([profile.levels for profile in self.profiles])
        self.lat = np.array([profile.lat for profile in self.profiles])
        self.lon = np.array([profile.lon for profile in self.profiles])
        self.time = np.array([profile.time for profile in self.profiles])

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

# combine_levels = [0,6,12,18,23,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75]
def model_vertical_coarsen(T,dz,combine_levels,z_name='z'):
    if len(T[z_name]) != len(dz):
        raise Exception('vertical levels not matched')
        
def argo_vertical_coarsen(depth,T,S,model_dz):
    return depth,T,S