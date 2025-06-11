class argo:
    def __init__(
            self,dataset,T_name='temp',S_name='salt',depth_name='depth',
            station_name='station_index',time_name='time',
            lon_name='longitude',lat_name='latitude',link_name='link'):
        links_all=dataset[link_name]
        nobs = len(dataset[station_name])
    
        i = 0

        while (i < nobs):
            depth = []
            T = []
            S = []
            lat = dataset[lat_name][i]
            lon = dataset[lon_name][i]
            time = dataset[time_name][i]
            while (links_all[i]):
                depth.append(dataset[depth_name][i,:])
                T.append(dataset[T_name][i,:])
                S.append(dataset[S_name][i,:])
            

class argo_float:
    def __init__(self,time,lat,lon,depth,T,S):
        self.time = time
        self.lat = lat
        self.lon = lon
        self.depth = depth
        self.T = T
        self.S = S