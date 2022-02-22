import numpy as np


def haversine_vectorized(df,
                         start_lat="pickup_latitude",
                         start_lon="pickup_longitude",
                         end_lat="dropoff_latitude",
                         end_lon="dropoff_longitude"):
    """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).
        Vectorized version of the haversine distance for pandas df
        Computes distance in kms
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)),\
        np.radians(df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)),\
        np.radians(df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) *\
        np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())

def minkowski_distance_1(df, x1, x2, y1, y2):
    delta_x = df[x1] - df[x2]
    delta_y = df[y1] - df[y2]
    return ((abs(delta_x) ** 1) + (abs(delta_y)) ** 1) ** (1 / 1)

def minkowski_distance_2(df, x1, x2, y1, y2):
    delta_x = df[x1] - df[x2]
    delta_y = df[y1] - df[y2]
    return ((abs(delta_x) ** 2) + (abs(delta_y)) ** 2) ** (1 / 2)

def calculate_direction(df, d_lon, d_lat):
    result = np.zeros(len(df[d_lon]))
    l = np.sqrt(df[d_lon]**2 + df[d_lat]**2)
    result[df[d_lon]>0] = (180/np.pi)*np.arcsin(df[d_lat][df[d_lon]>0]/l[df[d_lon]>0])
    idx = (df[d_lon]<0) & (df[d_lat]>0)
    result[idx] = 180 - (180/np.pi)*np.arcsin(df[d_lat][idx]/l[idx])
    idx = (df[d_lon]<0) & (df[d_lat]<0)
    result[idx] = -180 - (180/np.pi)*np.arcsin(df[d_lat][idx]/l[idx])
    return result