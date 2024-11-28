import numpy as np
from geographiclib.geodesic import Geodesic

class SensorDataProcessor:
    def __init__(self, ref_lat, ref_lon):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.geod = Geodesic.WGS84

    def gnss_to_cartesian(self, latitude, longitude):
        """
        Converts GNSS latitude and longitude to Cartesian (x, y) coordinates.
        """
        g = self.geod.Inverse(self.ref_lat, self.ref_lon, latitude, longitude)
        x = g['s12'] * np.cos(np.radians(g['azi1']))
        y = g['s12'] * np.sin(np.radians(g['azi1']))
        return np.array([x, y])

    def process_imu(self, accel, gyro):
        """
        Processes IMU data (linear acceleration and angular velocity).
        """
        return np.array(accel), np.array(gyro)
