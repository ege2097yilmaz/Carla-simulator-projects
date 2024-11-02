import carla, time
import numpy as np
import open3d as o3d

"""
to get pointcloud data
"""
class PointCloudData:
    def __init__(self, host='localhost', port=2000, lidar_bp_name='sensor.lidar.ray_cast'):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.lidar_bp_name = lidar_bp_name
        self.lidar_sensor = None
        self.point_cloud = []

    def setup_lidar_sensor(self, vehicle):
        # Get the blueprint for the LIDAR sensor
        blueprint_library = self.world.get_blueprint_library()
        lidar_bp = blueprint_library.find(self.lidar_bp_name)

        # Set LIDAR sensor attributes
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('rotation_frequency', '90')
        lidar_bp.set_attribute('channels', '128')
        lidar_bp.set_attribute('points_per_second', '560000')

        # Create the LIDAR sensor and attach it to the vehicle
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        # Listen for point cloud data
        self.lidar_sensor.listen(lambda data: self.process_lidar_data(data))

    def process_lidar_data(self, data):
        # Convert LIDAR data to a numpy array
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        self.point_cloud = points

    def get_open3d_point_cloud(self):
        # Convert the point cloud to Open3D format for visualization
        if len(self.point_cloud) == 0:
            return None

        # Extract x, y, z coordinates
        points = self.point_cloud[:, :3]
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
        return point_cloud_o3d

    def destroy_sensor(self):
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
