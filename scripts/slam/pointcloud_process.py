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
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('rotation_frequency', '180')
        lidar_bp.set_attribute('channels', '128')
        lidar_bp.set_attribute('points_per_second', '1560000')

        # Create the LIDAR sensor and attach it to the vehicle
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        # Listen for point cloud data
        self.lidar_sensor.listen(lambda data: self.process_lidar_data(data))

    def process_lidar_data(self, data):
        # Convert LIDAR data to a numpy array
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        self.point_cloud = points

    # def get_open3d_point_cloud(self, distance_threshold):
    #     # Convert the point cloud to Open3D format for visualization
    #     if len(self.point_cloud) == 0:
    #         return None

    #     # Extract x, y, z coordinates
    #     points = self.point_cloud[:, :3]
    #     point_cloud_o3d = o3d.geometry.PointCloud()
    #     point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    #     return point_cloud_o3d
        

    def get_open3d_point_cloud(self, distance_threshold):
        # Check if the point cloud is empty
        if len(self.point_cloud) == 0:
            return None

        # Extract x, y, z coordinates
        points = self.point_cloud[:, :3]

        # Filter the points based on the distance threshold in the z-axis
        # for ground segmentation
        filtered_points = points[points[:, 2] > distance_threshold]

        # Create Open3D point cloud with the filtered points
        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_points)

        return point_cloud_o3d

    def destroy_sensor(self):
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()

    
    def visualize_pose_in_carla(self, world, pose, color=(255, 0, 0)):
        """
        Visualizes a pose in CARLA by adding an arrow marker at the pose location.
        Args:
            world (carla.World): The CARLA world object.
            pose (numpy.ndarray): The pose as a 4x4 transformation matrix.
            color (tuple): RGB color of the arrow.
        """
        location = carla.Location(x=pose[0, 3], y=pose[1, 3], z=pose[2, 3] + 1)
        rotation = carla.Rotation(
            pitch=np.rad2deg(np.arcsin(pose[2, 1])),
            yaw=np.rad2deg(np.arctan2(pose[1, 0], pose[0, 0])),
            roll=np.rad2deg(np.arctan2(pose[2, 0], pose[2, 2]))
        )

        # world.debug.draw_arrow(
        #     begin=location,
        #     end=carla.Location(
        #         x=pose[0, 3] + 2 * pose[0, 0],
        #         y=pose[1, 3] + 2 * pose[1, 0],
        #         z=pose[2, 3] + 2 * pose[2, 0] + 3
        #     ),
        #     thickness=0.025,
        #     arrow_size=0.2,
        #     color=carla.Color(*color),
        #     life_time=0.5
        # )

        world.debug.draw_point(
            location=location,
            size=0.1,
            color=carla.Color(*color),
            life_time=0.5
        )
