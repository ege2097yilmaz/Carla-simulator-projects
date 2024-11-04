#!/usr/bin/env python3

from pointcloud_process import PointCloudData
from utility import *
import carla, time
from slam import SLAM
from collections import deque

# Set the rolling window size
pose_history = deque(maxlen=10)

def main():
    # Initialize CARLA client and load the world
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Spawn a vehicle to attach the sensor to
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Enable autopilot to make the vehicle move autonomously
    vehicle.set_autopilot(True)

    # Initialize PointCloudData and set up the LIDAR sensor
    point_cloud_data = PointCloudData()    

    file_index = 0
    start_time = time.time()
    try:
        for _ in range(100):
            point_cloud_data.setup_lidar_sensor(vehicle)
            current_time = time.time()
            
            # to save pcd files
            # save_point_cloud_data(point_cloud_data, start_time, current_time, file_index, output_directory="point_cloud_data")

            world.tick()
            file_index += 1
    finally:
        # Cleanup
        point_cloud_data.destroy_sensor()
        vehicle.destroy()

def main2():
    # Initialize CARLA client and load the world
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    spectator = world.get_spectator()

    # Spawn a vehicle to attach the sensor to
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    settings = world.get_settings()
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    # Enable autopilot to make the vehicle move autonomously
    vehicle.set_autopilot(True)

    # Initialize PointCloudData and set up the LIDAR sensor
    point_cloud_data = PointCloudData()
    point_cloud_data.setup_lidar_sensor(vehicle)

    transform = vehicle.get_transform()
    location = transform.location
    rotation = transform.rotation
    spectator.set_transform(carla.Transform(location + carla.Location(z=10), rotation)) 

    # initialize SLAM class
    slam_system = SLAM()

    slam_system.set_initial_point(location.x, location.y)

    file_index = 0

    try:
        while True:
            # Retrieve the first frame of point cloud data from CARLA
            points_frame_1 = point_cloud_data.get_open3d_point_cloud(0.1)

            if points_frame_1 is not None:
                world.tick()

                # todo process the frames and apply slam
                transform = vehicle.get_transform()
                slam_system.build_graph(points_frame_1, transform.rotation.yaw)

                estimated_poses = slam_system.get_estimated_poses()
                for pose in estimated_poses:
                    pose_history.append(pose)

                for pose in pose_history:
                    point_cloud_data.visualize_pose_in_carla(world, pose)


                # to save pcd files
                # save_point_cloud_data(point_cloud_data, file_index, output_directory="point_cloud_data")
                # file_index += 1

    finally:
        # Cleanup
        point_cloud_data.destroy_sensor()
        vehicle.destroy()

if __name__ == '__main__':
    main2()
