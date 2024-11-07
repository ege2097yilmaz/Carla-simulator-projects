#!/usr/bin/env python3

from pointcloud_process import PointCloudData
from utility import *
import carla, time
from slam import SLAM
from collections import deque

# Set the rolling window size
pose_history = deque(maxlen=10)
vehicle_max_velocity = 3

def main():
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

    # setting for render mode
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

    # Set maxiumum velocity of the car
    traffic_manager = client.get_trafficmanager()
    traffic_manager.vehicle_percentage_speed_difference(vehicle, (1 - (vehicle_max_velocity / 13.89)) * 100)

    # initialize SLAM class
    slam_system = SLAM(optimization_interval=10, pcd_filename="real_time_map.pcd")

    slam_system.set_initial_point(location.x, location.y)

    # file_index = 0

    try:
        while True:
            # Retrieve the first frame of point cloud data from CARLA
            points_frame_1 = point_cloud_data.get_open3d_point_cloud(-2.0)
            
            if points_frame_1 is not None:
                # print(len(points_frame_1.points))
                if len(points_frame_1.points) >= 2400:
                    world.tick()

                    transform = vehicle.get_transform()
                    slam_system.build_graph(points_frame_1, transform.rotation.yaw) # SLAM main function to run it



                    """ Carla visualiztion functions """
                    # visualize keypoints
                    keypoints = slam_system.get_keypoints(transform.location.x, transform.location.y, transform.rotation.yaw)
                    visualize_keypoints_in_carla(client, keypoints)
                    points_frame_1 = None

                    # visualize pose grah
                    pose_graph = slam_system.get_pose_graph() 
                    visualize_pose_graph_in_carla(pose_graph, world, location.x, location.y)

                    # visualize visual odometry points
                    estimated_poses = slam_system.get_estimated_poses()
                    for pose in estimated_poses:
                        pose_history.append(pose)

                    for pose in pose_history:
                        point_cloud_data.visualize_pose_in_carla(world, pose)


                    # to save pcd files
                    # save_point_cloud_data(point_cloud_data, file_index, output_directory="point_cloud_data")
                    # file_index += 1
            
            points_frame_1 = None

    finally:
        # Cleanup
        point_cloud_data.destroy_sensor()
        vehicle.destroy()

if __name__ == '__main__':
    main()
