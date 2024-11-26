import cv2, carla, time, csv
import numpy as np
from visualOdom import VisualOdometry
from utility import *

vo = None
img1 = None
trajectory = []
start_time = time.time()

save_iteration = 0
iteration = 0
csv_data = []


trajectory_file = "trajectory.txt"

# Ensure the file is empty before starting
if os.path.exists(trajectory_file):
    os.remove(trajectory_file)

def main():
    # example scaling calculation
    gps_prev = (52.520008, 13.404954)  
    gps_curr = (52.520020, 13.405000)
    t_vo = np.array([[0.5], [0.0], [0.2]])
    scale, t_scaled = gnsProcess().estimate_scale(t_vo, gps_prev, gps_curr)

    print("Scale Factor:", scale)
    print("Scaled Translation Vector:\n", t_scaled)
    
    # Connect to the CARLA simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()

    # Spawn a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = world.get_map().get_spawn_points()[3]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print("Vehicle spawned.")

    # Attach a camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1000') 
    camera_bp.set_attribute('image_size_y', '800')
    camera_bp.set_attribute('fov', '90')           # Field of View (FOV)
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4)) 
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    print("Camera attached.")

    # Attach an IMU sensor
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_transform = carla.Transform(carla.Location(x=0.0, z=1.0))  # Adjust IMU position
    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
    print("IMU attached.")

    # imu.listen(lambda imu_data: print(f"IMU: {imu_data}"))

    # Attach a GNSS sensor
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_transform = carla.Transform(carla.Location(x=0.0, z=2.0))  # Adjust GNSS position
    gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
    print("GNSS attached.")

    # gnss.listen(lambda gnss_data: print(f"GNSS: Lat={gnss_data.latitude}, Lon={gnss_data.longitude}"))

    initial_position = vehicle.get_transform()
    rotation = vehicle.get_transform().rotation
    trajectory.append(initial_position.location)

    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(initial_position.location + carla.Location(x = -5.0, z=3), rotation)) 


    image_width = int(camera_bp.get_attribute('image_size_x').as_int())
    image_height = int(camera_bp.get_attribute('image_size_y').as_int())
    fov = float(camera_bp.get_attribute('fov').as_float())

    """ Calculate the intrinsic matrix """
    focal_length = image_width / (2.0 * (2 * (fov / 2) * 3.141592653589793 / 180.0))  # Convert FOV to radians
    K = np.array([
        [focal_length, 0, image_width / 2.0],
        [0, focal_length, image_height / 2.0],
        [0, 0, 1]]
    , dtype=np.float64)

    print("intrinsic matrix ")
    print(K)

    """Calculate the extrinsic matrix"""
    camera_transform = camera.get_transform()
    location = camera_transform.location
    t = np.array([[location.x], [location.y], [location.z]])  

    rotation = camera_transform.rotation
    roll = np.radians(rotation.roll)
    pitch = np.radians(rotation.pitch)
    yaw = np.radians(rotation.yaw)

    R = np.array([
        [np.cos(yaw) * np.cos(pitch), np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll), np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)],
        [np.sin(yaw) * np.cos(pitch), np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll), np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)],
        [-np.sin(pitch), np.cos(pitch) * np.sin(roll), np.cos(pitch) * np.cos(roll)]
    ])

    # Combine into the extrinsic matrix [R|t]
    Rt = np.hstack((R, t)) 

    print("Extrinsic Matrix [R|t]:")
    print(Rt)

    camera_matrix = K  
    camera_params = {"camera_matrix": camera_matrix, "dist_coeffs": np.zeros(4)}

    global vo
    vo = VisualOdometry(camera_params)

    def imu_callback(imu_data):
        # print("processing IMU datas")
        
        # Extract IMU readings
        # accel = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z])
        accel = np.array([0, 0, 0]) # ignore accelerations now
        gyro = np.array([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z * -1])
        timestamp = imu_data.timestamp
        vo.process_imu(accel, gyro, timestamp)
        time.sleep(0.02)

    def process_frame(image):
        # xtract the velocity components 
        velocity_vector = vehicle.get_velocity()
        velocity_x = velocity_vector.x
        velocity_y = velocity_vector.y
        velocity_z = velocity_vector.z

        # Calculate the speed in m/s
        speed = (velocity_x**2 + velocity_y**2 + velocity_z**2)**0.5
        # print("vehicle velocity: ")
        # print(speed)

        if(speed > 0.03):
            # Convert raw data to frame
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            frame = array.reshape((image.height, image.width, 4))[:, :, :3]  
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

            # Convert frame to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Process frame using visual odometry
            pose = vo.process_frame(frame_gray)
            trajectory = vo.get_trajectory()

            # Append the last trajectory point to the file
            with open(trajectory_file, "a") as file:
                for point in trajectory[-1:]:
                    file.write(f"{point[0]} {point[1]} {0.0}\n")

            print("Current Pose:\n", pose)

            black_screen = np.zeros((image_width, image_height, 3), dtype=np.uint8)

            center_x = black_screen.shape[1] // 2 
            center_y = black_screen.shape[0] // 2  

            # Draw trajectory centered
            if trajectory.shape[0] > 1:
                for i in range(1, len(trajectory)):
                    # Offset trajectory points to center
                    pt1 = (np.round(trajectory[i - 1][:2] + [center_x, center_y]).astype(int))
                    pt2 = (np.round(trajectory[i][:2] + [center_x, center_y]).astype(int))

                    height, width, _ = black_screen.shape
                    if all(0 <= coord < dim for coord, dim in zip(pt1 + pt2, [width, height] * 2)):
                        cv2.line(black_screen, tuple(pt1 ), tuple(pt2), (0, 255, 0), 5)

        # Show visualization
        # cv2.imshow("Visual Odometry", black_screen)
        # if cv2.waitKey(1) == ord('q'):
        #     return
            
        time.sleep(0.06)

            
            
    
    # camera.listen(lambda image: processImage(camera_bp, image, vehicle))
    camera.listen(lambda image: process_frame(image))
    imu.listen(imu_callback)

    # Control the vehicle
    vehicle.set_autopilot(True)

    # Set maxiumum velocity of the car
    traffic_manager = client.get_trafficmanager()
    traffic_manager.vehicle_percentage_speed_difference(vehicle, (1 - (4 / 13.89)) * 100)

    index = 0
    time.sleep(1.0)

    start_location = vehicle.get_location()

    try:
        print("starting simulation")
        while True:
            current_location = vehicle.get_location()
            distance_traveled = calculate_distance(start_location, current_location)
            if distance_traveled >= 250:  
                print("50 meters reached. Stopping the vehicle.")
                vehicle.set_autopilot(False) 
                break

            # control 
            # if(45 < index < 550):
            #     # print(trajectory)
            #     control = carla.VehicleControl()
            #     control.steer = np.clip(0.0, -1.0, 1.0) * -1
            #     control.throttle = np.clip(0.55, 0.0, 1.0)
            #     vehicle.apply_control(control)
                
            # else:
            #     vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

            time.sleep(0.06)
            index +=1
            world.tick()

    finally:
        print("Destroying actors...")
        vehicle.destroy()
        camera.destroy()
        imu.destroy()
        gnss.destroy()
        cv2.destroyAllWindows()
        print("All actors destroyed.")

if __name__ == '__main__':
    main()