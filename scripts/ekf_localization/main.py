import carla, time
import numpy as np
from data_processing import SensorDataProcessor
from fusion import KF, EKF
import csv

def measurement_function(state):
    return np.array([state[0], state[1]])  # GNSS provides [x, y]

def measurement_jacobian():
    H = np.zeros((2, 5))
    H[0, 0] = 1  # ∂x/∂x
    H[1, 1] = 1  # ∂y/∂y
    return H

def main():
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Set up vehicle
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    time.sleep(2.0)
    initial_position = vehicle.get_transform()
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(initial_position.location + carla.Location(x = -5.0, z=3))) 

    # Set up sensors
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_sensor = world.spawn_actor(
        imu_bp,
        carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0)),
        attach_to=vehicle,
    )

    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_sensor = world.spawn_actor(
        gnss_bp,
        carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5)),
        attach_to=vehicle,
    )

    # Sensor data containers
    imu_data = None
    gnss_data = None

    # Sensor callbacks
    def imu_callback(data):
        nonlocal imu_data
        imu_data = {
            'accel': [data.accelerometer.x, data.accelerometer.y, data.accelerometer.z],
            'gyro': [data.gyroscope.x, data.gyroscope.y, data.gyroscope.z],
        }

    def gnss_callback(data):
        nonlocal gnss_data
        gnss_data = {'latitude': data.latitude, 'longitude': data.longitude}

    imu_sensor.listen(imu_callback)
    gnss_sensor.listen(gnss_callback)

    # Wait until GNSS data is available
    while gnss_data is None:
        print("Waiting for GNSS data...")
        world.tick()  # Step the simulation forward
        continue

    print("GNSS data received. Initializing EKF...")

    # Initialize processor and EKF
    processor = SensorDataProcessor(ref_lat=gnss_data['latitude'], ref_lon=gnss_data['longitude'])
    
    initial_state = np.zeros(5)  # [x, y, velocity, orientation, yaw_rate]
    # initial_state = np.array([0.0, 0.0, 0.0, initial_position.rotation.yaw, 0.0])
    initial_covariance = np.eye(5) * 0.1
    process_noise = np.eye(5) * 0.01
    measurement_noise = np.eye(2) * 0.1

    # ekf = KF(initial_state, initial_covariance, process_noise, measurement_noise)
    ekf = EKF(initial_state, initial_covariance, process_noise, measurement_noise)


    vehicle.set_autopilot(True)

    # Set maxiumum velocity of the car
    traffic_manager = client.get_trafficmanager()
    traffic_manager.vehicle_percentage_speed_difference(vehicle, (1 - (4 / 13.89)) * 100)

    output_file = open('trajectory_comparison.csv', 'w', newline='')
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(['time', 'gt_x', 'gt_y', 'ekf_x', 'ekf_y'])

    try:
        previous_time = world.get_snapshot().timestamp.elapsed_seconds

        # Initialize the reference point for ground truth
        ref_x, ref_y = None, None

        while True:
            if imu_data and gnss_data:
                # Preprocess GNSS and IMU data
                gnss_cartesian = processor.gnss_to_cartesian(gnss_data['latitude'], gnss_data['longitude'])
                accel, gyro = processor.process_imu(imu_data['accel'], imu_data['gyro'])

                # Get ground truth (vehicle location from CARLA)
                vehicle_transform = vehicle.get_transform()
                gt_x = vehicle_transform.location.x
                gt_y = vehicle_transform.location.y

                if ref_x is None and ref_y is None:
                    ref_x, ref_y = gt_x, gt_y

                # Calculate relative ground truth
                relative_gt_x = gt_x - ref_x
                relative_gt_y = gt_y - ref_y

                # Calculate time step
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                dt = current_time - previous_time
                previous_time = current_time

                # EKF Prediction Step
                # F = np.eye(5)  # State transition matrix
                # ekf.predict(F, dt, accel, gyro)
                ekf.predict(dt, accel, gyro)

                # EKF Update Step
                # H = np.zeros((2, 5))  # Observation matrix for x and y
                # H[0, 0], H[1, 1] = 1, 1
                # ekf.update(gnss_cartesian, H)
                def measurement_function(state):
                    # GNSS provides [x, y] in Cartesian coordinates
                    return np.array([state[0], state[1]])

                def measurement_jacobian(state):
                    # Jacobian of measurement function
                    H = np.zeros((2, 5))
                    H[0, 0] = 1  # ∂z_x/∂x
                    H[1, 1] = 1  # ∂z_y/∂y
                    return H

                ekf.update(gnss_cartesian, measurement_function, measurement_jacobian)


                ekf_x, ekf_y = ekf.state[0], ekf.state[1]
                adjusted_ekf_x = ekf_y
                adjusted_ekf_y = -ekf_x
                
                print(f"Time: {current_time}, GT: ({relative_gt_x}, {relative_gt_y}), EKF: ({adjusted_ekf_x}, {adjusted_ekf_y})")

                # Write data to CSV
                csv_writer.writerow([current_time, relative_gt_x, relative_gt_y, adjusted_ekf_x, adjusted_ekf_y])

                time.sleep(0.05)

    except KeyboardInterrupt:
        print("Simulation stopped.")

    finally:
        # Cleanup
        imu_sensor.stop()
        gnss_sensor.stop()
        vehicle.destroy()
        print("All sensors and vehicle destroyed.")

if __name__ == '__main__':
    main()