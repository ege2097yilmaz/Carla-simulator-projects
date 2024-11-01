import carla
import time
import numpy as np
from mpc_class import MPCController

dt = 0.1
N = 2
L = 2.5

def main():

    # Connect to the Carla server
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    # Get the world and map
    world = client.get_world()
    map = world.get_map()

    # Initialize MPC controller
    mpc = MPCController(N, dt, L)

    # Set spawn and target locations
    fixed_spawn_location = carla.Location(x=-50.0, y=85.0, z=0.3)
    fixed_spawn_rotation = carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
    spawn_point = carla.Transform(fixed_spawn_location, fixed_spawn_rotation)

    # Choose a vehicle blueprint and spawn it at the fixed location
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    spectator = world.get_spectator()
    spectator_location = carla.Location(
        x=fixed_spawn_location.x - 10.0,
        y=fixed_spawn_location.y,
        z=fixed_spawn_location.z + 5.0
    )
    spectator_rotation = carla.Rotation(pitch=-15, yaw=0, roll=0)
    spectator.set_transform(carla.Transform(spectator_location, spectator_rotation))

    if vehicle is not None:
        # Generate waypoints for the path from spawn to target
        path_waypoints = []
        current_waypoint = map.get_waypoint(vehicle.get_location())

        for _ in range(100):
            path_waypoints.append(current_waypoint)
            next_waypoints = current_waypoint.next(2.0)
            if next_waypoints:
                current_waypoint = next_waypoints[0]
            else:
                break

        for i in range(len(path_waypoints) - 1):
            start_location = path_waypoints[i].transform.location
            end_location = path_waypoints[i + 1].transform.location
            world.debug.draw_line(
                start_location,
                end_location,
                thickness=0.1,
                color=carla.Color(0, 0, 255),
                life_time=0.0,
                persistent_lines=True
            )

        # Main MPC control loop
        try:
            while True:
                vehicle_transform = vehicle.get_transform()
                vehicle_location = vehicle_transform.location
                vehicle_x = vehicle_location.x
                vehicle_y = vehicle_location.y
                vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

                vehicle_velocity = vehicle.get_velocity()
                v = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)

                x0 = np.array([vehicle_x, vehicle_y, vehicle_yaw, v])

                waypoints_x = [wp.transform.location.x for wp in path_waypoints]
                waypoints_y = [wp.transform.location.y for wp in path_waypoints]

                # Find the closest waypoint index
                distances = [(vehicle_x - wx)**2 + (vehicle_y - wy)**2 for wx, wy in zip(waypoints_x, waypoints_y)]
                closest_idx = np.argmin(distances)

                # Get reference waypoints over the horizon
                ref_waypoints = path_waypoints[closest_idx:closest_idx + N + 1]

                # Handle if not enough waypoints
                if len(ref_waypoints) < N + 1:
                    # Pad the rest with the last waypoint
                    last_wp = ref_waypoints[-1]
                    for _ in range(N + 1 - len(ref_waypoints)):
                        ref_waypoints.append(last_wp)

                ref_traj = np.zeros((4, N + 1))
                for i, wp in enumerate(ref_waypoints):
                    ref_x = wp.transform.location.x
                    ref_y = wp.transform.location.y
                    ref_yaw = np.radians(wp.transform.rotation.yaw)
                    ref_v = 5.0 
                    ref_traj[:, i] = [ref_x, ref_y, ref_yaw, ref_v]

                    ref_location = carla.Location(x=ref_x, y=ref_y, z=0.0)

                    # Draw the waypoint on the map
                    world.debug.draw_point(ref_location, size=0.1, color=carla.Color(255, 0, 0), life_time=1.0)

                # Call MPC 
                delta, a = mpc.solve(x0, ref_traj)

                delta_max = mpc.delta_max
                steer = delta / delta_max
                steer = np.clip(steer, -1.0, 1.0)

                a_max = mpc.a_max
                a_min = mpc.a_min

                # normalize the throttle
                if a >= 0:
                    throttle = min(a / a_max, 1.0)
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = min(-a / abs(a_min), 1.0)

                print(f"Output of the MPC - Throttle: {throttle:.2f}, Steer: {steer:.2f}, Brake: {brake:.2f}")

                # Apply control to the vehicle
                vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

                world.tick()
                time.sleep(dt)
        finally:
            # Clean up and destroy the vehicle
            vehicle.destroy()
    else:
        print("Failed to spawn vehicle at the fixed location.")

if __name__ == '__main__':
    main()