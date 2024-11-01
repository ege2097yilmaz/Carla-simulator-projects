import carla
import time
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def get_cross_track_error(vehicle_location, target_waypoint):
    # Calculate the cross-track error (distance between vehicle and target point)
    vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])
    target_pos = np.array([target_waypoint.transform.location.x, target_waypoint.transform.location.y])
    error = np.linalg.norm(vehicle_pos - target_pos)
    return error
    
def track_error(vehicle_location, current_waypoint, next_waypoint):
    """
    Calculate the signed cross-track error (perpendicular distance from the vehicle to the path line segment).
    The sign indicates whether the vehicle is to the left or right of the path.
    """
    vehicle_pos = np.array([vehicle_location.x, vehicle_location.y])

    # Get the positions of the current and next waypoints
    wp_current_pos = np.array([current_waypoint.transform.location.x, current_waypoint.transform.location.y])
    wp_next_pos = np.array([next_waypoint.transform.location.x, next_waypoint.transform.location.y])

    path_vector = wp_next_pos - wp_current_pos
    path_length = np.linalg.norm(path_vector)

    # Normalize the path vector (unit vector in the direction of the path)
    path_direction = path_vector / path_length if path_length != 0 else path_vector

    # Vector from current waypoint to vehicle position
    vehicle_vector = vehicle_pos - wp_current_pos

    # Project the vehicle vector onto the path direction (dot product)
    projection_length = np.dot(vehicle_vector, path_direction)
    
    # Compute the projection point on the path (closest point on the path to the vehicle)
    projection_point = wp_current_pos + projection_length * path_direction

    # Calculate the cross-track error (perpendicular distance to the path)
    cross_track_error = np.linalg.norm(vehicle_pos - projection_point)

    # Now calculate the sign of the cross-track error using the cross product
    # We treat the vectors as 2D and compute the z-component of their 3D cross product
    cross_product_z = np.cross(path_direction, vehicle_vector)

    # If the cross product z-component is positive, the vehicle is to the left, otherwise to the right
    if cross_product_z < 0:
        cross_track_error = -cross_track_error

    return cross_track_error

def visualize_path(waypoints, world):
    """
    Visualize the waypoints as lines and points in the Carla simulator.
    """
    for i in range(len(waypoints) - 1):
        wp_current = waypoints[i]
        wp_next = waypoints[i + 1]
        world.debug.draw_line(wp_current.transform.location, wp_next.transform.location, thickness=0.1, color=carla.Color(255, 0, 0), life_time=1000.0)
        world.debug.draw_point(wp_current.transform.location, size=0.2, color=carla.Color(0, 255, 0), life_time=1000.0)

def find_closest_waypoint(vehicle_location, waypoints):
    """
    Find the closest waypoint to the current vehicle location.
    """
    closest_distance = float('inf')
    closest_idx = 0
    for i, waypoint in enumerate(waypoints):
        distance = get_cross_track_error(vehicle_location, waypoint)
        if distance < closest_distance:
            closest_distance = distance
            closest_idx = i
    return closest_idx

def follow_path(vehicle, waypoints, pid_steer, pid_throttle, world, lookahead_index=2):
    target_speed = 12.0  # Target speed in m/s
    dt = 1.0 / 20.0  # Assuming simulation runs at 20Hz

    while True:
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location

        # Find the closest waypoint to the vehicle
        closest_idx = find_closest_waypoint(vehicle_location, waypoints)

        # Use the lookahead waypoint (closest waypoint + lookahead_index)
        reference_idx = min(closest_idx + lookahead_index, len(waypoints) - 1)
        current_waypoint = waypoints[closest_idx]
        target_waypoint = waypoints[reference_idx]

        # Visualize the reference point (in yellow) as the lookahead waypoint
        world.debug.draw_point(target_waypoint.transform.location, size=0.3, color=carla.Color(25, 25, 255), life_time=0.1)

        # Calculate cross-track error
        # error = get_cross_track_error(vehicle_location, target_waypoint)
        cross_track_error = track_error(vehicle_location, current_waypoint, target_waypoint)

        # Control steering based on PID
        steer_output = pid_steer.control(cross_track_error, dt)

        # Apply throttle to maintain constant speed
        vehicle_velocity = vehicle.get_velocity()
        current_speed = np.linalg.norm([vehicle_velocity.x, vehicle_velocity.y])
        throttle_output = pid_throttle.control(target_speed - current_speed, dt)

        # Apply vehicle control
        control = carla.VehicleControl()
        control.steer = np.clip(steer_output, -1.0, 1.0) * -1
        control.throttle = np.clip(throttle_output, 0.0, 1.0)
        control.brake = 0.0

        vehicle.apply_control(control)

        # Stop if the vehicle is close to the last waypoint
        if closest_idx >= len(waypoints) - 1:
            print("Reached the final waypoint!")
            break
        
        time.sleep(dt)
        world.tick()  # Advance the simulation

def main():
    # Connect to Carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get a vehicle blueprint and spawn it
    blueprint_library = world.get_blueprint_library()
    vehicle_blueprint = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)

    # Get specific waypoints for the path (start point and next points)
    map = world.get_map()
    start_waypoint = map.get_waypoint(spawn_point.location)
    
    # Generate the path by following waypoints ahead (for example, 100 meters)
    waypoints = []
    current_waypoint = start_waypoint
    distance = 2.0  
    for _ in range(150):  
        waypoints.append(current_waypoint)
        next_waypoints = current_waypoint.next(distance)
        if len(next_waypoints) > 0:
            current_waypoint = next_waypoints[0]
        else:
            break

    # Visualize the path
    visualize_path(waypoints, world)

    # Initialize PID controllers for steering and throttle
    pid_steer = PIDController(Kp=0.3, Ki=0.001, Kd=0.05)
    pid_throttle = PIDController(Kp=0.05, Ki=0.01, Kd=0.0)

    # Follow the path with lookahead
    try:
        print("Path following with lookahead started")
        follow_path(vehicle, waypoints, pid_steer, pid_throttle, world, lookahead_index=6)
    finally:
        vehicle.destroy()

if __name__ == '__main__':
    main()