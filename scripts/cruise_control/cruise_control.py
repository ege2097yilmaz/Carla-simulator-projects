import carla
import time, math

# Connect to CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load the CARLA world
world = client.get_world()

# Get the blueprint library
blueprint_library = world.get_blueprint_library()

# Define custom spawn points for the lead and follow cars
lead_car_location = carla.Transform(carla.Location(x=-49.0, y=95.0, z=0.3), carla.Rotation(yaw=90))
follow_car_location = carla.Transform(carla.Location(x=-49.0, y=85.0, z=0.3), carla.Rotation(yaw=90))

# Spawn the lead car
lead_car_bp = blueprint_library.filter('vehicle.*')[0]
lead_car = world.try_spawn_actor(lead_car_bp, lead_car_location)

# Spawn the follow car
follow_car_bp = blueprint_library.filter('vehicle.*')[1]
follow_car = world.try_spawn_actor(follow_car_bp, follow_car_location)

# Make sure both cars spawned successfully
if not lead_car or not follow_car:
    raise Exception("Could not spawn both cars. Check your locations or try running the script again.")

# Set up an autopilot for the lead car
lead_car.set_autopilot(True)

def follow_lead_car(follow_car, lead_car):
    while True:
        lead_location = lead_car.get_location()
        follow_location = follow_car.get_location()

        lead_transform = lead_car.get_transform()
        follow_transform = follow_car.get_transform()

        dx = lead_location.x - follow_location.x
        dy = lead_location.y - follow_location.y

        distance_to_lead = math.sqrt(dx**2 + dy**2)

        # Calculate the desired distance to maintain
        desired_distance = 9.0 

        lead_yaw = math.radians(lead_transform.rotation.yaw)
        follow_yaw = math.radians(follow_transform.rotation.yaw)
        heading_error = lead_yaw - follow_yaw

        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # Calculate the cross-track (lateral) error
        cross_track_error = (dx * math.sin(follow_yaw) - dy * math.cos(follow_yaw))

        steer_correction = heading_error + 0.2 * cross_track_error * -1

        steer_correction = max(min(steer_correction, 1.0), -1.0)

        # P control
        throttle = (distance_to_lead - desired_distance) * 1.2

        if distance_to_lead > desired_distance:
            print("throttle")
            follow_car.apply_control(carla.VehicleControl(throttle=throttle, steer=steer_correction))
        else:
            print("braking")
            follow_car.apply_control(carla.VehicleControl(throttle=0.0, brake=throttle * 5, steer=steer_correction))

        time.sleep(0.1)

# Run the follow function
try:
    follow_lead_car(follow_car, lead_car)
finally:
    # Clean up and destroy actors
    lead_car.destroy()
    follow_car.destroy()
