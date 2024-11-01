import carla
import random
import time

def main():
    # Connect to the Carla server
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)  # Increase timeout to 20 seconds

    try:
        # Get the world and blueprint library
        world = client.get_world()
    except RuntimeError as e:
        print(f"Failed to get world: {e}")
        return

    blueprint_library = world.get_blueprint_library()

    # Get a random vehicle blueprint (Tesla Model 3)
    vehicle_bp = blueprint_library.filter('model3')[0]

    # Get a random spawn point
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    print(f"Vehicle {vehicle.type_id} spawned at {spawn_point.location}")

    # Set autopilot on the vehicle
    vehicle.set_autopilot(True)

    # Create a spectator view (camera) and set it to follow the vehicle
    spectator = world.get_spectator()
    transform = carla.Transform(spawn_point.location + carla.Location(z=40), carla.Rotation(pitch=-45))
    spectator.set_transform(transform)

    # Constantly update the camera to follow the car
    try:
        while True:
            # Get the vehicle's current transform
            transform = vehicle.get_transform()
            # spectator.set_transform(carla.Transform(transform.location + carla.Location(x = -5.0, y=0.0, z=30), carla.Rotation(roll = 0.0, pitch=-45, yaw=45)))
            time.sleep(0.1)  # Update every 100ms
    finally:
        # Clean up: destroy the vehicle when done
        vehicle.destroy()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Cancelled by user.')
