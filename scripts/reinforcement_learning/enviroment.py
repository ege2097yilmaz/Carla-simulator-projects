
import numpy as np
import time, carla

class CarlaParkingEnv:
    def __init__(self, client, goal_location):
        self.client = client
        self.world = client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spectator = self.world.get_spectator()
        self.vehicle = None
        self.goal_location = goal_location
        self.collision_sensor = None
        self.actor_list = []
        
        self.reset()

    def reset(self):
        # Clean up actors
        if self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None

        for actor in self.actor_list:
            if actor.is_alive:
                actor.destroy()
        self.actor_list = []

        # Spawn a new vehicle
        spawn_points = self.map.get_spawn_points()
        # spawn_point = random.choice(spawn_points)
        spawn_point = self.world.get_map().get_spawn_points()[0]
        vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)

        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        self.spectator.set_transform(carla.Transform(location + carla.Location(x=-75.0, y=25, z=6), rotation)) 

        self.visualize_goal_with_pointer()

        # Add a collision sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, z=2))
        self.collision_sensor = self.world.spawn_actor(collision_bp, collision_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)
        self.actor_list.append(self.collision_sensor)

        self.collision_occurred = False
        return self._get_state()

    def _on_collision(self, event):
        self.collision_occurred = True

    def _get_state(self):
        vehicle_transform = self.vehicle.get_transform()
        location = vehicle_transform.location
        x, y = location.x, location.y
        return np.array([x, y])

    def _compute_reward(self):
        current_location = self.vehicle.get_transform().location
        distance_to_goal = np.linalg.norm(np.array([current_location.x, current_location.y]) - np.array(self.goal_location))

        if distance_to_goal < 2.0:  # Reached goal
            return 100.0
        if self.collision_occurred:  # Collision occurred
            return -50.0
        return -distance_to_goal * 0.1  # Penalize based on distance to goal

    def step(self, action):
        throttle, steer = action
        control = carla.VehicleControl(throttle=throttle, steer=-1*steer)
        self.vehicle.apply_control(control)

        time.sleep(0.05)  # Step duration
        next_state = self._get_state()
        reward = self._compute_reward()
        done = self.collision_occurred or reward == 100.0
        return next_state, reward, done, {}
    
    def visualize_goal_with_pointer(self):
        """Visualizes the goal point with a debug pointer in the CARLA simulator."""
        # Convert goal to carla.Location
        goal_location = carla.Location(x=self.goal_location[0], y=self.goal_location[1], z=1.0)  # Set z for visibility

        # Color of the debug pointer (r, g, b)
        pointer_color = carla.Color(r=0, g=255, b=0)  # Green color

        # Draw a sphere to mark the goal location
        self.world.debug.draw_point(
            location=goal_location,  # Location of the goal
            size=0.3,                # Size of the pointer
            color=pointer_color,     # Color of the pointer
            life_time=0.0,           # Set to 0 for persistent visualization
            persistent_lines=True    # Keep the pointer visible
        )

    def close(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []
