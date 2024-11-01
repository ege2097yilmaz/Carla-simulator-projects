#include "carla_client.h"

int main(int argc, char* argv[]) {
    CarlaClient client;
    client.Connect("localhost", 2000);

    // Define a spawn location and rotation (X, Y, Z) and orientation (pitch, yaw, roll)
    carla::geom::Location spawn_location(230, 195, 40);
    carla::geom::Rotation spawn_rotation(0, 90, 0);  // Facing 90 degrees to the right

    // Spawn the vehicle with the defined location and rotation
    client.SpawnVehicle("vehicle.tesla.model3", spawn_location, spawn_rotation, false);

    // Run the simulation
    client.RunSimulation();

    return 0;
}
