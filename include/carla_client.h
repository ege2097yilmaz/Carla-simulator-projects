#ifndef CARLA_CLIENT_H
#define CARLA_CLIENT_H

#include <carla/client/Client.h>
#include <carla/client/Vehicle.h>
#include <carla/client/World.h>
#include <carla/client/Actor.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Sensor.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/geom/Transform.h>
#include <carla/geom/Location.h>
#include <carla/geom/Rotation.h>
#include <carla/client/Map.h>
#include <string>

class CarlaClient {
public:
    // Constructor
    CarlaClient();
    
    // Connect to the CARLA server
    void Connect(const std::string& host, int port);

    // Load the world and vehicle blueprint
    void LoadWorld();

    // Spawn a vehicle at a given location
    void SpawnVehicle(const std::string& vehicle_model, const carla::geom::Location& location, const carla::geom::Rotation& rotation, bool random_spawn = false);

    // Run the simulation (apply control, manage actors)
    void RunSimulation();

    // Destructor
    ~CarlaClient();

private:
    carla::client::Client *client_;   // Pointer to the CARLA client
    carla::client::World *world_;     // Pointer to the CARLA world
    carla::client::Vehicle::Control *vehicle_control_;  // Control object for vehicles
    carla::SharedPtr<carla::client::Vehicle> vehicle_;  // Pointer to the spawned vehicle
};

#endif  // CARLA_CLIENT_H
