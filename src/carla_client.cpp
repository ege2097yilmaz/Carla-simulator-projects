#include "carla_client.h"
#include <iostream>
#include <memory>
#include <boost/optional.hpp>
#include <boost/shared_ptr.hpp>

using namespace carla::client;
using namespace carla::geom;
using namespace carla::rpc;
using namespace std;

CarlaClient::CarlaClient() 
    : client_(nullptr), world_(nullptr), vehicle_control_(nullptr), vehicle_(nullptr) {}

CarlaClient::~CarlaClient() {
    // Clean up any dynamic memory or resources
    delete client_;
    delete world_;
    delete vehicle_control_;
}

void CarlaClient::Connect(const std::string& host, int port) {
    try {
        // Create a client and connect to the CARLA server
        client_ = new Client(host, port);
        client_->SetTimeout(10s);  // Set a 10-second timeout for network operations
        cout << "Connected to CARLA server at " << host << ":" << port << endl;

        // Load the world after connection
        LoadWorld();
    } catch (const std::exception &e) {
        cerr << "Error connecting to CARLA server: " << e.what() << endl;
    }
}

void CarlaClient::LoadWorld() {
    if (!client_) {
        cerr << "Client not initialized. Call Connect() first." << endl;
        return;
    }

    try {
        // Get the world from the server
        world_ = new World(client_->GetWorld());
        cout << "World loaded successfully!" << endl;
    } catch (const std::exception &e) {
        cerr << "Error loading world: " << e.what() << endl;
    }
}

void CarlaClient::SpawnVehicle(const std::string& vehicle_model, const Location& location, const Rotation& rotation, bool random_spawn) {
    if (!world_) {
        cerr << "World not initialized. Call LoadWorld() first." << endl;
        return;
    }

    try {
        // Get the blueprint library for spawning actors
        auto blueprint_library = world_->GetBlueprintLibrary();
        
        // Find the blueprint of the desired vehicle model
        auto vehicle_bp = blueprint_library->Find(vehicle_model);
        if (!vehicle_bp) {
            cerr << "Vehicle model " << vehicle_model << " not found." << endl;
            return;
        }

        // If random_spawn is true, find a random location in the world to spawn the vehicle
        Transform spawn_transform;
        if (random_spawn) {
            auto spawn_points = world_->GetMap()->GetRecommendedSpawnPoints();
            if (!spawn_points.empty()) {
                spawn_transform = spawn_points[std::rand() % spawn_points.size()];
                cout << "Random spawn point selected." << endl;
            } else {
                cerr << "No spawn points available." << endl;
                return;
            }
        } else {
            // Specify the user-defined transform (location and orientation)
            spawn_transform = Transform(location, rotation);
        }

        // Spawn the vehicle in the world
        vehicle_ = boost::dynamic_pointer_cast<carla::client::Vehicle>(world_->SpawnActor(*vehicle_bp, spawn_transform));
        if (vehicle_) {
            cout << "Vehicle " << vehicle_model << " spawned successfully at location (" 
                 << spawn_transform.location.x << ", " 
                 << spawn_transform.location.y << ", " 
                 << spawn_transform.location.z << ")." << endl;
        } else {
            cerr << "Failed to spawn vehicle." << endl;
        }
    } catch (const std::exception &e) {
        cerr << "Error spawning vehicle: " << e.what() << endl;
    }
}

void CarlaClient::RunSimulation() {
    if (!vehicle_) {
        cerr << "Vehicle not initialized. Call SpawnVehicle() first." << endl;
        return;
    }

    try {
        // Set up vehicle control parameters
        vehicle_control_ = new Vehicle::Control();
        vehicle_control_->throttle = 0.5f;  // Set throttle to 50%
        vehicle_control_->steer = 0.0f;     // No steering (straight)

        // Run the simulation for a set number of iterations (or continuously)
        for (int i = 0; i < 1000; ++i) {
            // Apply control to the vehicle
            vehicle_->ApplyControl(*vehicle_control_);

            // Tick the world to move the simulation forward
            world_->Tick(1s);

            // Print vehicle location every few steps
            if (i % 100 == 0) {
                auto location = vehicle_->GetTransform().location;
                cout << "Vehicle location: x = " << location.x << ", y = " << location.y << ", z = " << location.z << endl;
            }
        }
    } catch (const std::exception &e) {
        cerr << "Error running simulation: " << e.what() << endl;
    }
}
