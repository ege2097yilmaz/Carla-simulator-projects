#include <Python.h>
#include <iostream>

void initializePython() {
    Py_Initialize();
}

void finalizePython() {
    Py_Finalize();
}

int main() {
    initializePython();

    // Import the CARLA library
    PyObject* carlaModule = PyImport_ImportModule("carla");
    if (!carlaModule) {
        PyErr_Print();
        std::cerr << "Error: could not import 'carla' module" << std::endl;
        return 1;
    }

    // Connect to the CARLA server using the Client class
    PyObject* carlaClientClass = PyObject_GetAttrString(carlaModule, "Client");
    PyObject* clientInstance = nullptr;
    if (carlaClientClass && PyCallable_Check(carlaClientClass)) {
        clientInstance = PyObject_CallFunction(carlaClientClass, "si", "localhost", 2000);
        if (!clientInstance) {
            PyErr_Print();
            std::cerr << "Error: could not create a Client instance" << std::endl;
            return 1;
        }
    }

    // Get the world object
    PyObject* worldObject = PyObject_CallMethod(clientInstance, "get_world", nullptr);
    if (!worldObject) {
        PyErr_Print();
        std::cerr << "Error: could not get world" << std::endl;
        return 1;
    }

    // Get the blueprint library to choose a vehicle blueprint
    PyObject* blueprintLibrary = PyObject_CallMethod(worldObject, "get_blueprint_library", nullptr);
    if (!blueprintLibrary) {
        PyErr_Print();
        std::cerr << "Error: could not get blueprint library" << std::endl;
        return 1;
    }

    // Find a specific blueprint for a car, e.g., "vehicle.tesla.model3"
    PyObject* vehicleBlueprint = PyObject_CallMethod(blueprintLibrary, "find", "s", "vehicle.tesla.model3");
    if (!vehicleBlueprint) {
        PyErr_Print();
        std::cerr << "Error: could not find vehicle blueprint" << std::endl;
        return 1;
    }

    // Create a specific spawn location with x, y, z coordinates
    PyObject* locationClass = PyObject_GetAttrString(carlaModule, "Location");
    PyObject* spawnLocation = nullptr;
    if (locationClass && PyCallable_Check(locationClass)) {
        spawnLocation = PyObject_CallFunction(locationClass, "(ddd)", -50.0, 85.0, 10.0);  // Example coordinates (x, y, z)
    }
    if (!spawnLocation) {
        PyErr_Print();
        std::cerr << "Error: could not create location" << std::endl;
        return 1;
    }

    // Create a specific rotation with pitch, yaw, roll
    PyObject* rotationClass = PyObject_GetAttrString(carlaModule, "Rotation");
    PyObject* rotation = nullptr;
    if (rotationClass && PyCallable_Check(rotationClass)) {
        rotation = PyObject_CallFunction(rotationClass, "(ddd)", 0.0, 0.0, 0.0);  // Example rotation (pitch, yaw, roll)
    }
    if (!rotation) {
        PyErr_Print();
        std::cerr << "Error: could not create rotation" << std::endl;
        return 1;
    }

    // Create a Transform object with the specific location and rotation
    PyObject* transformClass = PyObject_GetAttrString(carlaModule, "Transform");
    PyObject* spawnTransform = nullptr;
    if (transformClass && PyCallable_Check(transformClass)) {
        spawnTransform = PyObject_CallFunction(transformClass, "OO", spawnLocation, rotation);
    }
    if (!spawnTransform) {
        PyErr_Print();
        std::cerr << "Error: could not create transform" << std::endl;
        return 1;
    }

    // Spawn the vehicle at the specific spawn point
    PyObject* vehicleActor = PyObject_CallMethod(worldObject, "spawn_actor", "OO", vehicleBlueprint, spawnTransform);
    if (!vehicleActor) {
        PyErr_Print();
        std::cerr << "Error: could not spawn vehicle" << std::endl;
        return 1;
    }

    std::cout << "Vehicle spawned successfully at the specified location!" << std::endl;

    // Cleanup
    Py_DECREF(vehicleActor);
    Py_DECREF(spawnTransform);
    Py_DECREF(spawnLocation);
    Py_DECREF(rotation);
    Py_DECREF(vehicleBlueprint);
    Py_DECREF(blueprintLibrary);
    Py_DECREF(worldObject);
    Py_DECREF(clientInstance);
    Py_DECREF(carlaClientClass);
    Py_DECREF(carlaModule);

    finalizePython();
    return 0;
}
