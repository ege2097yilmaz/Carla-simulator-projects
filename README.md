# Carla-simulator-projects
A couple of robotic projects withing Carla Simulator

This CMake package integrates `libcarla`, `rpclib`, and custom control algorithms to facilitate the development and simulation of autonomous vehicle behavior. The package is structured to support efficient and modular use in projects requiring CARLA-based vehicle simulations.

## Folder Structure

- **libcarla**: Contains the CARLA library, which provides interfaces for simulating autonomous driving environments.
- **rpclib**: Includes the RPC (Remote Procedure Call) library to enable communication between CARLA and client scripts.
- **scripts**: A collection of control algorithms and utility scripts. Notable algorithms include:
  - **PID Controller**: A classical control algorithm for managing vehicle dynamics.
  - **MPC Controller**: A Model Predictive Controller for optimizing vehicle performance over a prediction horizon.
  - **Cruise control**: A controller to follow lead car in simulation.
  - **SLAM**: A slam project that localizes the vehicle using visul odometry (pointcloud matchin with ICP) and creates a map with pointcloud datas.

## Installation

To build and install this package, follow these steps:

1. Ensure you have CMake and necessary dependencies installed.
2. Clone this repository and navigate to its root directory.
3. Run the following commands:

   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

# Usage
* libcarla and rpclib in the include directory are set up to work seamlessly with CARLA simulations.
* The control algorithms in the scripts folder can be run or modified to experiment with various vehicle control strategies.

### PID controller to follow path
![Watch Video](videos/PID.gif)


### MPC controller to follow path
![Watch Video](videos/mpc.gif)

### SLAM implementation to localize the vehicle and mapping enviroment where robot moves
![Watch Video](videos/slam.gif)

### Mapping outputs
<img src="images/slam4.png" alt="slam4" width="800">

<img src="images/slam5.png" alt="slam5" width="800">


<img src="images/slam_all_map.png" alt="slam_all_map" width="800">