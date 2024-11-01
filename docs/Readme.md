# 1. Installation of docker carla container

``` bash 
cd /path/to/the/CARLA-ROS-INTERFACE/docs
```

make and create carla image
```bash
docker build -t carla-gui-gpu .
```

If you want to delete the container after each runuse this
```bash
docker run -it --privileged  --gpus all   --network host  --rm -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix carla0.9.9_image
```


ıf you do not want to delete use this command
```bash
docker run -it --privileged  --gpus all   --network host   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix carla0.9.9_image
```

## 2. Go into the container
```bash
# optionally you can go into the container using this
docker ecxec -it [DOCKER_ID] bash
```

## 3. If you want to compile the LibCarla for CPP API
```bash
git clone https://github.com/carla-simulator/carla
cd carla

make PythonAPI          # Generates the Python API, optional if only C++ API is needed
make launch           # Builds the CARLA C++ API
```

## 4. Run CARLA with the C++ API
You can now launch CARLA and start interacting with the simulator using C++ code. 
Make sure you link to the libcarla library and include the appropriate headers in your C++ project.

In terminal
```bash
docker run -it --privileged --user root --gpus all   --network host  --rm -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix carla0.9.9_image

cd /path/to/the/API # where you want to build th API

export CC=/usr/lib/llvm-10/lib/clang  # Replace with actual path
export CXX=/usr/bin/clang++  # Replace with actual path
```
Include headers;
```cpp
#include "carla/client/Client.h"
#include "carla/client/Vehicle.h"
```

you need to adjust the CMakeLists.txt as seen

```cmake

project(carla_example)

find_package(Carla REQUIRED)

add_executable(example main.cpp)

target_link_libraries(example ${CARLA_LIBRARIES})
target_include_directories(example PRIVATE ${CARLA_INCLUDE_DIRS})
```

Compile the project
```bash
mkdir build
cd build
cmake ..
make
```
## Troubleshootings
If you face any problem about carla vulkam while running carla, run this command in the container
```bash
sudo apt-get install vulkan-utils
```
