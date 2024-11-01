cmake_minimum_required(VERSION 3.10)
project(carla_simulator_projects)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

set(RPCLIB_INCLUDE_DIR ${CMAKE_SOURCE_DIR}include/rpclib/include)
set(RPCLIB_LIB_DIR ${CMAKE_SOURCE_DIR}include/rpclib/lib)

set(CARLALIB_DIR ${CMAKE_SOURCE_DIR}include/libcarla-install/lib)
set(carla_libs  
    ${CARLALIB_DIR}/libboost_filesystem.a 
    ${CARLALIB_DIR}/libboost_numpy36.a 
    ${CARLALIB_DIR}/libboost_program_options.a 
    ${CARLALIB_DIR}/libboost_python36.a 
    ${CARLALIB_DIR}/libboost_system.a 
    ${CARLALIB_DIR}/libcarla_client.a 
    ${CARLALIB_DIR}/librpc.a 
    ${CARLALIB_DIR}/libDebugUtils.a 
    ${CARLALIB_DIR}/libDetour.a 
    ${CARLALIB_DIR}/libDetourCrowd.a 
    ${CARLALIB_DIR}/libDetourTileCache.a 
    ${CARLALIB_DIR}/libRecast.a
)


include_directories(
    include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/third_party/boost/include
    ${CMAKE_SOURCE_DIR}/include/libcarla-install/include
    ${CMAKE_SOURCE_DIR}/include/libcarla-install/include/system/
)

include_directories(${RPCLIB_INCLUDE_DIR})

link_directories(
    ${CMAKE_SOURCE_DIR}include/libcarla-install/lib
    ${CMAKE_SOURCE_DIR}include/lib   
)

# Define the executable
add_executable(carla_cpp_project src/main.cpp src/carla_client.cpp)

# Link against required libraries (CARLA client, Boost, etc.)
target_link_libraries(carla_cpp_project
    libcarla_client.a
    libboost_system.a
    libpthread.so 
    ${carla_libs}
    ${RPCLIB_LIB_DIR}
)
