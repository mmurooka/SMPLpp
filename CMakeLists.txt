cmake_minimum_required(VERSION 3.5)
project(smplpp)

# Need g++-8 to use C++17
# Ref. https://github.com/xtensor-stack/xtensor/issues/2289#issuecomment-766655824
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# catkin
find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  roslib
  rosbag
  std_msgs
  geometry_msgs
  visualization_msgs
  qp_solver_collection
)

# Eigen
find_package(Eigen3 REQUIRED)

# xtensor
find_package(xtensor REQUIRED)

# json
find_package(nlohmann_json REQUIRED)

# Torch
message("-- LIBTORCH_PATH: ${LIBTORCH_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_PATH})
find_package(Torch REQUIRED)

# libigl
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libigl)
igl_include(core)

# ezc3d
find_package(ezc3d REQUIRED)

add_message_files(
  FILES
  Motion.msg
  Instant.msg
  PoseParam.msg
)

generate_messages(
  DEPENDENCIES std_msgs geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  roscpp
  roslib
  rosbag
  std_msgs
  geometry_msgs
  visualization_msgs
  qp_solver_collection
)

add_subdirectory(src)

add_subdirectory(node)

if(CATKIN_ENABLE_TESTING)
  add_subdirectory(tests)
endif()
