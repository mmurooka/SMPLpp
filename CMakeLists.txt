##
## CMAKE VERSION
##
cmake_minimum_required(VERSION 3.5)

##
## PROJECT NAME
##
project(smplpp)

##
## COMPILER SETTINGS
##
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

##
## PACKAGES
##

# catkin
find_package(catkin REQUIRED COMPONENTS
  message_generation
  roscpp
  roslib
  std_msgs
  geometry_msgs
  visualization_msgs
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

##
## CATKIN
##
add_message_files(
  FILES
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
  std_msgs
  geometry_msgs
  visualization_msgs
)

##
## GATHER FILES
##
file(GLOB HEADER
    "include/definition/*.h"
    "include/definition/*.hpp"
    "include/toolbox/*.h"
    "include/toolbox/*.hpp"
    "include/smpl/*.h"
    "include/smpl/*.hpp"
)

file(GLOB SOURCE
    "src/definition/*.cpp"
    "src/toolbox/*.cpp"
    "src/smpl/*.cpp"
)

##
## INCLUDE DIRECTORIES
##
set(PROJECT_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${xtensor_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)

##
## LIBRARIES
##
set(PROJECT_LIBRARIES
    xtensor
    ${xtensor_blas_LIBRARIES}
    nlohmann_json::nlohmann_json
    ${TORCH_LIBRARIES}
    stdc++fs # std::experimental::filesystem
    ${catkin_LIBRARIES}
)

##
## OUTPUTS
##
add_library(smplpp SHARED
  ${HEADER}
  ${SOURCE}
)
target_include_directories(smplpp PUBLIC
  ${PROJECT_INCLUDE_DIRS}
)
target_link_libraries(smplpp PUBLIC
  ${PROJECT_LIBRARIES}
)

add_executable(main
  src/main.cpp)
target_link_libraries(main PRIVATE
  smplpp
)
