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
# Need g++-8 to use C++17
# Ref: https://github.com/xtensor-stack/xtensor/issues/2289#issuecomment-766655824
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
## OUTPUTS
##
add_library(smplpp SHARED
  src/BlendShape.cpp
  src/JointRegression.cpp
  src/LinearBlendSkinning.cpp
  src/SMPL.cpp
  src/VPoser.cpp
  src/WorldTransformation.cpp
  src/definition/def.cpp
  src/toolbox/Exception.cpp
  src/toolbox/Tester.cpp
  src/toolbox/TorchEx.cpp
)
target_include_directories(smplpp PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${xtensor_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)
target_link_libraries(smplpp PUBLIC
  xtensor
  ${xtensor_blas_LIBRARIES}
  nlohmann_json::nlohmann_json
  ${TORCH_LIBRARIES}
  stdc++fs # std::experimental::filesystem
  ${catkin_LIBRARIES}
)

add_executable(node
  node/node.cpp)
target_link_libraries(node PRIVATE
  smplpp
)
