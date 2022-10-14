add_library(smplpp SHARED
  BlendShape.cpp
  JointRegression.cpp
  LinearBlendSkinning.cpp
  SMPL.cpp
  VPoser.cpp
  WorldTransformation.cpp
  toolbox/Exception.cpp
  # \todo Move Tester.cpp to tests directory
  toolbox/Tester.cpp
  toolbox/TorchEx.cpp
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
