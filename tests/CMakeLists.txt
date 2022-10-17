set(smplpp_gtest_list
  TestVPoser
  TestGeometryUtils
  )

foreach(NAME IN LISTS smplpp_gtest_list)
  catkin_add_gtest(${NAME} src/${NAME}.cpp)
  target_link_libraries(${NAME} smplpp)
endforeach()
