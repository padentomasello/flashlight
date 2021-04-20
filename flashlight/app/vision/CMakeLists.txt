cmake_minimum_required(VERSION 3.10)

add_library(
  flashlight-app-vision
  ""
  )

target_link_libraries(
  flashlight-app-vision
  PUBLIC
  ${gflags_LIBRARIES}
  ${GLOG_LIBRARIES}
  flashlight
  flashlight-app-common
  )

target_include_directories(
  flashlight-app-vision
  PUBLIC
  ${gflags_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/models/CMakeLists.txt)

add_executable(Resnet34 "${CMAKE_CURRENT_LIST_DIR}/Resnet34.cpp")
target_link_libraries(Resnet34
  flashlight-app-vision
)
install(TARGETS Resnet34 RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
set_executable_output_directory(Resnet34 ${FL_BUILD_BINARY_OUTPUT_DIR}/vision)
