cmake_minimum_required(VERSION 3.5.1)

add_library(
  flashlight-app-image-classification
  ""
  )

# Find GLog
#find_package(GLOG REQUIRED)
#if (GLOG_FOUND)
  #message(STATUS "GLOG found")
#else()
  #message(FATAL_ERROR "GLOG not found")
#endif()

## Find GFlags
#find_package(GFLAGS REQUIRED)
#if (GFLAGS_FOUND)
  #message(STATUS "GFLAGS found")
#else()
  #message(FATAL_ERROR "GFLAGS not found")
#endif()

target_link_libraries(
  flashlight-app-image-classification
  PUBLIC
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-image-classification
  PUBLIC
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)


add_executable(Resnet34 ${CMAKE_CURRENT_LIST_DIR}/examples/Resnet34.cpp)
target_link_libraries(Resnet34 flashlight-app-image-classification)
