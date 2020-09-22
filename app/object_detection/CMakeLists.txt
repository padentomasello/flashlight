cmake_minimum_required(VERSION 3.5.1)

add_library(
  flashlight-app-object-det
  ""
  )

# Find GLog
find_package(GLOG REQUIRED)
if (GLOG_FOUND)
  message(STATUS "GLOG found")
else()
  message(FATAL_ERROR "GLOG not found")
endif()

# Find GFlags
find_package(GFLAGS REQUIRED)
if (GFLAGS_FOUND)
  message(STATUS "GFLAGS found")
else()
  message(FATAL_ERROR "GFLAGS not found")
endif()

target_link_libraries(
  flashlight-app-object-det
  PUBLIC
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-app-object-det
  PUBLIC
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  )

# Dataset
include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)

# Criterion
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

# NN
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)

# TODO padentomasello have a "trainers" folder
add_executable(Detr ${CMAKE_CURRENT_LIST_DIR}/examples/Detr.cpp)
target_link_libraries(Detr flashlight-app-object-det)

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
