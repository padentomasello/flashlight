cmake_minimum_required(VERSION 3.10)

target_sources(
  flashlight-app-vision
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ImageDataset.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ImagenetUtils.cpp
)
