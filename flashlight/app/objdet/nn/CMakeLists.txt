cmake_minimum_required(VERSION 3.5.1)

target_sources(
  flashlight-app-obj-det
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Detr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Transformer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PositionalEmbeddingSine.cpp
)

