CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

FIND_PACKAGE(PythonLibs REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(pytorch)
FIND_PACKAGE(ATen REQUIRED)

FILE(GLOB SRCS *.cpp)
ADD_LIBRARY(autograd STATIC ${SRCS})

ADD_CUSTOM_COMMAND(
  TARGET autograd PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/build/lib*/torch/_C.*.so" "${CMAKE_BINARY_DIR}/libpythpp.so"
)
ADD_CUSTOM_COMMAND(
  TARGET autograd PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/build/lib*/torch/_dl.*.so" "${CMAKE_BINARY_DIR}/libpythpp_dl.so"
)
ADD_CUSTOM_COMMAND(
  TARGET autograd PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pytorch/build/lib*/torch/_nvrtc.*.so" "${CMAKE_BINARY_DIR}/libpythpp_nvrtc.so"
)

TARGET_LINK_LIBRARIES(autograd ATen ${PYTHON_LIBRARIES}
    "${CMAKE_BINARY_DIR}/libpythpp.so"
    "${CMAKE_BINARY_DIR}/libpythpp_dl.so"
    "${CMAKE_BINARY_DIR}/libpythpp_nvrtc.so"
    cuda
    nvrtc 
    )
ADD_EXECUTABLE(autograd_test test.cpp)
TARGET_LINK_LIBRARIES(autograd_test ATen autograd nvrtc cuda)
