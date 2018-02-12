ADD_EXECUTABLE(autograd_test test.cpp)

ADD_CUSTOM_COMMAND(
  TARGET autograd_test PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/mnist" "${CMAKE_CURRENT_BINARY_DIR}/mnist"
)

TARGET_LINK_LIBRARIES(autograd_test autogradpp)
