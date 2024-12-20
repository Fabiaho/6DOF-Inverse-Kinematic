cmake_minimum_required(VERSION 3.0.2)
project(pose_listener)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)

## Declare a catkin package
catkin_package()

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare the listener executable
add_executable(pose_listener_node src/pose_listener.cpp)

## Specify libraries to link the executable against
target_link_libraries(pose_listener_node
  ${catkin_LIBRARIES}
)
