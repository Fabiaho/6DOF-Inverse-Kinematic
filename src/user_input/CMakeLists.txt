cmake_minimum_required(VERSION 3.0.2)
project(inverse_kinematics_node)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

## Declare a catkin package
catkin_package()

## Install Python files
install(PROGRAMS
  scripts/user_input_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Specify libraries to link with
link_directories(${catkin_LIBRARY_DIRS})

