cmake_minimum_required(VERSION 3.0.2)
project(data_generator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

install(PROGRAMS
  scripts/generator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
