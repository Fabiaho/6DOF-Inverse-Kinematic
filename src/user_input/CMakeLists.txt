cmake_minimum_required(VERSION 3.0.2)
project(user_input)

# Find dependencies
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

# Declare catkin package
catkin_package(
  CATKIN_DEPENDS rospy std_msgs message_runtime
)

# Add Python package setup
catkin_python_setup()

# Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/user_input.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install modules directory
install(DIRECTORY modules/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}/user_input/modules
)
