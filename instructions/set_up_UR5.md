# This are the commands for setting up the UR5 industrial robot in ROS
- Start container 
- In container Terminal:
    ##### Clone Repo as package
    - cd ~/catkin_ws/src/fhtw
    - git clone https://github.com/ros-industrial/universal_robot.git -b noetic-devel
    ##### Install all needed Dependencies
    - cd ~/catkin_ws (back on catkin_ws level)
    - rosdep update
    - rosdep install --from-paths src --ignore-src -r -y
    ##### Build workspace
    - catkin_make
    - source devel/setup.bash
    - rospack list | grep ur5_moveit_config (ensure that config is correctly installed and recognized by ROS: Expected output = ur5_moveit_config /home/fhtw_user/catkin_ws/src/fhtw/universal_robot/ur5_moveit_config)
    - rospack list | grep ur5_gazebo (ensure that gazebo is correctly installed and recognized by ROS: Expected output = ur5_moveit_config /home/fhtw_user/catkin_ws/src/fhtw/universal_robot/ur5_moveit_config)
    ##### Run UR5 with moveit with gazebo simualtion
    - (please make sure your X Server is running with the required Xlaunch config.)
    - roslaunch ur_gazebo ur5_bringup.launch (This will start your gazebo setup)
    - roslaunch ur5_moveit_config demo.launch (This will start your demo moveit demo)





