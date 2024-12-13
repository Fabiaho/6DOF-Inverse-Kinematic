Please Read this for details: https://roboticscasual.com/ros-tutorial-control-the-ur5-robot-with-ros_control-tuning-a-pid-controller/


# This are the commands for setting up the UR5 Joint Controller in Ros
- Start Docker container (run_docker_from_hub.bat) 
- Start X Server 
- In container Terminal:
    ##### Clone Repo as package
    - cd ~/catkin_ws/src/fhtw
    - git clone https://github.com/dairal/ur5-joint-position-control.git
    ##### Install all needed Dependencies
    - cd ~/catkin_ws (back on catkin_ws level)
    - rosdep update
    - rosdep install --from-paths src --ignore-src -r -y
    ##### Build workspace
    - catkin_make
    - source devel/setup.bash
    - rospack list | grep ur5-joint-position-control (ensure that config is correctly installed and recognized by ROS)
    ##### Add Ros controllers
    - sudo apt-get install ros-noetic-ros-control ros-noetic-ros-controllers
    ##### Launching the UR5 in Gazebo with ros_control
    - roslaunch gazebo_ros empty_world.launch

    - rosrun gazebo_ros spawn_model -file `rospack find ur5-joint-position-control`/urdf/ur5_jnt_pos_ctrl.urdf -urdf -x 0 -y 0 -z 0.1 -model ur5
    - roslaunch ur5-joint-position-control ur5_joint_position_control.launch



    
    ##### Run UR5 with moveit with gazebo simualtion
    - (please make sure your X Server is running with the required Xlaunch config.)
    - roslaunch ur_gazebo ur5_bringup.launch (This will start your gazebo setup)
    - roslaunch ur5_moveit_config demo.launch (This will start your demo moveit demo)
