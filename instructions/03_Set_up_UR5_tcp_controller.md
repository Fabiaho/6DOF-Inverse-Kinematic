Tutorial: https://roboticscasual.com/tutorial-controlling-tcp-position-of-the-ur5-robot-with-kdl-in-c-inverse-kinematics/
    - (Please read this for details)

# This are the commands for setting up the UR5 TCP Controller in Ros
- Start Docker container (run_docker_from_hub.bat) 
- Start X Server 
- In container Terminal:
    ##### Clone Repo as package
    - cd ~/catkin_ws/src/fhtw
    - git clone https://github.com/dairal/ur5-tcp-position-control.git
    ##### Build workspace
    - cd ../..
    - catkin_make
    - source devel/setup.bash
    - rospack list | grep ur5-tcp-position-control (ensure that config is correctly installed and recognized by ROS)
    ### Launch gazebo world with arm and joint controller
    - roslaunch ur5-joint-position-control ur5_gazebo_joint_position_control.launch
    ### Launch gazebo tcp controller
    - rosrun ur5-tcp-position-control  tcp_position_controller
