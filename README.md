# Project 3: Learning the inverse kinematics

## Short Description

Problem-Beschreibung:

Forward Kinematics (FK): Calculating the position and orientation of the robot's Tool Center Point (TCP) given joint angles.
Inverse Kinematics (IK): Finding the joint angles for a robot arm to achieve a specific TCP pose.

Using Machine Learning for IK: Instead of Maths ==> the model learns the relationship between TCP poses and joint angles from data

Steps and Components

- We need a 6-DoF robot model (Umiversal Robots UR5 hab ich gefunden, ist supported In Gazebo and Moveit)
- Collect Data: Training data which is data mapping join angles to TCP positions and orientations (input vs output)
- Develop the Machine Learning model which trains and predicts based on the data gathered
- Load the Train model and make a service in ROS that takes tcp pose and returns joint positions

How can Data Gathering be done ?

- We cannot simply take random positions as every robot as a specific range of movement (=certain degrees of freedom)
- We do still need to generate random joint positions which need to be restricted based on the robot models actual limits
- After each new joint => send them to robot => robot triggers a new tcp pos and this is then collected

Subscriber Architecture:

- Robot Joint Control:
  - Random joint angles are generated for all 6 joints within a predefined range.
  - These angles are published to the robot's controller topic (/arm_controller/command), which moves the robot to the desired pose.
- Data Collection:
  - While the robot moves to the new pose, its TCP position is published on the /tcp_pose topic.
  - The subscriber listens to this topic, retrieves the TCP position, and pairs it with the joint angles.

## How to use?

### ROS Docker

Start up the docker container.

We experienced issues with copying commands into the docker container after it was freshly booted, so we recommend reconnecting once.

If the container is running already you can connect to it via:

```bash
docker exec -it fhtw_ros bash
source /opt/ros/noetic/setup.bash
cd /home/fhtw_user/catkin_ws
source devel/setup.bash
tmux new-session
source devel/setup.bash
```

### Copy source code into ROS

The following code snippet needs to be executed outside of the ROS Container.

```bash
cp -r ./src/pose_listener/ ./Docker-ROS/catkin_ws/src
cp -r ./src/ur5-tcp-position-generator/ ./Docker-ROS/catkin_ws/src
cp -r ./src/user_input/ ./Docker-ROS/catkin_ws/src
```

### Setup

```bash
cd /home/fhtw_user/catkin_ws/src/fhtw
git clone https://github.com/ros-industrial/universal_robot.git -b noetic-devel
git clone https://github.com/dairal/ur5-joint-position-control.git

sudo apt-get update
sudo apt-get install -y ros-noetic-tf-conversions ros-noetic-ros-control ros-noetic-ros-controllers ros-noetic-rqt ros-noetic-rqt-common-plugins x11-apps

cd /home/fhtw_user/catkin_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ikpy

cd /home/fhtw_user/catkin_ws/src/fhtw/user_input
mkdir model
cd model
wget -O orientation_model.pt https://huggingface.co/hange/IK8/resolve/main/best_model.pt
wget -O no_orientation_model.pt https://huggingface.co/hange/IK_no_orientation_best_model/resolve/main/best_model.pt
cd ..
pip install -e .
cd /home/fhtw_user/catkin_ws

catkin_make
source devel/setup.bash
```

### Fix GUI Error

If the container is running for some time, it can be the case that your IPv4 Adress is outdated and Windows cannot be shown to you.

To fix that you need to update the environment variable in the docker image.

Do this in every new tmux session.

```bash
source devel/setup.bash
export DISPLAY=<IPv4>:0
```

### Start Project

We recommend using tmux windows within the started session to better manage multiple terminals.

#### Window 1 - roscore

```bash
roscore
```

#### Window 2 - Gazebo World

```bash
roslaunch gazebo_ros empty_world.launch
```

#### Window 3 - UR5 arm

```bash
rosrun gazebo_ros spawn_model -file `rospack find ur5-joint-position-control`/urdf/ur5_jnt_pos_ctrl.urdf -urdf -x 0 -y 0 -z 0.1 -model ur5
roslaunch ur5-joint-position-control ur5_joint_position_control.launch
```

#### Window 4 - Pose Listener

```bash
rosrun pose_listener pose_listener_node
```

#### Window 5 - Position Generator

```bash
rosrun ur5-tcp-position-generator  tcp_position_generator
```

#### Window 6 - User Input Node

```bash
rosrun user_input user_input.py
```

## Architecture

### Architecture 1 (Stand: 05.12.)

![Deployment Diagram ROS](https://github.com/user-attachments/assets/f811b9b5-c677-4ff4-8f16-d7d731bf7a48)
