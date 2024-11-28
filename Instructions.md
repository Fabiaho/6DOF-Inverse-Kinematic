# Project Instructions

## Setting Up the Simulation Environment

Download and set up an industrial robot model in Gazebo, such as a 6-DOF arm (e.g., Universal
Robots UR5 or similar) with at least three joints. Ensure that the robot is configured with ROS, and
test that its ROS interfaces (e.g., /joint_states and /tcp_pose) are active and functional. Set up
controllers in ROS that allow control over individual joint angles. These controllers will enable you
to move the robot arm to different poses and collect data automatically.

## Data Collection

Develop a ROS node to automate the movement of each joint within a specified range, collecting
corresponding TCP poses for each unique joint configuration. Randomly vary joint angles within
their allowed ranges, capturing the TCP’s 3D position (and orientation, if required) for each joint
position.

## Developing the Machine Learning Model

Develop a machine learning model to predict joint positions based on the data. Demonstrate that
the model successfully learns inverse kinematics and applies it to a pick-and-place task within
ROS. Show the uncertainty in relation to the dataset size.
