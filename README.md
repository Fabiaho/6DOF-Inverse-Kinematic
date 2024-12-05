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

## Architecture

### Architecture 1 (Stand: 05.12.):
![Deployment Diagram ROS](https://github.com/user-attachments/assets/f811b9b5-c677-4ff4-8f16-d7d731bf7a48)

