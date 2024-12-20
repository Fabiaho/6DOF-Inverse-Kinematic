cd ~/catkin_ws/src/fhtw
git clone https://github.com/ros-industrial/universal_robot.git -b noetic-devel
git clone https://github.com/dairal/ur5-joint-position-control.git
git clone https://github.com/dairal/ur5-tcp-position-control.git

sudo apt-get update
sudo apt-get install ros-noetic-tf-conversions
sudo apt-get install ros-noetic-ros-control ros-noetic-ros-controllers
sudo apt-get install ros-noetic-rqt ros-noetic-rqt-common-plugins
sudo apt-get install -y x11-apps

cd ~/catkin_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y

catkin_make
source devel/setup.bash
