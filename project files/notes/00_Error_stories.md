# Dependency Problems

- There can be error with the dependencies for controller - Then you have to install the depenencies needed either due to ros index or with git clone and building them
- Example problem with package tf_conversions - dependency in ur-kinematics
- I search the package in the web and find out that it is in the ros index so i used `sudo apt-get install ros-noetic-tf-conversions` for installing it
- Some times it still doesn't workj becaus apt (packagemanager) is not updated so you have to run `sudo apt-get update` first

- Exit code 134 Gazebo gui doesn't work:

```bash
sudo apt-get update
sudo apt-get install -y x11-apps
xclock #(optional einfach testen ob die xclock angezeigt wird)
export DISPLAY=<IP>:0 #(einfach mit ipconfig deine IPV4 Addresse rein)
```

- Exit code 127 : env: ‘python\r’: No such file or directory - liegt daran wenn python files in windows geschrieben und dann in ubunt ausgeführt werden
- Zeilen umbruch muss LF sein!! (einfach in VS code umstellen oder)

```bash
sudo apt-get install dos2unix
dos2unix /home/fhtw_user/catkin_ws/src/fhtw/data_generator/scripts/generator_node.py
```
