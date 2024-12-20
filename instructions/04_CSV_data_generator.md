# How to run the data generator package

- Start Docker container (run_docker_from_hub.bat)
- Start X Server
- In container Terminal:

## Build workspace

```bash
cd ../.. #(catkin-folder)
catkin_make
source devel/setup.bash
rospack list | grep data_generator #(ensure that config is correctly installed and recognized by ROS)
```

## Launch the generator

```bash
roslaunch data_generator generator.launch
```

## Test with some topics (soon depricated)

```bash
rostopic pub /example_float_topic std_msgs/Float64 "data: 1.23"
rostopic pub /example_string_topic std_msgs/String "data: 'Hallo ROS'"
```

## Pandas install

Üblicherweise installiert man Python-Abhängigkeiten wie pandas außerhalb von package.xml, z.B. durch

```bash
sudo apt-get install python3-pandas
python3 -m pip install --user pandas
python3 -m pip install --user --upgrade numpy pandas
```
