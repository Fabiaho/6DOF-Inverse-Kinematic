# Using the User Input Node

This documentation provides a step-by-step guide to set up and use the `User Input Node` for calculating robot joint angles using a pre-trained neural network model.

---

## Prerequisites

### 1. ROS Setup
Ensure your ROS environment is properly set up, and the `user_input` package is built and sourced.

### 2. Dependencies for Set Up of Node
#### PyTorch Installation
Verify PyTorch installation in your Docker container:
```bash
python3 -c "import torch; print(torch.__version__)"
```
If PyTorch is not installed, install it using the following command:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Further Libraries
Further Libraries might have to be installed in the docker container
- ikpy
```bash
pip install ikpy
```

#### Confirm Module Recognition
Ensure the custom modules directory is recognized as a Python module:
```bash
python3 -c "import user_input.modules.modules as mod; print(mod)"
```

#### Download the Pre-trained Model
The .pt file for the pre-trained model must be downloaded from the following source: https://huggingface.co/hange/inverseKinematicsSmall/tree/main

Save the best_model.pt file in the following directory within your package:
user_input/model/best_model.pt

If all the above steps are successful, proceed to run the node.

# Running the Node

After all the Prerequs. are installed and set up run the following commands in the terminal:
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash    
```

1. Start the user_input node:

```bash
rosrun user_input user_input.py
```

2. Open two additional terminals for publishing and listening to topics:

- Terminal 1: For publishing desired TCP poses.
- Terminal 2: For listening to the calculated joint positions.

3. Listening: Output of Calculated Join Positions

    The node will process the input and publish the calculated joint positions on the /calculated_joint_positions topic. The output will be a flat list of 6 joint angle values in radians.
    
    ```bash
    rostopic echo /calculated_joint_positions
   ```
    

4. Publishing Poses
    
    To send a desired TCP pose to the node, use the following example command:
    ```bash
    rostopic pub /desired_tcp_pose std_msgs/Float64MultiArray "data: [0.5, 0.3, 0.2, 0.0, 1.57, 0.0]"
    ```
   
    - Input Format:
        - The input message must contain 6 values:
          - Position (X, Y, Z): Cartesian coordinates of the tool center point.
          - Orientation (roll, pitch, yaw): Orientation of the tool in radians.


## File Structure

````
user_input/
├── data/
│   ├── ur5_train_data.csv
│   └── ur5_test_data.csv
├── model/
│   └── best_model.pt
├── modules/
│   ├── __init__.py
│   └── modules.py
├── scripts/
│   ├── __init__.py
│   └── user_input.py
├── urdf/
│   ├── ur5_robot.urdf
│   └── ur5_jnt_pos_ctrl.urdf
├── CMakeLists.txt
├── package.xml
└── setup.py
