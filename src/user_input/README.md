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
pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cuda
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

# Run and Use 

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


2. Publishing Poses
    
   The user can choose between publishing TCP Poses with or without Orientation.
   Formats:

    - TCP Pose: 0.3, 0.1, 0.5
    - Orientation (roll, pitch, yaw) : 0.0, 1.57, 0.0 
      If you do not wish to proceed with Orientation then press "Enter" 

   The Node is publishing onto the same Topic as set in pose_listener_node, therefore make sure this node is running!


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
