#!/usr/bin/env python3
import rospy
import torch
from std_msgs.msg import Float64
import requests
import os

# Hugging Face Model URL for the .pt file
MODEL_URL = "https://huggingface.co/hange/inverseKinematicsSmall/resolve/main/model.pt"
MODEL_PATH = "/tmp/inverse_kinematics_model.pt"  # Temporary path to store the model


# Download the model if it doesn't exist locally
def download_model():
    if not os.path.exists(MODEL_PATH):
        rospy.loginfo(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            rospy.loginfo("Model downloaded successfully.")
        else:
            rospy.logerr(f"Failed to download model: {response.status_code} {response.text}")
            exit(1)


# Function to predict joint angles using the loaded model
def predict_joint_angles(tcp_pose):
    # Convert the input pose to a tensor
    tcp_pose_tensor = torch.tensor([tcp_pose], dtype=torch.float32)

    # Perform inference and return the predicted joint angles
    with torch.no_grad():
        joint_angles = model(tcp_pose_tensor).tolist()[0]
    return joint_angles


def user_input_node():
    rospy.init_node("user_input_node")  # Initialize the ROS node
    rate = rospy.Rate(10)  # Define the loop rate (10 Hz)

    # Publishers for each joint command
    joint_pubs = [
        rospy.Publisher("/shoulder_pan_joint_position_controller/command", Float64, queue_size=10),
        rospy.Publisher("/shoulder_lift_joint_position_controller/command", Float64, queue_size=10),
        rospy.Publisher("/elbow_joint_position_controller/command", Float64, queue_size=10),
        rospy.Publisher("/wrist_1_joint_position_controller/command", Float64, queue_size=10),
        rospy.Publisher("/wrist_2_joint_position_controller/command", Float64, queue_size=10),
        rospy.Publisher("/wrist_3_joint_position_controller/command", Float64, queue_size=10),
    ]

    while not rospy.is_shutdown():
        # Collect TCP pose input from the user
        try:
            print("\n--- Input Desired TCP Pose ---")
            tcp_x = float(input("Enter desired TCP X position: "))
            tcp_y = float(input("Enter desired TCP Y position: "))
            tcp_z = float(input("Enter desired TCP Z position: "))
            orientation_x = float(input("Enter desired TCP orientation X (e.g., roll): "))
            orientation_y = float(input("Enter desired TCP orientation Y (e.g., pitch): "))
            orientation_z = float(input("Enter desired TCP orientation Z (e.g., yaw): "))

            # Form the TCP pose list
            tcp_pose = [tcp_x, tcp_y, tcp_z, orientation_x, orientation_y, orientation_z]
            rospy.loginfo(f"TCP Pose: {tcp_pose}")
        except ValueError:
            rospy.logerr("Invalid input. Please enter numerical values.")
            continue

        # Predict joint angles using the loaded model
        rospy.loginfo("Predicting joint angles...")
        joint_angles = predict_joint_angles(tcp_pose)

        if joint_angles:
            rospy.loginfo(f"Predicted Joint Angles: {joint_angles}")

            # Publish the predicted joint angles to the ROS topics
            for i, joint_angle in enumerate(joint_angles):
                msg = Float64()
                msg.data = joint_angle
                joint_pubs[i].publish(msg)
                rospy.loginfo(f"Published joint {i + 1} angle: {joint_angle}")

        rate.sleep()


if __name__ == "__main__":
    try:
        # Ensure the model is downloaded
        download_model()

        # Load the model from the local file system
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()  # Set the model to evaluation mode

        rospy.loginfo("Model loaded successfully. Starting user input node...")
        user_input_node()
    except rospy.ROSInterruptException:
        pass
