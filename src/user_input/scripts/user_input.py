import os
import rospkg
import rospy
from std_msgs.msg import Float64
from control_msgs.msg import JointControllerState
import torch
from modules.orientation.modules import HyperNet, MainNet
from modules.without.modules import HyperNet as HyperNetNo, MainNet as MainNetNo
import numpy as np
import ikpy.chain


class UserInputNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('user_input_node', anonymous=True)

        # Load configurations for two models
        self.config_model_1 = self.load_config('orientation_model.pt')  # with orientation
        self.config_model_2 = self.load_config_no('no_orientation_model.pt')

        # Load models
        self.hypernet_1, self.jointnet_1 = self.load_model(self.config_model_1)
        self.hypernet_2, self.jointnet_2 = self.load_model_no(self.config_model_2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Publishers for specific joints
        self.joint_pubs = [
            rospy.Publisher("/shoulder_pan_joint_position_controller/command", Float64, queue_size=10),
            rospy.Publisher("/shoulder_lift_joint_position_controller/command", Float64, queue_size=10),
            rospy.Publisher("/elbow_joint_position_controller/command", Float64, queue_size=10),
            rospy.Publisher("/wrist_1_joint_position_controller/command", Float64, queue_size=10),
            rospy.Publisher("/wrist_2_joint_position_controller/command", Float64, queue_size=10),
            rospy.Publisher("/wrist_3_joint_position_controller/command", Float64, queue_size=10),
        ]

        # Subscriber for joint states
        self.joint_states = [0.0] * 6
        self.joint_subs = [
            rospy.Subscriber("/shoulder_pan_joint_position_controller/state", JointControllerState, self.update_joint_state, 0),
            rospy.Subscriber("/shoulder_lift_joint_position_controller/state", JointControllerState, self.update_joint_state, 1),
            rospy.Subscriber("/elbow_joint_position_controller/state", JointControllerState, self.update_joint_state, 2),
            rospy.Subscriber("/wrist_1_joint_position_controller/state", JointControllerState, self.update_joint_state, 3),
            rospy.Subscriber("/wrist_2_joint_position_controller/state", JointControllerState, self.update_joint_state, 4),
            rospy.Subscriber("/wrist_3_joint_position_controller/state", JointControllerState, self.update_joint_state, 5),
        ]

    def load_config_no(self, model_filename):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('user_input')

        class Config_no:
            chain_path = os.path.join(package_path, 'urdf', 'ur5_robot.urdf')
            model_path = os.path.join(package_path, 'model', model_filename)
            num_joints = 6
            lr = 0.001
            num_epochs = 200
            num_solutions_validation = 10
            batch_size = 1024
            early_stopping_epochs = 30
            grad_clip = 1
            embedding_dim = 128
            hypernet_input_dim = 3
            hypernet_hidden_size = 1024
            hypernet_num_hidden_layers = 3
            jointnet_hidden_size = 256
            num_gaussians = 50
            jointnet_output_dim = 2 if 50 == 1 else 50 * 2 + 50

        return Config_no()

    def load_config(self, model_filename):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('user_input')

        class Config:
            chain_path = os.path.join(package_path, 'urdf', 'ur5_robot.urdf')
            model_path = os.path.join(package_path, 'model', model_filename)
            num_joints = 6
            lr = 0.001
            num_epochs = 150
            num_solutions_validation = 10
            batch_size = 2048
            early_stopping_epochs = 50
            grad_clip = 1
            embedding_dim = 128
            hypernet_input_dim = 6
            hypernet_hidden_size = 1024
            hypernet_num_hidden_layers = 3
            jointnet_hidden_size = 256
            num_gaussians = 50
            exp_dir = os.path.join(package_path, 'runs', 'exp_12')
            jointnet_output_dim = 2 if 50 == 1 else 50 * 2 + 50

        return Config()

    def load_model(self, cfg):
        hypernet = HyperNet(cfg)
        jointnet = MainNet(cfg)

        rospy.loginfo(f"Loading model from {cfg.model_path}")
        checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'))
        hypernet.load_state_dict(checkpoint)
        hypernet.eval()
        return hypernet, jointnet

    def load_model_no(self, cfg):
        hypernet = HyperNetNo(cfg)
        jointnet = MainNetNo(cfg)

        rospy.loginfo(f"Loading model from {cfg.model_path}")
        checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'))
        hypernet.load_state_dict(checkpoint)
        hypernet.eval()
        return hypernet, jointnet

    def update_joint_state(self, msg, joint_index):
        self.joint_states[joint_index] = msg.process_value

    def calculate_joint_angles(self, tcp_pose, model_type):
        r_arm = ikpy.chain.Chain.from_urdf_file(self.config_model_1.chain_path)

        upper, lower = [], []
        for i in range(1, len(r_arm.links) - 1):
            lower.append(r_arm.links[i].bounds[0])
            upper.append(r_arm.links[i].bounds[1])

        upper, lower = np.array(upper), np.array(lower)

        hypernet, jointnet = (
            (self.hypernet_1, self.jointnet_1) if model_type == '1' else (self.hypernet_2, self.jointnet_2)
        )

        with torch.no_grad():
            weights = hypernet(tcp_pose)
            initial_input = torch.ones((tcp_pose.shape[0], 1), dtype=torch.float32).to(self.device)
            samples, *_ = jointnet.validate(initial_input, weights, lower, upper)

        predicted_joint_angles = [sample.tolist() for sample in samples]
        rospy.loginfo(f"Predicted joint angles: {predicted_joint_angles}")
        return predicted_joint_angles  # whole batch

    def publish_joint_angles(self, joint_angles_batch):
        for joint_angles in joint_angles_batch:
            # Flatten nested lists if necessary
            if isinstance(joint_angles[0], list):
                joint_angles = [angle for sublist in joint_angles for angle in sublist]

            # Publish joint commands
            for i, angle in enumerate(joint_angles):
                msg = Float64()
                msg.data = float(angle)  # Convert to float
                self.joint_pubs[i].publish(msg)

            # Wait for the robot to reach the position
            rospy.sleep(2.0)

        rospy.loginfo("Only joint angles are being published.")

    def run(self):
        rospy.loginfo("User Input Node is running...")

        while not rospy.is_shutdown():
            try:
                # Step 1: Choose model type
                model_type = input("Select model type (1: Model 1 (Orientation), 2: Model 2 (No Orientation)): ")
                if model_type not in ['1', '2']:
                    rospy.logerr("Invalid model type. Choose 1 or 2.")
                    continue

                # Step 2: Enter TCP position
                tcp_input = input("Enter TCP position as x, y, z: ")
                try:
                    x, y, z = map(float, tcp_input.split(","))
                except ValueError:
                    rospy.logerr("Invalid TCP position input. Provide x, y, z as comma-separated floats.")
                    continue

                orientation_input = input("Enter orientation as roll, pitch, yaw (or press Enter to skip): ")
                if orientation_input.strip():
                    try:
                        roll, pitch, yaw = map(float, orientation_input.split(","))
                        tcp_pose = torch.tensor([[x, y, z, roll, pitch, yaw]], dtype=torch.float32)
                    except ValueError:
                        rospy.logerr("Invalid orientation input. Provide roll, pitch, yaw as comma-separated floats.")
                        continue
                else:
                    tcp_pose = torch.tensor([[x, y, z]], dtype=torch.float32)

                # Step 3: Calculate joint angles
                joint_angles = self.calculate_joint_angles(tcp_pose, model_type)

                # Step 4: Publish joint angles
                self.publish_joint_angles(joint_angles)
                rospy.loginfo("Arm movement command issued.")

            except rospy.ROSInterruptException:
                break


if __name__ == '__main__':
    try:
        node = UserInputNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("User Input Node terminated.")
