import os
import rospkg
import rospy
from std_msgs.msg import Float64MultiArray
import torch
from modules.modules import HyperNet, MainNet
import numpy as np
import ikpy.chain


class UserInputNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('user_input_node', anonymous=True)

        # Load pre-trained model
        self.config = self.load_config()
        self.hypernet, self.jointnet = self.load_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Publisher for joint angles
        self.joint_pub = rospy.Publisher('/calculated_joint_positions', Float64MultiArray, queue_size=10)

        # Subscriber for TCP poses
        rospy.Subscriber('/desired_tcp_pose', Float64MultiArray, self.callback)

    def load_config(self):
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('user_input')  # Replace 'user_input' with your actual package name if different

        class Config:
            chain_path = os.path.join(package_path, 'urdf', 'ur5_robot.urdf')
            train_data_path = os.path.join(package_path, 'data', 'ur5', 'ur5_train_data.csv')
            test_data_path = os.path.join(package_path, 'data', 'ur5', 'ur5_test_data.csv')
            num_joints = 6
            lr = 0.001
            num_epochs = 200
            num_solutions_validation = 10
            batch_size = 1024
            early_stopping_epochs = 30
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

    def load_model(self):
        cfg = self.config
        hypernet = HyperNet(cfg)
        jointnet = MainNet(cfg)

        # Use rospkg to resolve the absolute path to the model file
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('user_input')
        model_path = os.path.join(package_path, 'model/best_model.pt')

        rospy.loginfo(f"Loading model from {model_path}")

        # Load the model weights and map them to the CPU
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        hypernet.load_state_dict(checkpoint)
        # jointnet.load_state_dict(checkpoint['jointnet'])

        hypernet.eval()
        # jointnet.eval()
        return hypernet, jointnet

    def callback(self, msg):
        rospy.loginfo(f"Received input TCP pose: {msg.data}")

        try:
            # Ensure the input has 6 elements (3 for position, 3 for orientation)
            if len(msg.data) != 6:
                rospy.logerr("Invalid input! Expected 6 values: [X, Y, Z, roll, pitch, yaw].")
                return

            # Convert the input data to a PyTorch tensor
            tcp_pose = torch.tensor([msg.data], dtype=torch.float32)  # Wrap in a list to create a batch
            rospy.loginfo(f"Input tensor for model: {tcp_pose}")

            # Load the kinematic chain for FK calculations (optional)
            r_arm = ikpy.chain.Chain.from_urdf_file(self.config.chain_path)

            # Extract joint limits
            upper = []
            lower = []
            for i in range(1, len(r_arm.links) - 1):
                lower.append(r_arm.links[i].bounds[0])
                upper.append(r_arm.links[i].bounds[1])

            upper = np.array(upper)
            lower = np.array(lower)

            # Generate weights using the hypernet
            with torch.no_grad():
                weights = self.hypernet(tcp_pose)
                rospy.loginfo(f"Generated weights from hypernet: {weights}")

                # Use the jointnet to predict joint positions
                # joint_positions = self.jointnet(tcp_pose, weights).squeeze(0).tolist()
                initial_input = torch.ones((tcp_pose.shape[0], 1), dtype=torch.float32).to(self.device)
                samples, distributions, means, variance, selection = self.jointnet.validate(
                    initial_input, weights, lower, upper
                )

                predicted_joint_angles = []
                for sample in samples:
                    predicted_joint_angles.append([angle.item() for angle in sample])

                rospy.loginfo(f"Predicted joint positions: {predicted_joint_angles}")

            # Publish joint positions
            joint_msg = Float64MultiArray()

            # Flatten and convert to a flat list of floats
            flat_joint_angles = [float(angle) for joint_set in predicted_joint_angles for angle in joint_set]
            joint_msg.data = flat_joint_angles

            # Publish the message
            self.joint_pub.publish(joint_msg)
            rospy.loginfo(f"Published joint positions: {joint_msg.data}")


        except Exception as e:
            rospy.logerr(f"Error processing input: {e}")

    def run(self):
        rospy.loginfo('User Input Node is running...')
        rospy.spin()


if __name__ == '__main__':
    try:
        node = UserInputNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo('User Input Node terminated.')
