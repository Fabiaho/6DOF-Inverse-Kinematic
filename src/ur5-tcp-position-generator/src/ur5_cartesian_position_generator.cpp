#include "ros/ros.h"
#include <ros/package.h>
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
#include "control_msgs/JointControllerState.h"
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/jntarray.hpp>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

// Constants
const int Joints = 6;
KDL::JntArray current_joint_angles(Joints);

// Callback functions to read joint states
void get_joint_position(const control_msgs::JointControllerState::ConstPtr &msg, int joint_index) {
    current_joint_angles(joint_index) = msg->process_value;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "pose_generator");
    ros::NodeHandle nh;

    // Publishers to move joints
    ros::Publisher joint_pubs[Joints];
    joint_pubs[0] = nh.advertise<std_msgs::Float64>("/shoulder_pan_joint_position_controller/command", 1000);
    joint_pubs[1] = nh.advertise<std_msgs::Float64>("/shoulder_lift_joint_position_controller/command", 1000);
    joint_pubs[2] = nh.advertise<std_msgs::Float64>("/elbow_joint_position_controller/command", 1000);
    joint_pubs[3] = nh.advertise<std_msgs::Float64>("/wrist_1_joint_position_controller/command", 1000);
    joint_pubs[4] = nh.advertise<std_msgs::Float64>("/wrist_2_joint_position_controller/command", 1000);
    joint_pubs[5] = nh.advertise<std_msgs::Float64>("/wrist_3_joint_position_controller/command", 1000);

    // Subscribers to read joint states
    ros::Subscriber joint_subs[Joints];
    joint_subs[0] = nh.subscribe<control_msgs::JointControllerState>("/shoulder_pan_joint_position_controller/state", 1000, boost::bind(get_joint_position, _1, 0));
    joint_subs[1] = nh.subscribe<control_msgs::JointControllerState>("/shoulder_lift_joint_position_controller/state", 1000, boost::bind(get_joint_position, _1, 1));
    joint_subs[2] = nh.subscribe<control_msgs::JointControllerState>("/elbow_joint_position_controller/state", 1000, boost::bind(get_joint_position, _1, 2));
    joint_subs[3] = nh.subscribe<control_msgs::JointControllerState>("/wrist_1_joint_position_controller/state", 1000, boost::bind(get_joint_position, _1, 3));
    joint_subs[4] = nh.subscribe<control_msgs::JointControllerState>("/wrist_2_joint_position_controller/state", 1000, boost::bind(get_joint_position, _1, 4));
    joint_subs[5] = nh.subscribe<control_msgs::JointControllerState>("/wrist_3_joint_position_controller/state", 1000, boost::bind(get_joint_position, _1, 5));

    // Publisher for pose data
    ros::Publisher pose_pub = nh.advertise<std_msgs::Float64MultiArray>("arm_pose_topic", 1000);

    // Forward kinematics setup
    std::string urdf_path = ros::package::getPath("ur5-joint-position-control") + "/urdf/ur5_jnt_pos_ctrl.urdf";
    KDL::Tree ur5_tree;
    if (!kdl_parser::treeFromFile(urdf_path, ur5_tree)) {
        ROS_ERROR("Failed to construct kdl tree");
        return -1;
    }

    KDL::Chain ur5_chain;
    ur5_tree.getChain("base_link", "wrist_3_link", ur5_chain);
    KDL::ChainFkSolverPos_recursive fk_solver(ur5_chain);

    // Get number of poses from the user
    int num_poses;
    std::cout << "Enter the number of poses to generate: ";
    std::cin >> num_poses;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-M_PI, M_PI);

    ros::Rate loop_rate(10);

    for (int i = 0; i < num_poses; ++i) {
        // Generate and move to random joint angles
        for (int j = 0; j < Joints; ++j) {
            std_msgs::Float64 command;
            command.data = dis(gen);
            joint_pubs[j].publish(command);
        }

        // Wait for the robot to reach the position
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        ros::spinOnce();

        // Compute TCP pose
        KDL::Frame tcp_frame;
        fk_solver.JntToCart(current_joint_angles, tcp_frame);

        // Publish joint angles and TCP pose
        std_msgs::Float64MultiArray pose_msg;
        pose_msg.data.resize(12);

        for (int j = 0; j < Joints; ++j) {
            pose_msg.data[j] = current_joint_angles(j);
        }

        pose_msg.data[6] = tcp_frame.p.x();  // TCP position x
        pose_msg.data[7] = tcp_frame.p.y();  // TCP position y
        pose_msg.data[8] = tcp_frame.p.z();  // TCP position z

        pose_msg.data[9] = tcp_frame.M(0, 0);  // TCP orientation x
        pose_msg.data[10] = tcp_frame.M(1, 0); // TCP orientation y
        pose_msg.data[11] = tcp_frame.M(2, 0); // TCP orientation z

        pose_pub.publish(pose_msg);
        ROS_INFO("Published Pose %d", i + 1);

        loop_rate.sleep();
    }

    ROS_INFO("Pose generation completed.");
    return 0;
}
