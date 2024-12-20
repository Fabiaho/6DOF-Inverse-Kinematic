#include "ros/ros.h"
#include "geometry_msgs/Pose.h"
#include "std_msgs/Float64MultiArray.h"
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/jntarray.hpp>
#include <iostream>
#include <random>

const int Joints = 6;

int main(int argc, char **argv) {
    ros::init(argc, argv, "pose_publisher");
    ros::NodeHandle nh;

    // Publisher for joint angles and TCP pose
    ros::Publisher pose_pub = nh.advertise<std_msgs::Float64MultiArray>("arm_pose_topic", 1000);

    // Load URDF and parse KDL chain
    std::string urdf_path = ros::package::getPath("ur5-joint-position-control") + "/urdf/ur5_jnt_pos_ctrl.urdf";
    KDL::Tree ur5_tree;
    if (!kdl_parser::treeFromFile(urdf_path, ur5_tree)) {
        ROS_ERROR("Failed to construct kdl tree");
        return -1;
    }

    KDL::Chain ur5_chain;
    ur5_tree.getChain("base_link", "wrist_3_link", ur5_chain);
    KDL::ChainFkSolverPos_recursive fk_solver(ur5_chain);

    // Get user input for number of poses
    int num_poses;
    std::cout << "Enter the number of poses to generate: ";
    std::cin >> num_poses;

    KDL::JntArray joint_angles(Joints);
    ros::Rate loop_rate(10);

    for (int i = 0; i < num_poses; ++i) {
        // Generate random joint angles within typical ranges
        for (int j = 0; j < Joints; ++j) {
            joint_angles(j) = ((double)rand() / RAND_MAX) * 2 * M_PI - M_PI;
        }

        // Compute forward kinematics to get the TCP position
        KDL::Frame tcp_frame;
        fk_solver.JntToCart(joint_angles, tcp_frame);

        // Prepare data for publishing
        std_msgs::Float64MultiArray pose_msg;
        pose_msg.data.resize(12); // 6 joint angles + 3 TCP positions + 3 TCP orientations

        for (int j = 0; j < Joints; ++j) {
            pose_msg.data[j] = joint_angles(j);
        }

        pose_msg.data[6] = tcp_frame.p.x();  // TCP position x
        pose_msg.data[7] = tcp_frame.p.y();  // TCP position y
        pose_msg.data[8] = tcp_frame.p.z();  // TCP position z

        pose_msg.data[9] = tcp_frame.M(0, 0);  // TCP orientation x
        pose_msg.data[10] = tcp_frame.M(1, 0); // TCP orientation y
        pose_msg.data[11] = tcp_frame.M(2, 0); // TCP orientation z

        // Publish the message
        pose_pub.publish(pose_msg);

        ROS_INFO("Published Pose %d", i + 1);
        loop_rate.sleep();
    }

    ROS_INFO("Pose generation completed.");
    return 0;
}
