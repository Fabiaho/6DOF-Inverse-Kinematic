#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include <fstream>

std::ofstream data_file;

void poseCallback(const std_msgs::Float64MultiArray::ConstPtr &msg) {
    // Write joint angles and TCP pose to the CSV file
    data_file << msg->data[0] << "," << msg->data[1] << "," << msg->data[2] << ","
              << msg->data[3] << "," << msg->data[4] << "," << msg->data[5] << ","
              << msg->data[6] << "," << msg->data[7] << "," << msg->data[8] << ","
              << msg->data[9] << "," << msg->data[10] << "," << msg->data[11] << "\n";

    ROS_INFO("Received and logged pose.");
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "pose_listener");
    ros::NodeHandle nh;

    // Open the CSV file
    data_file.open("arm_pose_data.csv");
    data_file << "Shoulder_Pan,Shoulder_Lift,Elbow,Wrist_1,Wrist_2,Wrist_3,"
              << "TCP_X,TCP_Y,TCP_Z,Orientation_X,Orientation_Y,Orientation_Z\n";

    // Subscribe to the arm_pose_topic
    ros::Subscriber pose_sub = nh.subscribe("arm_pose_topic", 1000, poseCallback);

    ROS_INFO("Listening for poses...");
    ros::spin();

    // Close the CSV file
    data_file.close();
    ROS_INFO("Data collection completed.");
    return 0;
}
