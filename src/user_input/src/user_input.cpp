// ***************** INCLUDES ******************** //
#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
#include <iostream>
#include <string>
#include <vector>
#include <curl/curl.h>

// ***************** VARIABLES ******************** //
// ML Model ----- Constant
const std::string MODEL_API_URL = "https://huggingface.co/hange/inverseKinematicsSmall/resolve/main";


// ***************** FUNCTIONS ******************** //
/*
Send a TCP pose to the Hugging Face model and receive joint angles
*/
std::vector<double> queryModel(const std::vector<double> &tcp_pose) {
    // variable list
    std::vector<double> joint_angles;  // Vector to store the joint angles
    CURL *curl;
    CURLcode res;

    // init curl
    curl = curl_easy_init();
     if (curl) {
        // Create the JSON payload to send to the model
        std::string payload = "{\"tcp_pose\": [";
        for (size_t i = 0; i < tcp_pose.size(); ++i) {
            payload += std::to_string(tcp_pose[i]);  // Add each TCP pose value to the payload
            if (i != tcp_pose.size() - 1) payload += ",";
        }
        payload += "]}";  // Close the JSON array

        // VARS: Set CURL request
        curl_easy_setopt(curl, CURLOPT_URL, MODEL_API_URL.c_str());  // Set API URL
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());  // Add JSON payload to the POST request
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload.size());  // Specify payload size

        // Define a callback to capture the response from the API
        std::string response_string;
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](char *ptr, size_t size, size_t nmemb, std::string *data) {
            data->append(ptr, size * nmemb);  // Append received data to the response string
            return size * nmemb;  // Return processed bytes
        });
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);  // Pass  response string to callback

        res = curl_easy_perform(curl);
        if (res == CURLE_OK) {
            // Parse the response to extract joint angles
            size_t start = response_string.find("[");
            size_t end = response_string.find("]");
            if (start != std::string::npos && end != std::string::npos) {
                std::string angles = response_string.substr(start + 1, end - start - 1);  // Extract of joint angles
                size_t pos = 0;
                while ((pos = angles.find(",")) != std::string::npos) {
                    joint_angles.push_back(std::stod(angles.substr(0, pos)));  // Convert each angle to a double, add to the vector
                    angles.erase(0, pos + 1);  // Erase the processed value from the string
                }
                joint_angles.push_back(std::stod(angles));  // Add the last joint angle
            }
        } else {
            // Log error if API req failed
            ROS_ERROR("Failed to query model: %s", curl_easy_strerror(res));
        }

        curl_easy_cleanup(curl);  // Clean up CURL resources
    } else {
        ROS_ERROR("Failed to initialize CURL.");  // Log error CURL init fail
    }

    return joint_angles;  // Return the joint angles
}

// ***************** MAIN ******************** //
int main(int argc, char **argv) {
    ros::init(argc, argv, "user_input_node");  // Initialize the ROS node
    ros::NodeHandle nh;  // Create a NodeHandle

    // Publishers to send joint angle commands to the robot
    ros::Publisher joint_pubs[6];
    joint_pubs[0] = nh.advertise<std_msgs::Float64>("/shoulder_pan_joint_position_controller/command", 1000);
    joint_pubs[1] = nh.advertise<std_msgs::Float64>("/shoulder_lift_joint_position_controller/command", 1000);
    joint_pubs[2] = nh.advertise<std_msgs::Float64>("/elbow_joint_position_controller/command", 1000);
    joint_pubs[3] = nh.advertise<std_msgs::Float64>("/wrist_1_joint_position_controller/command", 1000);
    joint_pubs[4] = nh.advertise<std_msgs::Float64>("/wrist_2_joint_position_controller/command", 1000);
    joint_pubs[5] = nh.advertise<std_msgs::Float64>("/wrist_3_joint_position_controller/command", 1000);

    ros::Rate loop_rate(10);  // Define the loop rate (10 Hz)

    while (ros::ok()) {
        // Vector to store the user-provided TCP pose
        std::vector<double> tcp_pose(6);

        try {
            // Prompt the user for TCP pose input
            std::cout << "Enter desired TCP X position: ";
            std::cin >> tcp_pose[0];
            std::cout << "Enter desired TCP Y position: ";
            std::cin >> tcp_pose[1];
            std::cout << "Enter desired TCP Z position: ";
            std::cin >> tcp_pose[2];
            std::cout << "Enter desired TCP orientation X: ";
            std::cin >> tcp_pose[3];
            std::cout << "Enter desired TCP orientation Y: ";
            std::cin >> tcp_pose[4];
            std::cout << "Enter desired TCP orientation Z: ";
            std::cin >> tcp_pose[5];
        } catch (...) {
            // Handle invalid input gracefully
            ROS_ERROR("Invalid input. Please enter valid numbers.");
            continue;
        }

        ROS_INFO("Querying model with TCP pose...");  // Log the model query
        std::vector<double> joint_angles = queryModel(tcp_pose);  // Send TCP pose to the model and get joint angles

        if (!joint_angles.empty()) {
            // Log the received joint angles
            ROS_INFO("Received joint angles:");
            for (double angle : joint_angles) {
                std::cout << angle << " ";
            }
            std::cout << std::endl;

            // Publish the joint angles to the robot's controllers
            for (size_t i = 0; i < joint_angles.size(); ++i) {
                std_msgs::Float64 msg;
                msg.data = joint_angles[i];
                joint_pubs[i].publish(msg);
                ROS_INFO("Published joint %zu angle: %f", i + 1, joint_angles[i]);
            }
        }

        ros::spinOnce();  // Allow ROS callbacks to process
        loop_rate.sleep();  // Sleep to maintain the loop rate
    }

    return 0;
}