// Config loading and full-scene rendering: gates layout + drone pose -> segmentation mask.
#pragma once

#include <map>
#include <string>

#include <opencv2/core.hpp>

#include "segment_gate/transforms.hpp"

namespace segment_gate {

struct GatePose {
    double x = 0.0, y = 0.0, z = 0.0, yaw = 0.0;
};

struct GateDims {
    double outerSize = 0.0;
    double innerSize = 0.0;
    double thickness = 0.0;
};

struct DronePose {
    double x = 0.0, y = 0.0, z = 0.0;
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
};

struct CameraCalibration {
    int imageWidth = 0;
    int imageHeight = 0;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    // Transform from the camera frame to the drone base frame.
    Transform tBaseCam;
};

// Load a gates_config.yaml's `gates_poses` map (name -> [x, y, z, yaw]).
std::map<std::string, GatePose> loadGatesConfig(const std::string& path);

// Load a config.yaml's `drone_position` and `gate_dimensions` entries.
std::pair<DronePose, GateDims> loadDroneConfig(const std::string& path);

// Load a ROS2-style camera_calibration.yaml, unwrapping the `/**: ros__parameters` namespace.
CameraCalibration loadCameraCalibration(const std::string& path);

// Render the segmentation mask seen from `dronePos`.
cv::Mat renderPose(const std::map<std::string, GatePose>& gates, const GateDims& gateDims,
                    const DronePose& dronePos, const Transform& tBaseCam, const cv::Mat& cameraMatrix,
                    const cv::Mat& distCoeffs, int imageWidth, int imageHeight, bool fisheye = true,
                    double thetaMax = 89.0 * CV_PI / 180.0);

}  // namespace segment_gate
