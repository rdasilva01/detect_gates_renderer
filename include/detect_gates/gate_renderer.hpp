// High-level entry point: load configs once, then render masks for many poses.
#pragma once

#include <map>
#include <string>

#include <opencv2/core.hpp>

#include "detect_gates/scene.hpp"

namespace detect_gates {

// Loads a gates layout, gate dimensions, and camera calibration once, then
// renders segmentation masks for arbitrary drone poses without re-parsing
// config files on every call.
class GateRenderer {
public:
    // `droneConfigPath` only needs to provide `gate_dimensions`; each
    // `render()` call supplies its own pose.
    //
    // If `rectified` is true, masks are rendered as seen by a rectified
    // (pinhole) view of the fisheye camera (matching the `-r` example flag)
    // instead of the raw fisheye projection.
    GateRenderer(const std::string& gatesConfigPath, const std::string& droneConfigPath,
                 const std::string& cameraConfigPath, bool rectified = false);

    cv::Mat render(const DronePose& pose) const;
    cv::Mat render(double x, double y, double z, double roll, double pitch, double yaw) const;

    // Detect per-gate keypoints/bounding boxes for a pose instead of a mask.
    // See `detectGates()` in scene.hpp for the algorithm.
    std::vector<GateDetection> renderDetections(const DronePose& pose, int minVisibleCorners = 3) const;
    std::vector<GateDetection> renderDetections(double x, double y, double z, double roll, double pitch, double yaw,
                                                 int minVisibleCorners = 3) const;

    int imageWidth() const { return imageWidth_; }
    int imageHeight() const { return imageHeight_; }

private:
    std::map<std::string, GatePose> gates_;
    GateDims gateDims_;
    Transform tBaseCam_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    int imageWidth_ = 0;
    int imageHeight_ = 0;
    bool fisheye_ = true;
    double thetaMax_ = 89.0 * CV_PI / 180.0;
};

}  // namespace detect_gates
