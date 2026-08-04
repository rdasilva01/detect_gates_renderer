// Config loading and full-scene rendering: gates layout + drone pose -> segmentation mask.
#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "detect_gates/transforms.hpp"

namespace detect_gates {

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

struct Keypoint {
    std::string name;
    double x = 0.0, y = 0.0;
    bool visible = false;
    // True if this corner is within the camera's field of view at all (the
    // `theta < thetaMax` cone and the image bounds), regardless of
    // occlusion. When false, (x, y) is not a meaningful pixel location --
    // the fisheye projection formula is only well-defined inside that cone,
    // so a corner outside it can project to an arbitrary, unrelated pixel.
    bool inFrustum = false;
};

struct BoundingBox {
    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
};

struct GateDetection {
    std::string gate;  // source gate's config name (ground truth identity)
    // Spans the VISIBLE keypoints only, so it is empty (inf, inf, -inf, -inf)
    // for a gate that is fully occluded or has no corner in the frustum.
    BoundingBox boundingBox;
    std::vector<Keypoint> keypoints;  // 8: 4 *_inner + 4 *_outer, canonical order

    // This gate's own silhouette, CV_8UC1 with 0/255 -- exactly what
    // `renderPose` draws for this gate alone, before any other gate occludes
    // it. Already computed to test occlusion against the true shape; kept here
    // because two things cannot be recovered from the keypoints:
    //
    //   - a fisheye bows the gate's straight edges OUTSIDE the straight lines
    //     joining its corners (which is why the faces are subdivided before
    //     projection), so a box built from the 8 corners under-covers the real
    //     silhouette;
    //   - a gate can be close and off to one side such that every corner
    //     leaves the `theta < thetaMax` cone while its frame still crosses the
    //     image. It then has no usable keypoint at all, and the mask is the
    //     only truthful description of it.
    //
    // Always allocated at the full image size; all zeros when the gate
    // projects nowhere, which is the exact test for "this gate is not in the
    // picture at all".
    cv::Mat mask;

    // Bounding box of `mask`, inclusive of both corners. Unlike `boundingBox`
    // it follows the curved silhouette and stays meaningful with no visible
    // corner. (inf, inf, -inf, -inf) when the mask has no set pixel.
    BoundingBox maskBoundingBox;
};

// Load a gates_config.yaml's `gates_poses` map (name -> [x, y, z, yaw]).
std::map<std::string, GatePose> loadGatesConfig(const std::string& path);

// Load a config.yaml's `gate_dimensions` entry.
GateDims loadGateDims(const std::string& path);

// Load a ROS2-style camera_calibration.yaml, unwrapping the `/**: ros__parameters` namespace.
CameraCalibration loadCameraCalibration(const std::string& path);

// Render the segmentation mask seen from `dronePos`.
cv::Mat renderPose(const std::map<std::string, GatePose>& gates, const GateDims& gateDims,
                    const DronePose& dronePos, const Transform& tBaseCam, const cv::Mat& cameraMatrix,
                    const cv::Mat& distCoeffs, int imageWidth, int imageHeight, bool fisheye = true,
                    double thetaMax = 89.0 * CV_PI / 180.0);

// Detect per-gate keypoints (4 inner + 4 outer corners) and bounding boxes
// as seen from `dronePos`, with cross-gate occlusion handling. Gates with
// fewer than `minVisibleCorners` visible keypoints (before or after
// occlusion) are omitted. Survivors are returned nearest-camera-first.
std::vector<GateDetection> detectGates(const std::map<std::string, GatePose>& gates, const GateDims& gateDims,
                                        const DronePose& dronePos, const Transform& tBaseCam,
                                        const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, int imageWidth,
                                        int imageHeight, bool fisheye = true,
                                        double thetaMax = 89.0 * CV_PI / 180.0, int minVisibleCorners = 3);

}  // namespace detect_gates
