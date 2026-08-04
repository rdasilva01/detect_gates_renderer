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

// How a mask is resampled from the render resolution down to the output
// resolution. Unused when `OutputSettings::nativeInter` is set.
enum class InterMethod { Nearest, Linear, Area };

// Output mask resolution, from config.yaml. Entirely optional: leave
// `output_width`/`output_height` out and masks come out at the camera
// calibration's resolution, exactly as before this existed.
struct OutputSettings {
    int width = 0;  // <= 0 means "whatever the camera calibration says"
    int height = 0;
    InterMethod interMethod = InterMethod::Area;
    // Rasterize straight at the output resolution, with the intrinsics scaled
    // to match, instead of rendering at the calibration resolution and
    // resampling. Much faster, but a rasterizer only answers yes/no per pixel,
    // so it cannot represent a gate frame thinner than one output pixel as
    // anything but a whole one.
    bool nativeInter = false;
};

struct CameraCalibration {
    int imageWidth = 0;
    int imageHeight = 0;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    // Transform from the camera frame to the drone base frame.
    Transform tBaseCam;
    // Parsed from `distortion_model` ("fisheye"/"equidistant" -> true,
    // anything else, e.g. "radtan"/"plumb_bob" -> false). Defaults to true
    // (missing field) to match every calibration this library shipped with
    // before this field existed.
    bool fisheye = true;
};

struct Keypoint {
    std::string name;
    double x = 0.0, y = 0.0;
    bool visible = false;
    // True if this corner is within the camera's field of view at all (the
    // `theta < thetaMax` cone and the image bounds), regardless of occlusion.
    // When false, (x, y) is a real pixel location the corner would occupy on
    // an unbounded sensor, but one the camera cannot see.
    bool inFrustum = false;
};

struct BoundingBox {
    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
};

struct GateDetection {
    std::string gate;  // source gate's config name (ground truth identity)
    BoundingBox boundingBox;
    std::vector<Keypoint> keypoints;  // 8: 4 *_inner + 4 *_outer, canonical order
};

// Load a gates_config.yaml's `gates_poses` map (name -> [x, y, z, yaw]).
std::map<std::string, GatePose> loadGatesConfig(const std::string& path);

// Load a config.yaml's `gate_dimensions` entry.
GateDims loadGateDims(const std::string& path);

// Load a config.yaml's optional `output_width`, `output_height`,
// `inter_method` (nearest|linear|area) and `native_inter` entries.
// Missing keys leave `OutputSettings`'s defaults in place.
OutputSettings loadOutputSettings(const std::string& path);

// Load a ROS2-style camera_calibration.yaml, unwrapping the `/**: ros__parameters` namespace.
CameraCalibration loadCameraCalibration(const std::string& path);

// `thetaMax` below is the camera's angular extent: the FOV clipping cone for
// the pinhole model, and for the fisheye model the bound on which corners
// count as visible. The 89 deg default suits neither camera in particular --
// pass `fisheyeThetaMax` / `rectifiedThetaMax` (projection.hpp) for the real
// value, as `GateRenderer` does. It matters most for a >180 deg fisheye, where
// the true extent is past 90 deg and the default would discard the outer ring
// of the image.

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
