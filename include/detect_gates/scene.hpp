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

// Build a DronePose from a position and an orientation quaternion, scalar
// first: (w, x, y, z). Frames are REP-103 as everywhere else here -- world ENU
// (x east, y north, z up), body FLU (x forward, y left, z up) -- so this is a
// pure change of parameterization and the result renders identically to the
// equivalent roll/pitch/yaw.
//
// Note ROS's geometry_msgs/Quaternion orders its *fields* x, y, z, w; this
// takes w first. The quaternion need not be normalized.
//
// The rotation is decomposed back into roll/pitch/yaw, which is exact to
// ~3e-14 rad, degrading to ~7e-8 rad only when the pitch is within a
// micro-radian of straight up or down. Both are far below one pixel at any
// range this renders.
DronePose poseFromQuaternion(double x, double y, double z, double qw, double qx, double qy, double qz);

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
    // `detectGates` leaves this at the render resolution it was given.
    // `GateRenderer::renderDetections` then resamples it to the configured
    // output size, so every field of a detection handed back by GateRenderer
    // -- keypoints, boundingBox, mask, maskBoundingBox -- indexes the same
    // grid as `GateRenderer::render()`.
    //
    // All zeros when the gate projects nowhere. At `minVisibleCorners == 0`
    // such gates are dropped rather than returned, so a detection you get back
    // from that setting always has something in its mask.
    cv::Mat mask;

    // Bounding box of `mask`, inclusive of both corners. Unlike `boundingBox`
    // it follows the curved silhouette and stays meaningful with no visible
    // corner. (inf, inf, -inf, -inf) when the mask has no set pixel.
    BoundingBox maskBoundingBox;
};

// Bounding box of a CV_8UC1 mask's set pixels, INCLUSIVE of both corners --
// (x1, y1) and (x2, y2) are themselves set, so a single lit pixel gives
// x1 == x2. Note this differs from `GateDetection::boundingBox`, which is a
// continuous min/max over projected keypoint coordinates rather than a pixel
// index; a mask has no sub-pixel corners to span.
// (inf, inf, -inf, -inf) when no pixel is set, matching `boundingBox`'s
// convention for "nothing to describe".
BoundingBox boundingBoxOfMask(const cv::Mat& mask);

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

// Render the INSTANCE mask seen from `dronePos`: 0 for background, otherwise
// the gate's 1-based position in `gates` iteration order (i.e. sorted by gate
// name, which is what `gateNames()` returns). Pixel-for-pixel consistent with
// `renderPose`, because both rasterize the same per-gate silhouettes.
//
// Where two gates overlap the nearer one owns the pixel, resolved by mean
// camera-frame depth of the gate centre. `renderPose` needs no such rule: it
// OR-s, and OR does not care who contributed.
cv::Mat renderPoseInstances(const std::map<std::string, GatePose>& gates, const GateDims& gateDims,
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
