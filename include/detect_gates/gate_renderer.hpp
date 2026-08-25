// High-level entry point: load configs once, then render masks for many poses.
#pragma once

#include <map>
#include <string>
#include <vector>

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
    // `cameraConfigPath`'s `distortion_model` field ("fisheye"/"equidistant"
    // vs. anything else, e.g. "radtan"/"plumb_bob") selects the projection
    // model. If `rectified` is true, masks are rendered undistorted: for a
    // fisheye source that means a rectified (pinhole) view derived from it
    // (matching the `-r` example flag); for an already-pinhole source it
    // means the same camera matrix with distortion coefficients dropped.

    GateRenderer(const std::string& gatesConfigPath, const std::string& droneConfigPath,
                 const std::string& cameraConfigPath, bool rectified = false);

    // Returns an `imageHeight() x imageWidth()` CV_8UC1 mask.
    //
    // Values are 0 or 255, *except* when config.yaml asks for an output
    // resolution reached by resampling with `inter_method: area` or `linear`:
    // those blend, so a gate frame thinner than an output pixel lands as a
    // partial value. Threshold it yourself if you need hard labels.
    cv::Mat render(const DronePose& pose) const;
    cv::Mat render(double x, double y, double z, double roll, double pitch, double yaw) const;

    // Semantic coverage and instance labels for one pose, together.
    //
    // `.coverage` is byte-for-byte what `render()` returns. `.instances` is 0
    // for background and otherwise the gate's 1-based index into `gateNames()`,
    // with the nearer gate owning any overlap.
    //
    // **The two are resampled by different rules and must be.** Coverage is a
    // measure and blends: `inter_method: area` is what turns a frame thinner
    // than an output pixel into a partial value, and that softness is the
    // point. A label is an identity and cannot blend -- averaging gate 3 and
    // gate 7 gives gate 5, a gate that is not there -- so instances always
    // resample nearest-neighbour regardless of `inter_method`.
    struct Segmentation {
        cv::Mat coverage;
        cv::Mat instances;
    };
    //
    // `instances > 0` holds exactly where `coverage > 0`: unowned coverage takes
    // the largest adjacent label and a label with no coverage under it is
    // dropped, so a consumer never meets a covered pixel it cannot attribute.
    Segmentation renderSegmented(const DronePose& pose) const;
    Segmentation renderSegmented(double x, double y, double z, double roll, double pitch, double yaw) const;

    // Gate names in label order, so `instances == i + 1` is `gateNames()[i]`.
    // Config order is alphabetical (the gates live in a std::map), which need
    // not be track order -- the caller maps names, never indices.
    std::vector<std::string> gateNames() const;

    // Detect per-gate keypoints/bounding boxes for a pose instead of a mask.
    // See `detectGates()` in scene.hpp for the algorithm. Coordinates are in
    // output-resolution pixels, so they line up with `render()`'s mask.
    std::vector<GateDetection> renderDetections(const DronePose& pose, int minVisibleCorners = 3) const;
    std::vector<GateDetection> renderDetections(double x, double y, double z, double roll, double pitch, double yaw,
                                                 int minVisibleCorners = 3) const;

    // Size of what `render()` hands back: config.yaml's `output_width` /
    // `output_height` if set, else the camera calibration's resolution.
    int imageWidth() const { return outputWidth_; }
    int imageHeight() const { return outputHeight_; }

private:
    std::map<std::string, GatePose> gates_;
    GateDims gateDims_;
    Transform tBaseCam_;
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;
    // Resolution the rasterizer works at. Equal to the output resolution
    // unless the mask is rendered large and resampled down.
    int renderWidth_ = 0;
    int renderHeight_ = 0;
    int outputWidth_ = 0;
    int outputHeight_ = 0;
    InterMethod interMethod_ = InterMethod::Area;
    bool resample_ = false;
    bool fisheye_ = true;
    double thetaMax_ = 89.0 * CV_PI / 180.0;
};

}  // namespace detect_gates
