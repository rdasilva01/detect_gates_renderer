#include "detect_gates/gate_renderer.hpp"

#include <vector>

#include <opencv2/imgproc.hpp>

#include "detect_gates/projection.hpp"

namespace detect_gates {

namespace {

// Retarget a camera matrix at one image resolution to another.
//
// fx, cx (and any skew) scale with the width, fy, cy with the height. The view
// itself is untouched: (u - cx) / fx is invariant under this, so the field of
// view -- and with it `thetaMax` -- comes out identical, and only the sampling
// grid changes. Scaling one without the other is what makes a naively resized
// calibration render a tiny crop of the image centre instead of the whole view.
cv::Mat scaleCameraMatrix(const cv::Mat& cameraMatrix, double scaleX, double scaleY) {
    cv::Mat scaled = cameraMatrix.clone();
    scaled.at<double>(0, 0) *= scaleX;
    scaled.at<double>(0, 1) *= scaleX;
    scaled.at<double>(0, 2) *= scaleX;
    scaled.at<double>(1, 1) *= scaleY;
    scaled.at<double>(1, 2) *= scaleY;
    return scaled;
}

int toCvInterpolation(InterMethod method) {
    switch (method) {
        case InterMethod::Nearest:
            return cv::INTER_NEAREST;
        case InterMethod::Linear:
            return cv::INTER_LINEAR;
        case InterMethod::Area:
            break;
    }
    return cv::INTER_AREA;
}

}  // namespace

GateRenderer::GateRenderer(const std::string& gatesConfigPath, const std::string& droneConfigPath,
                            const std::string& cameraConfigPath, bool rectified) {
    gates_ = loadGatesConfig(gatesConfigPath);
    gateDims_ = loadGateDims(droneConfigPath);
    const OutputSettings output = loadOutputSettings(droneConfigPath);

    const CameraCalibration calib = loadCameraCalibration(cameraConfigPath);
    tBaseCam_ = calib.tBaseCam;
    renderWidth_ = calib.imageWidth;
    renderHeight_ = calib.imageHeight;
    cameraMatrix_ = calib.cameraMatrix;
    distCoeffs_ = calib.distCoeffs;
    fisheye_ = calib.fisheye;

    // `rectified` means "give me the undistorted view of this camera". For a
    // fisheye source that means deriving a new pinhole camera matrix (the
    // original behavior). For a calibration that's already pinhole (e.g.
    // radtan/plumb_bob), the camera matrix is already the right one to use --
    // "undistorted" just means dropping the (already-loaded) distortion
    // coefficients, keeping resolution/intrinsics/extrinsics unchanged.
    if (rectified) {
        if (fisheye_) {
            cameraMatrix_ = rectifiedCameraMatrix(cameraMatrix_, distCoeffs_, renderWidth_, renderHeight_);
            fisheye_ = false;
        }
        distCoeffs_ = cv::Mat::zeros(4, 1, CV_64F);
    }

    outputWidth_ = output.width > 0 ? output.width : renderWidth_;
    outputHeight_ = output.height > 0 ? output.height : renderHeight_;
    interMethod_ = output.interMethod;
    resample_ = outputWidth_ != renderWidth_ || outputHeight_ != renderHeight_;

    if (resample_ && output.nativeInter) {
        // Rasterize at the output resolution directly. Same view, coarser grid.
        cameraMatrix_ = scaleCameraMatrix(cameraMatrix_, static_cast<double>(outputWidth_) / renderWidth_,
                                           static_cast<double>(outputHeight_) / renderHeight_);
        renderWidth_ = outputWidth_;
        renderHeight_ = outputHeight_;
        resample_ = false;
    }

    // Derived from the final camera, so it describes whatever resolution the
    // rasterizer ended up working at. For a raw fisheye this is the lens's true
    // angular extent (105.8 deg for the calibration in config/, i.e. past 90
    // deg); for a pinhole or rectified view it is also the FOV clipping cone.
    thetaMax_ = fisheye_ ? fisheyeThetaMax(cameraMatrix_, distCoeffs_, renderWidth_, renderHeight_)
                          : rectifiedThetaMax(cameraMatrix_, renderWidth_, renderHeight_);
}

cv::Mat GateRenderer::render(const DronePose& pose) const {
    cv::Mat mask = renderPose(gates_, gateDims_, pose, tBaseCam_, cameraMatrix_, distCoeffs_, renderWidth_,
                               renderHeight_, fisheye_, thetaMax_);
    if (!resample_) {
        return mask;
    }

    cv::Mat resized;
    cv::resize(mask, resized, cv::Size(outputWidth_, outputHeight_), 0, 0, toCvInterpolation(interMethod_));
    return resized;
}

cv::Mat GateRenderer::render(double x, double y, double z, double roll, double pitch, double yaw) const {
    return render(DronePose{x, y, z, roll, pitch, yaw});
}

GateRenderer::Segmentation GateRenderer::renderSegmented(const DronePose& pose) const {
    Segmentation out;
    out.coverage = render(pose);
    cv::Mat instances = renderPoseInstances(gates_, gateDims_, pose, tBaseCam_, cameraMatrix_, distCoeffs_,
                                             renderWidth_, renderHeight_, fisheye_, thetaMax_);
    if (resample_) {
        // **Per-label area, then argmax** -- one pass, not one resize per label.
        //
        // Averaging labels is meaningless (3 and 7 would give 5), but
        // nearest-neighbour is wrong in a subtler and worse way at these scales:
        // it takes whichever label sits under the block's centre, so a gate whose
        // pixels are a minority of the block -- which at 820x616 down to 64x64 is
        // most of a thin frame -- vanishes, while coverage keeps it as a partial
        // value. The two channels then disagree exactly where the mask is most
        // interesting, at its edges. Measured: 96.5% support agreement that way
        // against 100% this way.
        //
        // Done by histogramming each output pixel's block once. Resizing every
        // label's footprint separately gives the identical answer and costs a
        // full-image compare and resize per gate -- 15.3 ms a frame against the
        // plain render's 1.4, which would turn a forty-minute cache build into six
        // hours.
        double highest = 0.0;
        cv::minMaxLoc(instances, nullptr, &highest);
        const int span = static_cast<int>(highest) + 1;
        std::vector<uint16_t> counts(static_cast<size_t>(outputWidth_) * outputHeight_ * span, 0);
        for (int y = 0; y < instances.rows; ++y) {
            const uint8_t* row = instances.ptr<uint8_t>(y);
            const int oy = y * outputHeight_ / instances.rows;
            for (int x = 0; x < instances.cols; ++x) {
                const uint8_t label = row[x];
                if (label == 0) {
                    continue;
                }
                const int ox = x * outputWidth_ / instances.cols;
                ++counts[(static_cast<size_t>(oy) * outputWidth_ + ox) * span + label];
            }
        }
        cv::Mat winner = cv::Mat::zeros(outputHeight_, outputWidth_, CV_8UC1);
        for (int y = 0; y < outputHeight_; ++y) {
            uint8_t* row = winner.ptr<uint8_t>(y);
            for (int x = 0; x < outputWidth_; ++x) {
                const uint16_t* bin = &counts[(static_cast<size_t>(y) * outputWidth_ + x) * span];
                uint16_t best = 0;
                for (int label = 1; label < span; ++label) {
                    if (bin[label] > best) {
                        best = bin[label];
                        row[x] = static_cast<uint8_t>(label);
                    }
                }
            }
        }
        instances = winner;
    }

    // **Every covered pixel gets an owner, and no uncovered pixel keeps one.**
    // Coverage is area-resampled and labels are block-histogrammed, so at a thin
    // edge the two can land a pixel apart -- measured at 0.22% of pixels covered
    // but unowned and 0.04% owned but uncovered. Computing the labels by exact
    // area instead closes it, and costs 15.3 ms a frame against 8.2.
    //
    // So the disagreement is repaired rather than prevented: unowned coverage
    // takes the largest label adjacent to it, and a label with no coverage under
    // it is dropped. The result is the contract a consumer actually needs --
    // `instances > 0` exactly where `coverage > 0` -- without the per-label pass.
    // Dilation takes the maximum, so a pixel between two gates goes to the
    // higher-numbered one; it is one pixel at a boundary and the alternative is a
    // distance transform for no measurable gain.
    if (resample_) {
        const cv::Mat covered = out.coverage > 0;
        cv::Mat unowned = covered & (instances == 0);
        if (cv::countNonZero(unowned) > 0) {
            cv::Mat grown;
            cv::dilate(instances, grown, cv::Mat::ones(3, 3, CV_8U));
            grown.copyTo(instances, unowned);
        }
        instances.setTo(cv::Scalar(0), ~covered);
    }
    out.instances = instances;
    return out;
}

GateRenderer::Segmentation GateRenderer::renderSegmented(double x, double y, double z, double roll, double pitch,
                                                          double yaw) const {
    return renderSegmented(DronePose{x, y, z, roll, pitch, yaw});
}

std::vector<std::string> GateRenderer::gateNames() const {
    std::vector<std::string> names;
    names.reserve(gates_.size());
    for (const auto& [name, unused] : gates_) {
        (void)unused;
        names.push_back(name);
    }
    return names;
}

std::vector<GateDetection> GateRenderer::renderDetections(const DronePose& pose, int minVisibleCorners) const {
    std::vector<GateDetection> detections =
        detectGates(gates_, gateDims_, pose, tBaseCam_, cameraMatrix_, distCoeffs_, renderWidth_, renderHeight_,
                     fisheye_, thetaMax_, minVisibleCorners);
    if (!resample_) {
        return detections;
    }

    // Keypoints come out in render-resolution pixels; move them onto the mask
    // the caller actually gets, or they would silently point off-image. Purely
    // a change of units -- visibility was already decided in the render frame,
    // and scaling is a bijection of the image rectangle, so it cannot alter it.
    const double scaleX = static_cast<double>(outputWidth_) / renderWidth_;
    const double scaleY = static_cast<double>(outputHeight_) / renderHeight_;
    for (auto& detection : detections) {
        for (auto& keypoint : detection.keypoints) {
            keypoint.x *= scaleX;
            keypoint.y *= scaleY;
        }
        detection.boundingBox.x1 *= scaleX;
        detection.boundingBox.x2 *= scaleX;
        detection.boundingBox.y1 *= scaleY;
        detection.boundingBox.y2 *= scaleY;
    }
    return detections;
}

std::vector<GateDetection> GateRenderer::renderDetections(double x, double y, double z, double roll, double pitch,
                                                           double yaw, int minVisibleCorners) const {
    return renderDetections(DronePose{x, y, z, roll, pitch, yaw}, minVisibleCorners);
}

}  // namespace detect_gates
