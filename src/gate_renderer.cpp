#include "detect_gates/gate_renderer.hpp"

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
