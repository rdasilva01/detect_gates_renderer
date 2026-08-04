#include "detect_gates/gate_renderer.hpp"

#include "detect_gates/projection.hpp"

namespace detect_gates {

GateRenderer::GateRenderer(const std::string& gatesConfigPath, const std::string& droneConfigPath,
                            const std::string& cameraConfigPath, bool rectified) {
    gates_ = loadGatesConfig(gatesConfigPath);
    gateDims_ = loadGateDims(droneConfigPath);

    const CameraCalibration calib = loadCameraCalibration(cameraConfigPath);
    tBaseCam_ = calib.tBaseCam;
    imageWidth_ = calib.imageWidth;
    imageHeight_ = calib.imageHeight;
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
            cameraMatrix_ = rectifiedCameraMatrix(cameraMatrix_, distCoeffs_, imageWidth_, imageHeight_);
            fisheye_ = false;
        }
        distCoeffs_ = cv::Mat::zeros(4, 1, CV_64F);
        thetaMax_ = rectifiedThetaMax(cameraMatrix_, imageWidth_, imageHeight_);
    } else if (fisheye_) {
        // Raw fisheye render: this is the camera's true angular extent, not a
        // clipping cone, and for a >180 deg lens it exceeds 90 deg (105.8 deg
        // for the calibration in config/). Leaving it at the old 89 deg default
        // discarded the outermost ring of the image, which is precisely where
        // a gate sits while the drone is crossing it and just after.
        thetaMax_ = fisheyeThetaMax(cameraMatrix_, distCoeffs_, imageWidth_, imageHeight_);
    } else {
        // Native pinhole source, raw (still-distorted) render: the default
        // thetaMax_ (89 deg) assumes nothing about this camera's actual FOV;
        // derive the real clipping cone from its camera matrix + resolution.
        thetaMax_ = rectifiedThetaMax(cameraMatrix_, imageWidth_, imageHeight_);
    }
}

cv::Mat GateRenderer::render(const DronePose& pose) const {
    return renderPose(gates_, gateDims_, pose, tBaseCam_, cameraMatrix_, distCoeffs_, imageWidth_, imageHeight_,
                       fisheye_, thetaMax_);
}

cv::Mat GateRenderer::render(double x, double y, double z, double roll, double pitch, double yaw) const {
    return render(DronePose{x, y, z, roll, pitch, yaw});
}

std::vector<GateDetection> GateRenderer::renderDetections(const DronePose& pose, int minVisibleCorners) const {
    return detectGates(gates_, gateDims_, pose, tBaseCam_, cameraMatrix_, distCoeffs_, imageWidth_, imageHeight_,
                        fisheye_, thetaMax_, minVisibleCorners);
}

std::vector<GateDetection> GateRenderer::renderDetections(double x, double y, double z, double roll, double pitch,
                                                           double yaw, int minVisibleCorners) const {
    return renderDetections(DronePose{x, y, z, roll, pitch, yaw}, minVisibleCorners);
}

}  // namespace detect_gates
