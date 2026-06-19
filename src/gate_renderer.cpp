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

    if (rectified) {
        cameraMatrix_ = rectifiedCameraMatrix(cameraMatrix_, distCoeffs_, imageWidth_, imageHeight_);
        distCoeffs_ = cv::Mat::zeros(4, 1, CV_64F);
        thetaMax_ = rectifiedThetaMax(cameraMatrix_, imageWidth_, imageHeight_);
        fisheye_ = false;
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
