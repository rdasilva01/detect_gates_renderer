#include "segment_gate/scene.hpp"

#include <algorithm>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "segment_gate/gates.hpp"
#include "segment_gate/projection.hpp"
#include "segment_gate/render.hpp"

namespace segment_gate {

namespace {

Polygon3d toCameraFrame(const Polygon3d& ptsWorld, const Transform& transform) {
    Polygon3d ptsCam;
    ptsCam.reserve(ptsWorld.size());
    for (const auto& p : ptsWorld) {
        ptsCam.push_back(transform.R * p + transform.t);
    }
    return ptsCam;
}

cv::Mat matFromYamlData(const YAML::Node& node, int rows, int cols) {
    cv::Mat mat(rows, cols, CV_64F);
    const YAML::Node& data = node["data"];
    for (int i = 0; i < rows * cols; ++i) {
        mat.at<double>(i / cols, i % cols) = data[i].as<double>();
    }
    return mat;
}

}  // namespace

std::map<std::string, GatePose> loadGatesConfig(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node posesNode = root["gates_poses"];

    std::map<std::string, GatePose> gates;
    for (const auto& entry : posesNode) {
        const std::string name = entry.first.as<std::string>();
        const YAML::Node& pose = entry.second;
        gates[name] = GatePose{pose[0].as<double>(), pose[1].as<double>(), pose[2].as<double>(), pose[3].as<double>()};
    }
    return gates;
}

GateDims loadGateDims(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);

    const YAML::Node& dims = root["gate_dimensions"];
    return GateDims{dims["outer_size"].as<double>(), dims["inner_size"].as<double>(),
                     dims["thickness"] ? dims["thickness"].as<double>() : 0.0};
}

CameraCalibration loadCameraCalibration(const std::string& path) {
    YAML::Node root = YAML::LoadFile(path);
    if (root["/**"]) {
        root = root["/**"]["ros__parameters"];
    }

    CameraCalibration calib;
    calib.imageWidth = root["image_width"].as<int>();
    calib.imageHeight = root["image_height"].as<int>();
    calib.cameraMatrix = matFromYamlData(root["camera_matrix"], 3, 3);
    calib.distCoeffs = matFromYamlData(root["distortion_coefficients"], 4, 1);

    const YAML::Node& tf = root["camera_transform"];
    calib.tBaseCam = poseToTransform(tf["x"].as<double>(), tf["y"].as<double>(), tf["z"].as<double>(),
                                      tf["roll"].as<double>(), tf["pitch"].as<double>(), tf["yaw"].as<double>());

    return calib;
}

cv::Mat renderPose(const std::map<std::string, GatePose>& gates, const GateDims& gateDims,
                    const DronePose& dronePos, const Transform& tBaseCam, const cv::Mat& cameraMatrix,
                    const cv::Mat& distCoeffs, int imageWidth, int imageHeight, bool fisheye, double thetaMax) {
    const Transform tWorldBase =
        poseToTransform(dronePos.x, dronePos.y, dronePos.z, dronePos.roll, dronePos.pitch, dronePos.yaw);
    const Transform tWorldCam = compose(tWorldBase, tBaseCam);
    const Transform tCamWorld = invert(tWorldCam);

    auto projectFace = [&](const Polygon3d& faceWorld) -> FacePixels {
        const Polygon3d faceCam = toCameraFrame(faceWorld, tCamWorld);
        return projectPolygon(subdivideEdges(faceCam), cameraMatrix, distCoeffs, thetaMax, 32, fisheye);
    };

    std::vector<GateFacesPx> gatesPx;
    for (const auto& [name, gatePose] : gates) {
        const GateFaces faces =
            gateFaces(gatePose.x, gatePose.y, gatePose.z, gatePose.yaw, gateDims.outerSize, gateDims.innerSize,
                      gateDims.thickness);

        GateFacesPx gatePx;
        gatePx.outerFacesPx.reserve(faces.outerFaces.size());
        for (const auto& face : faces.outerFaces) {
            gatePx.outerFacesPx.push_back(projectFace(face));
        }

        const bool allOuterClipped = std::all_of(gatePx.outerFacesPx.begin(), gatePx.outerFacesPx.end(),
                                                   [](const FacePixels& f) { return !f.has_value(); });
        if (allOuterClipped) {
            continue;
        }

        gatePx.innerFacesPx.reserve(faces.innerFaces.size());
        for (const auto& face : faces.innerFaces) {
            gatePx.innerFacesPx.push_back(projectFace(face));
        }

        gatesPx.push_back(std::move(gatePx));
    }

    return renderSegmentation(gatesPx, imageWidth, imageHeight);
}

}  // namespace segment_gate
