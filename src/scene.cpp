#include "detect_gates/scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include "detect_gates/gates.hpp"
#include "detect_gates/projection.hpp"
#include "detect_gates/render.hpp"

namespace detect_gates {

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

// Canonical keypoint corner order, and where each canonical corner sits in
// `squareCorners()`'s own (lateral_sign, vertical_sign) index order
// (gates.cpp's kCornerSigns), depending on which side of the gate the
// camera is on (see detectGates()).
constexpr const char* kCanonicalNames[4] = {"top_left", "top_right", "bottom_right", "bottom_left"};
constexpr int kFrontIndexMap[4] = {2, 3, 0, 1};
constexpr int kBackIndexMap[4] = {3, 2, 1, 0};

// Transform a world-frame face to camera frame, subdivide its edges, and
// clip+project it -- the same per-face pipeline `renderPose` uses, shared so
// pose-mode occlusion tests against the exact rendered silhouette rather
// than an unsubdivided straight-line approximation.
FacePixels projectFaceClipped(const Polygon3d& faceWorld, const Transform& tCamWorld, const cv::Mat& cameraMatrix,
                               const cv::Mat& distCoeffs, double thetaMax, bool fisheye) {
    const Polygon3d faceCam = toCameraFrame(faceWorld, tCamWorld);
    return projectPolygon(subdivideEdges(faceCam), cameraMatrix, distCoeffs, thetaMax, 32, fisheye);
}

// One gate's detection candidate, before cross-gate occlusion is applied.
struct Candidate {
    std::string gateName;
    std::array<Keypoint, 8> keypoints;        // [0..3] = inner, [4..7] = outer, canonical order
    std::array<Eigen::Vector3d, 8> camPoints;  // same corners, camera-frame, for the per-ray depth check below
    cv::Mat footprint;                        // this gate's full rendered silhouette (same as segment mode)
    // Near face's plane in camera frame (unit normal + offset, normal.dot(x) == offset for x on the plane).
    // Used so occlusion requires the occluder to actually be nearer *along the
    // specific ray to each point*, not just nearer on average -- two gates
    // that are coplanar (e.g. stacked at different heights but otherwise
    // identical) must not occlude each other just because their fisheye-
    // curved silhouettes happen to overlap in pixel space.
    Eigen::Vector3d faceNormalCam;
    double faceOffsetCam = 0.0;
    double depth = 0.0;
};

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
    if (root["distortion_model"]) {
        const std::string model = root["distortion_model"].as<std::string>();
        calib.fisheye = (model == "fisheye" || model == "equidistant");
    }

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
        return projectFaceClipped(faceWorld, tCamWorld, cameraMatrix, distCoeffs, thetaMax, fisheye);
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

std::vector<GateDetection> detectGates(const std::map<std::string, GatePose>& gates, const GateDims& gateDims,
                                        const DronePose& dronePos, const Transform& tBaseCam,
                                        const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, int imageWidth,
                                        int imageHeight, bool fisheye, double thetaMax, int minVisibleCorners) {
    const Transform tWorldBase =
        poseToTransform(dronePos.x, dronePos.y, dronePos.z, dronePos.roll, dronePos.pitch, dronePos.yaw);
    const Transform tWorldCam = compose(tWorldBase, tBaseCam);
    const Transform tCamWorld = invert(tWorldCam);
    const Eigen::Vector3d camPosWorld = tWorldCam.t;

    std::vector<Candidate> candidates;
    for (const auto& [name, gatePose] : gates) {
        const GateFaces faces = gateFaces(gatePose.x, gatePose.y, gatePose.z, gatePose.yaw, gateDims.outerSize,
                                           gateDims.innerSize, gateDims.thickness);

        const Eigen::Vector3d center(gatePose.x, gatePose.y, gatePose.z);
        const Eigen::Vector3d normal(std::cos(gatePose.yaw), std::sin(gatePose.yaw), 0.0);
        const bool front = normal.dot(camPosWorld - center) < 0.0;

        const Polygon3d& outerFace = front ? faces.outerFaces[0] : faces.outerFaces[1];
        const Polygon3d& innerFace = front ? faces.innerFaces[0] : faces.innerFaces[1];
        const int* indexMap = front ? kFrontIndexMap : kBackIndexMap;

        // Canonical order: 4 inner corners, then 4 outer corners.
        Polygon3d cornersWorld;
        cornersWorld.reserve(8);
        for (int i = 0; i < 4; ++i) {
            cornersWorld.push_back(innerFace[indexMap[i]]);
        }
        for (int i = 0; i < 4; ++i) {
            cornersWorld.push_back(outerFace[indexMap[i]]);
        }

        const Polygon3d cornersCam = toCameraFrame(cornersWorld, tCamWorld);
        const std::vector<cv::Point2d> projected = projectPoints(cornersCam, cameraMatrix, distCoeffs, fisheye);

        Candidate cand;
        cand.gateName = name;
        // Outer corners (indices 4..6) are coplanar with the inner ones by
        // construction (gateFaces() builds both from the same front/back
        // center using the same lateral/vertical axes), so this plane
        // describes the whole near face.
        cand.faceNormalCam = (cornersCam[5] - cornersCam[4]).cross(cornersCam[6] - cornersCam[4]).normalized();
        cand.faceOffsetCam = cand.faceNormalCam.dot(cornersCam[4]);

        double depthSum = 0.0;
        int visibleCount = 0;
        for (int i = 0; i < 8; ++i) {
            const Eigen::Vector3d& p = cornersCam[i];
            const double theta = std::acos(p.z() / p.norm());
            const bool inCone = theta < thetaMax;
            const bool inBounds = projected[i].x >= 0 && projected[i].x < imageWidth && projected[i].y >= 0 &&
                                   projected[i].y < imageHeight;
            const bool visible = inCone && inBounds;

            const std::string suffix = i < 4 ? "_inner" : "_outer";
            cand.keypoints[i] = Keypoint{kCanonicalNames[i % 4] + suffix, projected[i].x, projected[i].y, visible,
                                          /*inFrustum=*/visible};
            cand.camPoints[i] = p;
            if (visible) {
                ++visibleCount;
            }
            depthSum += p.z();
        }
        if (visibleCount < minVisibleCorners) {
            continue;
        }

        // This gate's full rendered silhouette (all outer faces OR-ed minus
        // all inner faces AND-ed, subdivided + cone-clipped) -- identical to
        // what `renderPose` would draw for this gate alone. Used below so
        // occlusion tests against the true curved/extruded shape rather
        // than a straight-line approximation of just the near face.
        GateFacesPx gatePx;
        gatePx.outerFacesPx.reserve(faces.outerFaces.size());
        for (const auto& face : faces.outerFaces) {
            gatePx.outerFacesPx.push_back(
                projectFaceClipped(face, tCamWorld, cameraMatrix, distCoeffs, thetaMax, fisheye));
        }
        gatePx.innerFacesPx.reserve(faces.innerFaces.size());
        for (const auto& face : faces.innerFaces) {
            gatePx.innerFacesPx.push_back(
                projectFaceClipped(face, tCamWorld, cameraMatrix, distCoeffs, thetaMax, fisheye));
        }
        cand.footprint = singleGateMask(gatePx, imageWidth, imageHeight);

        cand.depth = depthSum / 8.0;
        candidates.push_back(std::move(cand));
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.depth < b.depth; });

    // Cross-gate occlusion: a farther gate's keypoint is hidden only if a
    // closer gate's silhouette covers its pixel *and* that closer gate's
    // face plane is actually nearer along this specific ray (not just
    // nearer on average) -- otherwise two coplanar gates whose fisheye-
    // curved silhouettes happen to overlap in pixel space would wrongly
    // occlude each other (see Candidate::faceNormalCam).
    constexpr double kCoplanarEpsilon = 1e-3;  // meters; numerical tolerance, not a physical margin
    for (size_t i = 1; i < candidates.size(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            const Candidate& occluder = candidates[j];
            for (int k = 0; k < 8; ++k) {
                Keypoint& kp = candidates[i].keypoints[k];
                if (!kp.visible) {
                    continue;
                }
                const int xi = static_cast<int>(std::lround(kp.x));
                const int yi = static_cast<int>(std::lround(kp.y));
                if (xi < 0 || xi >= occluder.footprint.cols || yi < 0 || yi >= occluder.footprint.rows) {
                    continue;
                }
                if (occluder.footprint.at<uint8_t>(yi, xi) == 0) {
                    continue;
                }

                const Eigen::Vector3d& p = candidates[i].camPoints[k];
                const double denom = occluder.faceNormalCam.dot(p) / p.norm();
                if (std::abs(denom) < 1e-9) {
                    continue;  // ray nearly parallel to the occluder's face plane
                }
                const double tOccluder = occluder.faceOffsetCam / denom;
                if (tOccluder < p.norm() - kCoplanarEpsilon) {
                    kp.visible = false;
                }
            }
        }
    }

    std::vector<GateDetection> detections;
    for (const auto& cand : candidates) {
        const int visibleCount = static_cast<int>(
            std::count_if(cand.keypoints.begin(), cand.keypoints.end(), [](const Keypoint& k) { return k.visible; }));
        if (visibleCount < minVisibleCorners) {
            continue;
        }

        GateDetection det;
        det.gate = cand.gateName;
        double minX = std::numeric_limits<double>::infinity();
        double minY = minX;
        double maxX = -minX;
        double maxY = -minX;
        for (const auto& kp : cand.keypoints) {
            det.keypoints.push_back(kp);
            if (!kp.visible) {
                continue;
            }
            minX = std::min(minX, kp.x);
            maxX = std::max(maxX, kp.x);
            minY = std::min(minY, kp.y);
            maxY = std::max(maxY, kp.y);
        }
        det.boundingBox = BoundingBox{minX, minY, maxX, maxY};
        detections.push_back(std::move(det));
    }
    return detections;
}

}  // namespace detect_gates
