#include "detect_gates/scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
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

// Canonical keypoint corner order -- clockwise in the image from the top
// flat's left end -- and where each canonical corner sits in the ring order
// `gateFaces()` builds (gates.cpp's squareCorners/octagonCorners), depending
// on which side of the gate the camera is on (see detectGates()).
constexpr const char* kSquareNames[4] = {"top_left", "top_right", "bottom_right", "bottom_left"};
constexpr int kSquareFrontIndexMap[4] = {2, 3, 0, 1};
constexpr int kSquareBackIndexMap[4] = {3, 2, 1, 0};
constexpr const char* kOctagonNames[8] = {"top_left",     "top_right",   "right_top",   "right_bottom",
                                          "bottom_right", "bottom_left", "left_bottom", "left_top"};
constexpr int kOctagonFrontIndexMap[8] = {4, 5, 6, 7, 0, 1, 2, 3};
constexpr int kOctagonBackIndexMap[8] = {5, 4, 3, 2, 1, 0, 7, 6};

struct CornerLayout {
    int count;  // corners per ring
    const char* const* names;
    const int* frontIndexMap;
    const int* backIndexMap;
};

CornerLayout cornerLayout(GateShape shape) {
    if (shape == GateShape::Octagon) {
        return {8, kOctagonNames, kOctagonFrontIndexMap, kOctagonBackIndexMap};
    }
    return {4, kSquareNames, kSquareFrontIndexMap, kSquareBackIndexMap};
}

// The frames a gate is made of for masks, instance labels and detections: the
// gate itself, or a double's two squares, top first. The pose is the bottom
// square's centre and the top square sits one outer size above it, so the two
// are exactly two square gates stacked -- which is what their masks must be.
std::vector<Gate> partsOf(const Gate& gate) {
    if (gate.shape != GateShape::Double) {
        return {gate};
    }
    Gate bottom = gate;
    bottom.shape = GateShape::Square;
    Gate top = bottom;
    top.pose.z += gate.dims.outerSize;
    return {top, bottom};
}

// Is the camera inside this gate's through-hole -- between the two apertures
// and laterally within them? See `singleGateMask` for what it changes.
bool cameraInAperture(const GatePose& gate, GateShape shape, const GateDims& gateDims,
                      const Eigen::Vector3d& camPosWorld) {
    const Eigen::Vector3d offset = camPosWorld - Eigen::Vector3d(gate.x, gate.y, gate.z);
    const Eigen::Vector3d normal(std::cos(gate.yaw), std::sin(gate.yaw), 0.0);
    const Eigen::Vector3d lateral(-std::sin(gate.yaw), std::cos(gate.yaw), 0.0);
    const double innerHalf = gateDims.innerSize / 2.0;
    const double u = std::abs(offset.dot(lateral));
    const double v = std::abs(offset.z());
    const bool inSquare = std::abs(offset.dot(normal)) <= gateDims.thickness / 2.0 && u <= innerHalf &&
                          v <= innerHalf;
    // An octagon is that square with its corners cut off: the diagonal flats
    // are innerHalf from the centre too.
    return inSquare && (shape != GateShape::Octagon || u + v <= innerHalf * std::sqrt(2.0));
}

// True if this outline lies wholly off one side of the canvas, and so cannot
// contribute a pixel no matter how it is filled.
bool offCanvas(const std::vector<cv::Point2d>& pts, int imageWidth, int imageHeight) {
    if (pts.empty()) {
        return true;
    }
    double minX = pts[0].x, maxX = pts[0].x, minY = pts[0].y, maxY = pts[0].y;
    for (const auto& p : pts) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }
    return maxX < 0 || minX > imageWidth || maxY < 0 || minY > imageHeight;
}

// Transform a world-frame face to camera frame, subdivide its edges, and
// clip+project it -- the same per-face pipeline `renderPose` uses, shared so
// pose-mode occlusion tests against the exact rendered silhouette rather
// than an unsubdivided straight-line approximation.
FacePixels projectFaceClipped(const Polygon3d& faceWorld, const Transform& tCamWorld, const cv::Mat& cameraMatrix,
                               const cv::Mat& distCoeffs, double thetaMax, bool fisheye, int imageWidth,
                               int imageHeight) {
    const Polygon3d faceCam = toCameraFrame(faceWorld, tCamWorld);
    return projectPolygon(subdivideEdges(faceCam), cameraMatrix, distCoeffs, imageWidth, imageHeight, thetaMax, 32,
                           fisheye);
}

// Can this gate contribute nothing to the canvas? Its mask is its outer faces
// minus its apertures, so if no outer face reaches the canvas there is nothing
// to draw and the whole gate can be skipped.
//
// This has to be judged per gate, never per face. An empty outline means
// "skip me" to `singleGateMask`, which is right for an outer face and quietly
// destructive for an inner one -- dropping an off-canvas aperture there leaves
// the gate with no hole to cut, and (both apertures gone) with no mask at all.
// That is the shape of the bug this whole path is fixing; do not reintroduce it
// as an optimization.
bool gateOffCanvas(const GateFacesPx& gatePx, int imageWidth, int imageHeight) {
    return std::all_of(gatePx.outerFacesPx.begin(), gatePx.outerFacesPx.end(), [&](const FacePixels& face) {
        // An inverted outline that misses the canvas means the face *covers*
        // the canvas, so it is never grounds for skipping.
        return face.points.empty() || (!face.inverted && offCanvas(face.points, imageWidth, imageHeight));
    });
}

// One gate's detection candidate, before cross-gate occlusion is applied.
struct Candidate {
    std::string gateName;
    std::vector<Keypoint> keypoints;         // inner ring, then outer ring, canonical order
    std::vector<Eigen::Vector3d> camPoints;  // same corners, camera-frame, for the per-ray depth check below
    cv::Mat footprint;                       // this gate's full rendered silhouette (same as segment mode)
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

DronePose poseFromQuaternion(double x, double y, double z, double qw, double qx, double qy, double qz) {
    const Eigen::Vector3d rpy = matrixToRpy(quaternionToMatrix(qw, qx, qy, qz));
    return DronePose{x, y, z, rpy.x(), rpy.y(), rpy.z()};
}

std::map<std::string, Gate> loadGatesConfig(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node gatesNode = root["gates"];

    std::map<std::string, Gate> gates;
    for (const auto& entry : gatesNode) {
        const std::string name = entry.first.as<std::string>();
        const YAML::Node& gateNode = entry.second;

        const std::string type = gateNode["type"].as<std::string>();
        GateShape shape = GateShape::Square;
        if (type == "octagon") {
            shape = GateShape::Octagon;
        } else if (type == "double") {
            shape = GateShape::Double;
        } else if (type != "square") {
            throw std::runtime_error("gates_config: gate '" + name + "' has unknown type '" + type +
                                     "' (known: square, octagon, double)");
        }

        const YAML::Node& pose = gateNode["pose"];
        const YAML::Node& dims = gateNode["dimensions"];
        gates[name] = Gate{
            shape,
            GatePose{pose[0].as<double>(), pose[1].as<double>(), pose[2].as<double>(), pose[3].as<double>()},
            GateDims{dims["outer_size"].as<double>(), dims["inner_size"].as<double>(),
                     dims["thickness"] ? dims["thickness"].as<double>() : 0.0}};
    }
    return gates;
}

OutputSettings loadOutputSettings(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);

    OutputSettings output;

    const bool hasWidth = static_cast<bool>(root["output_width"]);
    const bool hasHeight = static_cast<bool>(root["output_height"]);
    if (hasWidth != hasHeight) {
        throw std::runtime_error("config: output_width and output_height must be set together");
    }
    if (hasWidth) {
        output.width = root["output_width"].as<int>();
        output.height = root["output_height"].as<int>();
        if (output.width <= 0 || output.height <= 0) {
            throw std::runtime_error("config: output_width and output_height must be positive");
        }
    }

    if (root["inter_method"]) {
        const std::string method = root["inter_method"].as<std::string>();
        if (method == "nearest") {
            output.interMethod = InterMethod::Nearest;
        } else if (method == "linear") {
            output.interMethod = InterMethod::Linear;
        } else if (method == "area") {
            output.interMethod = InterMethod::Area;
        } else {
            throw std::runtime_error("config: inter_method must be nearest, linear or area (got '" + method + "')");
        }
    }

    if (root["native_inter"]) {
        output.nativeInter = root["native_inter"].as<bool>();
    }

    return output;
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

cv::Mat renderPose(const std::map<std::string, Gate>& gates, const DronePose& dronePos, const Transform& tBaseCam,
                    const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, int imageWidth, int imageHeight,
                    bool fisheye, double thetaMax) {
    const Transform tWorldBase =
        poseToTransform(dronePos.x, dronePos.y, dronePos.z, dronePos.roll, dronePos.pitch, dronePos.yaw);
    const Transform tWorldCam = compose(tWorldBase, tBaseCam);
    const Transform tCamWorld = invert(tWorldCam);

    auto projectFace = [&](const Polygon3d& faceWorld) -> FacePixels {
        return projectFaceClipped(faceWorld, tCamWorld, cameraMatrix, distCoeffs, thetaMax, fisheye, imageWidth,
                                   imageHeight);
    };

    std::vector<GateFacesPx> gatesPx;
    for (const auto& [name, entry] : gates) {
        for (const Gate& gate : partsOf(entry)) {
            const GatePose& gatePose = gate.pose;
            const GateDims& gateDims = gate.dims;
            const GateFaces faces =
                gateFaces(gate.shape, gatePose.x, gatePose.y, gatePose.z, gatePose.yaw, gateDims.outerSize,
                          gateDims.innerSize, gateDims.thickness);

            GateFacesPx gatePx;
            gatePx.cameraInAperture = cameraInAperture(gatePose, gate.shape, gateDims, tWorldCam.t);
            gatePx.outerFacesPx.reserve(faces.outerFaces.size());
            for (const auto& face : faces.outerFaces) {
                gatePx.outerFacesPx.push_back(projectFace(face));
            }

            if (gateOffCanvas(gatePx, imageWidth, imageHeight)) {
                continue;
            }

            gatePx.innerFacesPx.reserve(faces.innerFaces.size());
            for (const auto& face : faces.innerFaces) {
                gatePx.innerFacesPx.push_back(projectFace(face));
            }

            gatesPx.push_back(std::move(gatePx));
        }
    }

    return renderSegmentation(gatesPx, imageWidth, imageHeight);
}

namespace {

// Unit camera-frame ray through each pixel position, inverting exactly the
// projection the rasterizer used: the full-range equidistant model for a
// fisheye (so theta may pass 90 deg), `cv::undistortPoints` for a pinhole.
std::vector<Eigen::Vector3d> pixelRays(const std::vector<cv::Point2d>& pixels, const cv::Mat& cameraMatrix,
                                       const cv::Mat& distCoeffs, bool fisheye) {
    std::vector<Eigen::Vector3d> rays;
    rays.reserve(pixels.size());
    if (!fisheye) {
        std::vector<cv::Point2d> normalized;
        cv::undistortPoints(pixels, normalized, cameraMatrix, distCoeffs);
        for (const auto& q : normalized) {
            rays.push_back(Eigen::Vector3d(q.x, q.y, 1.0).normalized());
        }
        return rays;
    }
    const double fx = cameraMatrix.at<double>(0, 0), fy = cameraMatrix.at<double>(1, 1);
    const double cx = cameraMatrix.at<double>(0, 2), cy = cameraMatrix.at<double>(1, 2);
    std::array<double, 4> k{0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < std::min(4, static_cast<int>(distCoeffs.total())); ++i) {
        k[i] = distCoeffs.at<double>(i);
    }
    for (const auto& px : pixels) {
        const double mx = (px.x - cx) / fx, my = (px.y - cy) / fy;
        const double thetaD = std::hypot(mx, my);
        // Invert theta_d(theta) by Newton, as `fisheyeThetaMax` does.
        double theta = thetaD;
        for (int i = 0; i < 20; ++i) {
            const double t2 = theta * theta;
            const double f = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3])))) - thetaD;
            const double df = 1.0 + t2 * (3.0 * k[0] + t2 * (5.0 * k[1] + t2 * (7.0 * k[2] + t2 * 9.0 * k[3])));
            if (std::abs(df) < 1e-12) {
                break;
            }
            const double step = f / df;
            theta -= step;
            if (std::abs(step) < 1e-12) {
                break;
            }
        }
        theta = std::clamp(theta, 0.0, CV_PI);
        if (thetaD < 1e-12) {
            rays.emplace_back(0.0, 0.0, 1.0);
        } else {
            const double s = std::sin(theta) / thetaD;
            rays.emplace_back(s * mx, s * my, std::cos(theta));
        }
    }
    return rays;
}

// A gate's frame solid -- the slab |along| <= thickness / 2, inside the outer
// ring and outside the inner one -- set up for intersecting rays with. Rings
// are half-planes a*u + b*v <= c over (u, v) = (lateral, vertical) offsets.
struct FrameSolid {
    Eigen::Vector3d center, normal, lateral;
    double halfThickness = 0.0;
    std::vector<std::array<double, 3>> outer, inner;
};

std::vector<std::array<double, 3>> ringHalfPlanes(GateShape shape, double halfSize) {
    std::vector<std::array<double, 3>> planes = {
        {1.0, 0.0, halfSize}, {-1.0, 0.0, halfSize}, {0.0, 1.0, halfSize}, {0.0, -1.0, halfSize}};
    if (shape == GateShape::Octagon) {
        const double diagonal = halfSize * std::sqrt(2.0);  // the cut corners: |u| + |v| <= h * sqrt 2
        for (double a : {-1.0, 1.0}) {
            for (double b : {-1.0, 1.0}) {
                planes.push_back({a, b, diagonal});
            }
        }
    }
    return planes;
}

FrameSolid frameSolid(const Gate& gate) {
    FrameSolid solid;
    solid.center = Eigen::Vector3d(gate.pose.x, gate.pose.y, gate.pose.z);
    solid.normal = Eigen::Vector3d(std::cos(gate.pose.yaw), std::sin(gate.pose.yaw), 0.0);
    solid.lateral = Eigen::Vector3d(-std::sin(gate.pose.yaw), std::cos(gate.pose.yaw), 0.0);
    solid.halfThickness = gate.dims.thickness / 2.0;
    solid.outer = ringHalfPlanes(gate.shape, gate.dims.outerSize / 2.0);
    solid.inner = ringHalfPlanes(gate.shape, gate.dims.innerSize / 2.0);
    return solid;
}

// Narrow [lo, hi] to where the 2D line p + s * d lies inside the half-planes.
// False if nothing is left.
bool clipToRing(double pu, double pv, double du, double dv, const std::vector<std::array<double, 3>>& planes,
                double& lo, double& hi) {
    for (const auto& [a, b, c] : planes) {
        const double slack = c - (a * pu + b * pv);
        const double rate = a * du + b * dv;
        if (std::abs(rate) < 1e-15) {
            if (slack < 0.0) {
                return false;
            }
            continue;
        }
        const double s = slack / rate;
        if (rate > 0.0) {
            hi = std::min(hi, s);
        } else {
            lo = std::max(lo, s);
        }
        if (lo > hi) {
            return false;
        }
    }
    return true;
}

// Distance along the unit ray (origin, dir) to its first point inside the
// frame solid, or infinity if it misses. Exact: the ray is linear in (u, v)
// across the slab, so it is inside the outer ring over one interval and inside
// the aperture over another, and the first hit is where the former starts
// unless the latter already covers that.
double firstHit(const FrameSolid& solid, const Eigen::Vector3d& origin, const Eigen::Vector3d& dir) {
    const double infinity = std::numeric_limits<double>::infinity();
    const Eigen::Vector3d offset = origin - solid.center;
    const double along = offset.dot(solid.normal);
    const double rate = dir.dot(solid.normal);
    double s0 = 0.0, s1 = infinity;
    if (std::abs(rate) > 1e-15) {
        double enter = (-solid.halfThickness - along) / rate, leave = (solid.halfThickness - along) / rate;
        if (enter > leave) {
            std::swap(enter, leave);
        }
        s0 = std::max(enter, 0.0);
        s1 = leave;
    } else if (std::abs(along) > solid.halfThickness) {
        return infinity;
    }
    if (s1 < s0) {
        return infinity;
    }
    const double pu = offset.dot(solid.lateral), pv = offset.z();
    const double du = dir.dot(solid.lateral), dv = dir.z();
    double inOuterFrom = s0, inOuterTo = s1;
    if (!clipToRing(pu, pv, du, dv, solid.outer, inOuterFrom, inOuterTo)) {
        return infinity;
    }
    double inHoleFrom = s0, inHoleTo = s1;
    const bool throughHole = clipToRing(pu, pv, du, dv, solid.inner, inHoleFrom, inHoleTo);
    if (!throughHole || inOuterFrom < inHoleFrom || inOuterFrom > inHoleTo) {
        return inOuterFrom;
    }
    return inHoleTo < inOuterTo ? inHoleTo : infinity;
}

// Instance labels for a pose and, if `coverage` is given, the semantic mask
// too. Both come from the same per-gate silhouettes, so a caller that wants the
// pair gets each gate projected and rasterized once rather than twice.
cv::Mat renderInstancesAndCoverage(const std::map<std::string, Gate>& gates, const DronePose& dronePos,
                                   const Transform& tBaseCam, const cv::Mat& cameraMatrix,
                                   const cv::Mat& distCoeffs, int imageWidth, int imageHeight, bool fisheye,
                                   double thetaMax, cv::Mat* coverage) {
    const Transform tWorldBase =
        poseToTransform(dronePos.x, dronePos.y, dronePos.z, dronePos.roll, dronePos.pitch, dronePos.yaw);
    const Transform tWorldCam = compose(tWorldBase, tBaseCam);
    const Transform tCamWorld = invert(tWorldCam);

    auto projectFace = [&](const Polygon3d& faceWorld) -> FacePixels {
        return projectFaceClipped(faceWorld, tCamWorld, cameraMatrix, distCoeffs, thetaMax, fisheye, imageWidth,
                                   imageHeight);
    };

    struct Painted {
        GateFacesPx px;
        uint8_t label = 0;
        Gate gate;              // the square or octagon frame painted (one of a double's squares)
        double distance = 0.0;  // camera to that frame's centre
    };
    std::vector<Painted> painted;
    uint8_t label = 0;
    for (const auto& [name, entry] : gates) {
        // Incremented for EVERY gate, including ones that fall off canvas, so a
        // label means the same gate whatever happens to be in view. A label
        // that shifted with visibility would silently rename gates between
        // frames, which is precisely the identity this exists to provide. A
        // double is one gate, so both its squares carry its one label.
        ++label;
        for (const Gate& gate : partsOf(entry)) {
            const GatePose& gatePose = gate.pose;
            const GateDims& gateDims = gate.dims;
            const GateFaces faces =
                gateFaces(gate.shape, gatePose.x, gatePose.y, gatePose.z, gatePose.yaw, gateDims.outerSize,
                          gateDims.innerSize, gateDims.thickness);

            GateFacesPx gatePx;
            gatePx.cameraInAperture = cameraInAperture(gatePose, gate.shape, gateDims, tWorldCam.t);
            gatePx.outerFacesPx.reserve(faces.outerFaces.size());
            for (const auto& face : faces.outerFaces) {
                gatePx.outerFacesPx.push_back(projectFace(face));
            }

            if (gateOffCanvas(gatePx, imageWidth, imageHeight)) {
                continue;
            }

            gatePx.innerFacesPx.reserve(faces.innerFaces.size());
            for (const auto& face : faces.innerFaces) {
                gatePx.innerFacesPx.push_back(projectFace(face));
            }

            const Eigen::Vector3d centreCam =
                tCamWorld.R * Eigen::Vector3d(gatePose.x, gatePose.y, gatePose.z) + tCamWorld.t;
            painted.push_back(Painted{std::move(gatePx), label, gate, centreCam.norm()});
        }
    }

    // Each gate's silhouette is the one `renderPose` ORs in, so the union of
    // owned pixels matches it exactly; only who owns a pixel is decided below.
    cv::Mat canvas = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
    cv::Mat coverCount = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
    if (coverage != nullptr) {
        *coverage = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
    }
    std::vector<cv::Mat> footprints;
    footprints.reserve(painted.size());
    for (const auto& entry : painted) {
        footprints.push_back(singleGateMask(entry.px, imageWidth, imageHeight));
        canvas.setTo(cv::Scalar(entry.label), footprints.back());
        cv::add(coverCount, cv::Scalar(1), coverCount, footprints.back());
        if (coverage != nullptr) {
            // Exactly what `renderSegmentation` does with the same silhouettes.
            cv::bitwise_or(*coverage, footprints.back(), *coverage);
        }
    }

    // **Where silhouettes overlap, ownership is decided per pixel, not per
    // gate.** Ordering whole gates by one depth each -- their centres' -- is
    // wrong for gates that are large, close and touching: which one is in
    // front changes across the overlap, and a small tilt reorders the centres
    // and hands the whole overlap to the other gate (measured: 12.8% of
    // overlapping pixels owned wrongly on fisheye, 12.4% rectified, 2.5% even
    // ordering by centre distance). Instead the pixel's own ray is cast against
    // each covering gate's frame solid and the first hit wins. Overlaps are a
    // small part of the image, so this is cheap.
    std::vector<cv::Point> contested;
    cv::findNonZero(coverCount > 1, contested);
    if (contested.empty()) {
        return canvas;
    }
    std::vector<cv::Point2d> centres;
    centres.reserve(contested.size());
    for (const auto& p : contested) {
        centres.emplace_back(p.x + 0.5, p.y + 0.5);
    }
    const std::vector<Eigen::Vector3d> rays = pixelRays(centres, cameraMatrix, distCoeffs, fisheye);
    std::vector<FrameSolid> solids;
    solids.reserve(painted.size());
    for (const auto& entry : painted) {
        solids.push_back(frameSolid(entry.gate));
    }
    for (size_t i = 0; i < contested.size(); ++i) {
        const cv::Point& p = contested[i];
        const Eigen::Vector3d dirWorld = tWorldCam.R * rays[i];
        double nearestHit = std::numeric_limits<double>::infinity();
        double nearestCentre = std::numeric_limits<double>::infinity();
        uint8_t owner = 0, fallback = 0;
        for (size_t g = 0; g < painted.size(); ++g) {
            if (footprints[g].at<uint8_t>(p) == 0) {
                continue;
            }
            const double hit = firstHit(solids[g], tWorldCam.t, dirWorld);
            if (hit < nearestHit) {
                nearestHit = hit;
                owner = painted[g].label;
            }
            if (painted[g].distance < nearestCentre) {
                nearestCentre = painted[g].distance;
                fallback = painted[g].label;
            }
        }
        // No hit at all is the rasterizer's fringe, a pixel whose centre just
        // misses every frame: give it to the nearest gate centre.
        canvas.at<uint8_t>(p) = owner != 0 ? owner : fallback;
    }
    return canvas;
}

}  // namespace

cv::Mat renderPoseInstances(const std::map<std::string, Gate>& gates, const DronePose& dronePos,
                             const Transform& tBaseCam, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                             int imageWidth, int imageHeight, bool fisheye, double thetaMax) {
    return renderInstancesAndCoverage(gates, dronePos, tBaseCam, cameraMatrix, distCoeffs, imageWidth, imageHeight,
                                      fisheye, thetaMax, nullptr);
}

SceneSegmentation renderPoseSegmented(const std::map<std::string, Gate>& gates, const DronePose& dronePos,
                                      const Transform& tBaseCam, const cv::Mat& cameraMatrix,
                                      const cv::Mat& distCoeffs, int imageWidth, int imageHeight, bool fisheye,
                                      double thetaMax) {
    SceneSegmentation out;
    out.instances = renderInstancesAndCoverage(gates, dronePos, tBaseCam, cameraMatrix, distCoeffs, imageWidth,
                                               imageHeight, fisheye, thetaMax, &out.coverage);
    return out;
}

namespace {

// A face outline as the polylines worth drawing. The pinhole path clips
// outlines to the image rectangle, which adds segments running along the
// border; those are not edges of the gate, so the outline is cut there.
std::vector<std::vector<cv::Point2d>> outlineEdges(const FacePixels& face, int imageWidth, int imageHeight) {
    std::vector<std::vector<cv::Point2d>> runs;
    const std::vector<cv::Point2d>& pts = face.points;
    const size_t n = pts.size();
    if (face.inverted || n < 2) {
        return runs;
    }
    const auto onBorder = [&](const cv::Point2d& a, const cv::Point2d& b) {
        const auto both = [](double u, double v, double edge) {
            return std::abs(u - edge) < 1e-6 && std::abs(v - edge) < 1e-6;
        };
        return both(a.x, b.x, 0.0) || both(a.x, b.x, imageWidth) || both(a.y, b.y, 0.0) ||
               both(a.y, b.y, imageHeight);
    };
    // Start right after a border segment, if any, so no run is split across the wrap.
    size_t start = 0;
    for (size_t i = 0; i < n; ++i) {
        if (onBorder(pts[i], pts[(i + 1) % n])) {
            start = (i + 1) % n;
            break;
        }
    }
    std::vector<cv::Point2d> run;
    for (size_t k = 0; k < n; ++k) {
        const cv::Point2d& a = pts[(start + k) % n];
        const cv::Point2d& b = pts[(start + k + 1) % n];
        if (onBorder(a, b)) {
            if (run.size() >= 2 && !offCanvas(run, imageWidth, imageHeight)) {
                runs.push_back(run);
            }
            run.clear();
            continue;
        }
        if (run.empty()) {
            run.push_back(a);
        }
        run.push_back(b);
    }
    if (run.size() >= 2 && !offCanvas(run, imageWidth, imageHeight)) {
        runs.push_back(run);
    }
    return runs;
}

// The stretches of a face's outline that nothing hides. The outline is sampled
// at most 2 px apart; each sample's ray is met with the face's own plane, and
// the sample is hidden when any frame solid -- another gate's, or this gate's
// own front ring in front of a far inner wall -- is hit meaningfully nearer.
//
// The tolerance (1 cm plus 0.2% of the depth) absorbs the fisheye outline's
// chords: they run a fraction of a pixel off the curved edge, so near a corner
// their rays can meet the neighbouring face of the same gate millimetres early.
// Real occlusion is at least a frame's thickness deep and is not affected.
std::vector<std::vector<cv::Point2d>> visibleRuns(const std::vector<cv::Point2d>& line,
                                                  const Eigen::Vector3d& planePoint,
                                                  const Eigen::Vector3d& planeNormal,
                                                  const std::vector<FrameSolid>& solids,
                                                  const std::vector<double>& radii, const Transform& tWorldCam,
                                                  const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                                                  bool fisheye, int imageWidth, int imageHeight) {
    constexpr double kStep = 2.0;  // px
    std::vector<cv::Point2d> samples{line.front()};
    for (size_t i = 1; i < line.size(); ++i) {
        const cv::Point2d delta = line[i] - line[i - 1];
        const int n = std::max(1, static_cast<int>(std::ceil(std::hypot(delta.x, delta.y) / kStep)));
        for (int k = 1; k <= n; ++k) {
            samples.push_back(line[i - 1] + delta * (static_cast<double>(k) / n));
        }
    }
    const std::vector<Eigen::Vector3d> rays = pixelRays(samples, cameraMatrix, distCoeffs, fisheye);
    const Eigen::Vector3d& origin = tWorldCam.t;

    std::vector<std::vector<cv::Point2d>> runs;
    std::vector<cv::Point2d> run;
    for (size_t i = 0; i < samples.size(); ++i) {
        const cv::Point2d& s = samples[i];
        // Off the canvas nothing is drawn, so do not spend a test or break the run there.
        const bool onCanvas = s.x > -kStep && s.y > -kStep && s.x < imageWidth + kStep && s.y < imageHeight + kStep;
        bool visible = true;
        const Eigen::Vector3d dir = tWorldCam.R * rays[i];
        const double facing = dir.dot(planeNormal);
        if (onCanvas && std::abs(facing) > 1e-9) {
            const double depth = (planePoint - origin).dot(planeNormal) / facing;
            const double tolerance = 0.01 + 0.002 * depth;
            for (size_t g = 0; visible && depth > 0.0 && g < solids.size(); ++g) {
                const Eigen::Vector3d toCentre = solids[g].center - origin;
                const double along = toCentre.dot(dir);
                if (along + radii[g] < 0.0 || along - radii[g] > depth ||
                    (toCentre - along * dir).norm() > radii[g]) {
                    continue;  // the ray misses this gate's bounding sphere before the face
                }
                visible = firstHit(solids[g], origin, dir) >= depth - tolerance;
            }
        }
        if (visible) {
            run.push_back(s);
        } else {
            if (run.size() >= 2) {
                runs.push_back(run);
            }
            run.clear();
        }
    }
    if (run.size() >= 2) {
        runs.push_back(run);
    }
    return runs;
}

}  // namespace

std::vector<GateEdges> gateFaceEdges(const std::map<std::string, Gate>& gates, const DronePose& dronePos,
                                      const Transform& tBaseCam, const cv::Mat& cameraMatrix,
                                      const cv::Mat& distCoeffs, int imageWidth, int imageHeight, bool fisheye,
                                      double thetaMax) {
    const Transform tWorldBase =
        poseToTransform(dronePos.x, dronePos.y, dronePos.z, dronePos.roll, dronePos.pitch, dronePos.yaw);
    const Transform tWorldCam = compose(tWorldBase, tBaseCam);
    const Transform tCamWorld = invert(tWorldCam);
    const Eigen::Vector3d camPosWorld = tWorldCam.t;

    // Every gate's frame, and a sphere around it, for hiding occluded stretches.
    std::vector<FrameSolid> solids;
    std::vector<double> radii;
    for (const auto& [otherName, entry] : gates) {
        for (const Gate& other : partsOf(entry)) {
            solids.push_back(frameSolid(other));
            // Outer ring's circumradius squared over outerSize squared, as in detectGates' cull.
            const double circumFactor = other.shape == GateShape::Octagon ? 1.0 / (2.0 + std::sqrt(2.0)) : 0.5;
            radii.push_back(std::sqrt(circumFactor * other.dims.outerSize * other.dims.outerSize +
                                      0.25 * other.dims.thickness * other.dims.thickness));
        }
    }

    std::vector<GateEdges> result;
    for (const auto& [name, gate] : gates) {
        // Drawn from the gate's real faces: a double is one frame with two
        // apertures here, not two squares, so no line runs where they meet.
        const GatePose& gatePose = gate.pose;
        const GateDims& gateDims = gate.dims;
        const GateFaces faces = gateFaces(gate.shape, gatePose.x, gatePose.y, gatePose.z, gatePose.yaw,
                                           gateDims.outerSize, gateDims.innerSize, gateDims.thickness);
        const Eigen::Vector3d center(gatePose.x, gatePose.y, gatePose.z);
        const Eigen::Vector3d normal(std::cos(gatePose.yaw), std::sin(gatePose.yaw), 0.0);
        const size_t apertures = faces.innerFaces.size() / 2;

        // A ring face is visible from outside its own plane; from between the
        // two planes (inside the frame's thickness) neither is.
        std::vector<Polygon3d> visible;
        const double along = normal.dot(camPosWorld - center);
        if (along < -gateDims.thickness / 2.0 || along > gateDims.thickness / 2.0) {
            const int ring = along < 0.0 ? 0 : 1;  // front ring faces -normal, back ring +normal
            visible.push_back(faces.outerFaces[ring]);
            for (size_t a = 0; a < apertures; ++a) {
                visible.push_back(faces.innerFaces[2 * a + ring]);
            }
        }
        if (gateDims.thickness > 0.0) {
            // A wall is visible when the camera is on the side it faces. Its
            // normal is perpendicular to its edge and to the gate's axis, and
            // points away from its ring's centre for an outer wall, toward it for
            // an inner one -- true of any convex ring, a double's outline included.
            const auto facesCamera = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b,
                                         const Eigen::Vector3d& ringCentre, bool outward) {
                Eigen::Vector3d wallNormal = (b - a).cross(normal);
                if (wallNormal.dot(0.5 * (a + b) - ringCentre) < 0.0) {
                    wallNormal = -wallNormal;
                }
                return (outward ? 1.0 : -1.0) * wallNormal.dot(camPosWorld - a) > 0.0;
            };
            const auto centroid = [](const Polygon3d& ring) {
                Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                for (const auto& p : ring) {
                    sum += p;
                }
                return Eigen::Vector3d(sum / static_cast<double>(ring.size()));
            };
            const Eigen::Vector3d outerCentre = centroid(faces.outerFaces[0]);
            for (size_t i = 2; i < faces.outerFaces.size(); ++i) {
                const Polygon3d& wall = faces.outerFaces[i];  // {front[i], front[j], back[j], back[i]}
                if (facesCamera(wall[0], wall[1], outerCentre, true)) {
                    visible.push_back(wall);
                }
            }
            for (size_t a = 0; a < apertures; ++a) {
                const Polygon3d& frontInner = faces.innerFaces[2 * a];
                const Polygon3d& backInner = faces.innerFaces[2 * a + 1];
                const Eigen::Vector3d holeCentre = centroid(frontInner);
                const size_t n = frontInner.size();
                for (size_t i = 0; i < n; ++i) {
                    const size_t j = (i + 1) % n;
                    if (facesCamera(frontInner[i], frontInner[j], holeCentre, false)) {
                        visible.push_back(Polygon3d{frontInner[i], frontInner[j], backInner[j], backInner[i]});
                    }
                }
            }
        }

        GateEdges edges;
        edges.gate = name;
        for (const auto& face : visible) {
            const FacePixels facePx = projectFaceClipped(face, tCamWorld, cameraMatrix, distCoeffs, thetaMax, fisheye,
                                                         imageWidth, imageHeight);
            const Eigen::Vector3d planeNormal = (face[1] - face[0]).cross(face[2] - face[0]).normalized();
            for (const auto& run : outlineEdges(facePx, imageWidth, imageHeight)) {
                for (auto& part : visibleRuns(run, face[0], planeNormal, solids, radii, tWorldCam, cameraMatrix,
                                              distCoeffs, fisheye, imageWidth, imageHeight)) {
                    edges.polylines.push_back(std::move(part));
                }
            }
        }
        if (!edges.polylines.empty()) {
            result.push_back(std::move(edges));
        }
    }
    return result;
}

BoundingBox boundingBoxOfMask(const cv::Mat& mask) {
    const cv::Rect box = cv::boundingRect(mask);
    if (box.empty()) {
        return BoundingBox{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    }
    return BoundingBox{static_cast<double>(box.x), static_cast<double>(box.y),
                        static_cast<double>(box.x + box.width - 1), static_cast<double>(box.y + box.height - 1)};
}

std::vector<GateDetection> detectGates(const std::map<std::string, Gate>& gates, const DronePose& dronePos,
                                        const Transform& tBaseCam, const cv::Mat& cameraMatrix,
                                        const cv::Mat& distCoeffs, int imageWidth, int imageHeight, bool fisheye,
                                        double thetaMax, int minVisibleCorners) {
    const Transform tWorldBase =
        poseToTransform(dronePos.x, dronePos.y, dronePos.z, dronePos.roll, dronePos.pitch, dronePos.yaw);
    const Transform tWorldCam = compose(tWorldBase, tBaseCam);
    const Transform tCamWorld = invert(tWorldCam);
    const Eigen::Vector3d camPosWorld = tWorldCam.t;

    // Gates that cannot put a single pixel in the image are skipped before the
    // expensive part below, rasterizing their silhouette at full resolution. At
    // `minVisibleCorners == 0` nothing else filters them, so without this every
    // gate on the track -- the ones behind the camera included -- is drawn and
    // then thrown away for having an empty mask.
    //
    // The test is the gate's bounding sphere against the {theta <= thetaMax}
    // cone, and skipping on it changes nothing only if no point past thetaMax
    // lands in the image. Pinhole guarantees that by clipping to the cone in 3D.
    // A fisheye does as long as theta_d keeps growing past thetaMax, which is
    // where theta_d reaches the image corners; a calibration whose polynomial
    // turns back would fold far-off geometry into view, so there it is not used.
    bool canCull = true;
    if (fisheye) {
        std::array<double, 4> k{0.0, 0.0, 0.0, 0.0};
        for (int i = 0; i < std::min(4, static_cast<int>(distCoeffs.total())); ++i) {
            k[i] = distCoeffs.at<double>(i);
        }
        const auto thetaD = [&k](double t) {
            const double t2 = t * t;
            return t * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
        };
        constexpr int kSamples = 256;
        double prev = thetaD(thetaMax);
        for (int i = 1; i <= kSamples && canCull; ++i) {
            const double curr = thetaD(thetaMax + (CV_PI - thetaMax) * i / kSamples);
            canCull = curr > prev;
            prev = curr;
        }
    }

    std::vector<Candidate> candidates;
    for (const auto& [name, entry] : gates) {
        // One candidate per configured gate. A double is built from its two
        // squares, top first, leaving out the outer corners where they meet --
        // the top square's bottom_*_outer and the bottom square's top_*_outer --
        // because the real frame is one piece there and has no such corners.
        const std::vector<Gate> parts = partsOf(entry);
        const bool isDouble = entry.shape == GateShape::Double;
        // The gate's own frame for `Keypoint::gateLocal`: origin at its pose, x
        // lateral, y up, z along its facing normal.
        const Eigen::Vector3d gateOrigin(entry.pose.x, entry.pose.y, entry.pose.z);
        const Eigen::Vector3d gateLateral(-std::sin(entry.pose.yaw), std::cos(entry.pose.yaw), 0.0);
        const Eigen::Vector3d gateNormal(std::cos(entry.pose.yaw), std::sin(entry.pose.yaw), 0.0);

        Candidate cand;
        cand.gateName = name;
        std::vector<GateFaces> partFaces;
        partFaces.reserve(parts.size());
        double depthSum = 0.0;
        int visibleCount = 0;
        for (size_t partIndex = 0; partIndex < parts.size(); ++partIndex) {
            const Gate& gate = parts[partIndex];
            const GatePose& gatePose = gate.pose;
            const GateDims& gateDims = gate.dims;
            partFaces.push_back(gateFaces(gate.shape, gatePose.x, gatePose.y, gatePose.z, gatePose.yaw,
                                          gateDims.outerSize, gateDims.innerSize, gateDims.thickness));
            const GateFaces& faces = partFaces.back();

            const Eigen::Vector3d center(gatePose.x, gatePose.y, gatePose.z);
            const Eigen::Vector3d normal(std::cos(gatePose.yaw), std::sin(gatePose.yaw), 0.0);
            const bool front = normal.dot(camPosWorld - center) < 0.0;

            const Polygon3d& outerFace = front ? faces.outerFaces[0] : faces.outerFaces[1];
            const Polygon3d& innerFace = front ? faces.innerFaces[0] : faces.innerFaces[1];
            const CornerLayout layout = cornerLayout(gate.shape);
            const int n = layout.count;
            const int* indexMap = front ? layout.frontIndexMap : layout.backIndexMap;

            // Canonical order: n inner corners, then n outer corners.
            Polygon3d cornersWorld;
            cornersWorld.reserve(2 * n);
            for (int i = 0; i < n; ++i) {
                cornersWorld.push_back(innerFace[indexMap[i]]);
            }
            for (int i = 0; i < n; ++i) {
                cornersWorld.push_back(outerFace[indexMap[i]]);
            }

            const Polygon3d cornersCam = toCameraFrame(cornersWorld, tCamWorld);
            const std::vector<cv::Point2d> projected = projectPoints(cornersCam, cameraMatrix, distCoeffs, fisheye);

            if (partIndex == 0) {
                // Outer corners (indices n..n+2) are coplanar with the inner ones by
                // construction (gateFaces() builds both from the same front/back
                // center using the same lateral/vertical axes), so this plane
                // describes the whole near face -- a double's two squares share it.
                cand.faceNormalCam =
                    (cornersCam[n + 1] - cornersCam[n]).cross(cornersCam[n + 2] - cornersCam[n]).normalized();
                cand.faceOffsetCam = cand.faceNormalCam.dot(cornersCam[n]);
            }

            const std::string prefix = !isDouble ? "" : partIndex == 0 ? "top_" : "bottom_";
            for (int i = 0; i < 2 * n; ++i) {
                if (isDouble && i >= n) {
                    const bool bottomCorner = std::string(layout.names[i % n]).rfind("bottom", 0) == 0;
                    if (bottomCorner == (partIndex == 0)) {
                        continue;  // an outer corner where the two squares meet
                    }
                }
                const Eigen::Vector3d& p = cornersCam[i];
                const double theta = std::acos(p.z() / p.norm());
                const bool inCone = theta < thetaMax;
                const bool inBounds = projected[i].x >= 0 && projected[i].x < imageWidth && projected[i].y >= 0 &&
                                       projected[i].y < imageHeight;
                const bool visible = inCone && inBounds;

                const std::string suffix = i < n ? "_inner" : "_outer";
                Keypoint keypoint{prefix + layout.names[i % n] + suffix, projected[i].x, projected[i].y, visible,
                                  /*inFrustum=*/visible};
                keypoint.world = cornersWorld[i];
                const Eigen::Vector3d offset = cornersWorld[i] - gateOrigin;
                keypoint.gateLocal = Eigen::Vector3d(offset.dot(gateLateral), offset.z(), offset.dot(gateNormal));
                cand.keypoints.push_back(std::move(keypoint));
                cand.camPoints.push_back(p);
                if (visible) {
                    ++visibleCount;
                }
                depthSum += p.z();
            }
        }
        if (visibleCount < minVisibleCorners) {
            continue;
        }
        if (canCull) {
            // Skipped only when every frame of the gate is out of view.
            bool inView = false;
            for (const Gate& gate : parts) {
                const GateDims& gateDims = gate.dims;
                const Eigen::Vector3d center(gate.pose.x, gate.pose.y, gate.pose.z);
                // Outer ring's circumradius squared, over outerSize squared: a
                // square's half-diagonal, an octagon's 1 / (2 + sqrt 2).
                const double circumFactor = gate.shape == GateShape::Octagon ? 1.0 / (2.0 + std::sqrt(2.0)) : 0.5;
                const double gateRadius = std::sqrt(circumFactor * gateDims.outerSize * gateDims.outerSize +
                                                    0.25 * gateDims.thickness * gateDims.thickness);
                const Eigen::Vector3d centerCam = tCamWorld.R * center + tCamWorld.t;
                const double dist = centerCam.norm();
                if (!(dist > gateRadius &&
                      std::atan2(std::hypot(centerCam.x(), centerCam.y()), centerCam.z()) -
                              std::asin(gateRadius / dist) >
                          thetaMax)) {
                    inView = true;
                    break;
                }
            }
            if (!inView) {
                continue;
            }
        }

        // This gate's full rendered silhouette (all outer faces OR-ed minus
        // all inner faces AND-ed, subdivided + cone-clipped) -- identical to
        // what `renderPose` would draw for this gate alone, and for a double
        // the union of its squares'. Used below so occlusion tests against the
        // true curved/extruded shape rather than a straight-line approximation
        // of just the near face.
        for (size_t partIndex = 0; partIndex < parts.size(); ++partIndex) {
            const Gate& gate = parts[partIndex];
            const GateFaces& faces = partFaces[partIndex];
            GateFacesPx gatePx;
            gatePx.cameraInAperture = cameraInAperture(gate.pose, gate.shape, gate.dims, camPosWorld);
            gatePx.outerFacesPx.reserve(faces.outerFaces.size());
            for (const auto& face : faces.outerFaces) {
                gatePx.outerFacesPx.push_back(
                    projectFaceClipped(face, tCamWorld, cameraMatrix, distCoeffs, thetaMax, fisheye, imageWidth,
                                        imageHeight));
            }
            gatePx.innerFacesPx.reserve(faces.innerFaces.size());
            for (const auto& face : faces.innerFaces) {
                gatePx.innerFacesPx.push_back(
                    projectFaceClipped(face, tCamWorld, cameraMatrix, distCoeffs, thetaMax, fisheye, imageWidth,
                                        imageHeight));
            }
            const cv::Mat partMask = singleGateMask(gatePx, imageWidth, imageHeight);
            if (partIndex == 0) {
                cand.footprint = partMask;
            } else {
                cv::bitwise_or(cand.footprint, partMask, cand.footprint);
            }
        }

        cand.depth = depthSum / static_cast<double>(cand.keypoints.size());
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
            for (size_t k = 0; k < candidates[i].keypoints.size(); ++k) {
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

        // Free here: cv::Mat is refcounted, so this shares the buffer the
        // occlusion test already built rather than rendering the gate a second
        // time. `GateRenderer::renderDetections` resamples it to the output
        // resolution afterwards, which is where it stops being free.
        det.mask = cand.footprint;
        det.maskBoundingBox = boundingBoxOfMask(cand.footprint);

        // `minVisibleCorners == 0` means "everything in the picture", not
        // "every gate in the config". Without this the caller also gets the
        // gates behind the camera and off the far side of the track -- at 3
        // they are excluded by the corner count, but 0 excludes
        // nothing, and a gate that projects nowhere has an empty mask, an
        // empty box and only `inFrustum = false` keypoints, i.e. nothing to
        // describe it at all.
        //
        // This is the setting that makes the mask worth having: a gate can sit
        // close and off to one side so that every corner leaves the
        // `theta < thetaMax` cone while its frame still crosses the image.
        // Such a gate is dropped at any minVisibleCorners > 0, and the mask is
        // the only truthful description of it.
        if (minVisibleCorners == 0 && det.maskBoundingBox.x2 < det.maskBoundingBox.x1) {
            continue;
        }

        detections.push_back(std::move(det));
    }
    return detections;
}

}  // namespace detect_gates
