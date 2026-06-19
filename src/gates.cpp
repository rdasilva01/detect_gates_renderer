#include "segment_gate/gates.hpp"

#include <cmath>

namespace segment_gate {

namespace {

// Corner ordering in (lateral, vertical) sign pairs, CCW in the gate's local frame.
constexpr double kCornerSigns[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};

Polygon3d squareCorners(const Eigen::Vector3d& center, double halfSize, const Eigen::Vector3d& lateral,
                         const Eigen::Vector3d& vertical) {
    Polygon3d corners;
    corners.reserve(4);
    for (const auto& sign : kCornerSigns) {
        corners.push_back(center + sign[0] * halfSize * lateral + sign[1] * halfSize * vertical);
    }
    return corners;
}

}  // namespace

GateFaces gateFaces(double x, double y, double z, double yaw, double outerSize, double innerSize,
                     double thickness) {
    const Eigen::Vector3d center(x, y, z);
    const Eigen::Vector3d lateral(-std::sin(yaw), std::cos(yaw), 0.0);
    const Eigen::Vector3d vertical(0.0, 0.0, 1.0);
    const Eigen::Vector3d normal(std::cos(yaw), std::sin(yaw), 0.0);

    const Eigen::Vector3d frontCenter = center - normal * (thickness / 2.0);
    const Eigen::Vector3d backCenter = center + normal * (thickness / 2.0);

    const double outerHalf = outerSize / 2.0;
    const double innerHalf = innerSize / 2.0;

    const Polygon3d frontOuter = squareCorners(frontCenter, outerHalf, lateral, vertical);
    const Polygon3d backOuter = squareCorners(backCenter, outerHalf, lateral, vertical);
    const Polygon3d frontInner = squareCorners(frontCenter, innerHalf, lateral, vertical);
    const Polygon3d backInner = squareCorners(backCenter, innerHalf, lateral, vertical);

    GateFaces faces;
    faces.outerFaces.push_back(frontOuter);
    faces.outerFaces.push_back(backOuter);
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) % 4;
        faces.outerFaces.push_back(Polygon3d{frontOuter[i], frontOuter[j], backOuter[j], backOuter[i]});
    }

    faces.innerFaces.push_back(frontInner);
    faces.innerFaces.push_back(backInner);

    return faces;
}

Polygon3d subdivideEdges(const Polygon3d& corners, int nSegments) {
    const size_t n = corners.size();
    Polygon3d points;
    points.reserve(n * nSegments);
    for (size_t i = 0; i < n; ++i) {
        const Eigen::Vector3d& start = corners[i];
        const Eigen::Vector3d& end = corners[(i + 1) % n];
        for (int j = 0; j < nSegments; ++j) {
            const double t = static_cast<double>(j) / nSegments;
            points.push_back(start + t * (end - start));
        }
    }
    return points;
}

}  // namespace segment_gate
