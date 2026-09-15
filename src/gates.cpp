#include "detect_gates/gates.hpp"

#include <cmath>

namespace detect_gates {

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

// Regular octagon, `halfSize` from the centre to each flat, CCW in the gate's
// local frame from (-halfSide, -halfSize) -- the same start and winding as
// `squareCorners`.
Polygon3d octagonCorners(const Eigen::Vector3d& center, double halfSize, const Eigen::Vector3d& lateral,
                          const Eigen::Vector3d& vertical) {
    const double h = halfSize;
    const double a = halfSize * (std::sqrt(2.0) - 1.0);  // half a side
    const double local[8][2] = {{-a, -h}, {a, -h}, {h, -a}, {h, a}, {a, h}, {-a, h}, {-h, a}, {-h, -a}};
    Polygon3d corners;
    corners.reserve(8);
    for (const auto& p : local) {
        corners.push_back(center + p[0] * lateral + p[1] * vertical);
    }
    return corners;
}

Polygon3d ringCorners(GateShape shape, const Eigen::Vector3d& center, double halfSize, const Eigen::Vector3d& lateral,
                       const Eigen::Vector3d& vertical) {
    return shape == GateShape::Octagon ? octagonCorners(center, halfSize, lateral, vertical)
                                       : squareCorners(center, halfSize, lateral, vertical);
}

}  // namespace

GateFaces gateFaces(GateShape shape, double x, double y, double z, double yaw, double outerSize, double innerSize,
                     double thickness) {
    const Eigen::Vector3d center(x, y, z);
    const Eigen::Vector3d lateral(-std::sin(yaw), std::cos(yaw), 0.0);
    const Eigen::Vector3d vertical(0.0, 0.0, 1.0);
    const Eigen::Vector3d normal(std::cos(yaw), std::sin(yaw), 0.0);

    const Eigen::Vector3d frontCenter = center - normal * (thickness / 2.0);
    const Eigen::Vector3d backCenter = center + normal * (thickness / 2.0);

    const double outerHalf = outerSize / 2.0;
    const double innerHalf = innerSize / 2.0;

    const Polygon3d frontOuter = ringCorners(shape, frontCenter, outerHalf, lateral, vertical);
    const Polygon3d backOuter = ringCorners(shape, backCenter, outerHalf, lateral, vertical);
    const Polygon3d frontInner = ringCorners(shape, frontCenter, innerHalf, lateral, vertical);
    const Polygon3d backInner = ringCorners(shape, backCenter, innerHalf, lateral, vertical);

    GateFaces faces;
    faces.outerFaces.push_back(frontOuter);
    faces.outerFaces.push_back(backOuter);
    const int n = static_cast<int>(frontOuter.size());
    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;
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

}  // namespace detect_gates
