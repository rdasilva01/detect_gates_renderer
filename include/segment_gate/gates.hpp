// Gate geometry: 3D corner points.
#pragma once

#include <vector>

#include <Eigen/Core>

namespace segment_gate {

// A closed polygon as an ordered list of 3D points.
using Polygon3d = std::vector<Eigen::Vector3d>;

struct GateFaces {
    // 6 faces of the outer box: front outer square, back outer square, then the 4 side walls.
    std::vector<Polygon3d> outerFaces;
    // Front/back inner squares bounding the through-hole.
    std::vector<Polygon3d> innerFaces;
};

// Return world-frame face polygons for a gate.
//
// The gate is a square frame (outer square minus inner square) centered at
// (x, y, z), extruded by `thickness` along its facing direction (the world
// XY heading given by yaw), symmetric about the configured pose. The
// frame's lateral/vertical axes (in-plane) are as in the old flat-frame
// model: lateral is perpendicular to yaw in the world XY-plane, vertical is
// world Z.
GateFaces gateFaces(double x, double y, double z, double yaw, double outerSize, double innerSize,
                     double thickness = 0.0);

// Sample points along a closed polygon's perimeter, each edge split into nSegments.
//
// The 3D gate edges are straight, but their fisheye projections are curved.
// Projecting only the corners and connecting them with straight pixel edges
// can turn a square into a self-intersecting shape at oblique viewing
// angles. Subdividing each edge before projection lets the rendered polygon
// follow the true curved projection.
Polygon3d subdivideEdges(const Polygon3d& corners, int nSegments = 8);

}  // namespace segment_gate
