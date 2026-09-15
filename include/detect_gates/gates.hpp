// Gate geometry: 3D corner points.
#pragma once

#include <vector>

#include <Eigen/Core>

namespace detect_gates {

// A closed polygon as an ordered list of 3D points.
using Polygon3d = std::vector<Eigen::Vector3d>;

// Outline of a gate's frame, both outer edge and aperture.
//
// Square: sides of `size`. Octagon: regular, its flats on the gate's lateral
// and vertical axes, `size` across flats -- so both fit the same size x size
// box.
enum class GateShape { Square, Octagon };

struct GateFaces {
    // Outer box: front outer ring, back outer ring, then one side wall per ring edge.
    std::vector<Polygon3d> outerFaces;
    // Front/back inner rings bounding the through-hole.
    std::vector<Polygon3d> innerFaces;
};

// Return world-frame face polygons for a gate.
//
// The gate is a frame (outer ring minus inner ring, both of `shape`)
// centered at (x, y, z), extruded by `thickness` along its facing direction
// (the world XY heading given by yaw), symmetric about the configured pose.
// The frame's lateral/vertical axes (in-plane) are as in the old flat-frame
// model: lateral is perpendicular to yaw in the world XY-plane, vertical is
// world Z.
GateFaces gateFaces(GateShape shape, double x, double y, double z, double yaw, double outerSize, double innerSize,
                     double thickness = 0.0);

// Sample points along a closed polygon's perimeter, each edge split into nSegments.
//
// The 3D gate edges are straight, but their fisheye projections are curved.
// Projecting only the corners and connecting them with straight pixel edges
// can turn a square into a self-intersecting shape at oblique viewing
// angles. Subdividing each edge before projection lets the rendered polygon
// follow the true curved projection.
Polygon3d subdivideEdges(const Polygon3d& corners, int nSegments = 8);

}  // namespace detect_gates
