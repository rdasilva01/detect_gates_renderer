// Fisheye projection helpers.
//
// The OpenCV fisheye (equidistant) model maps a camera-frame point to pixel
// radius r = f * theta, where theta is the angle between the point and the
// optical axis (+Z). This is only well-defined (and matches cv::fisheye's
// behavior) for theta < pi/2, i.e. points strictly in front of the camera and
// not too oblique. `projectPolygon` clips polygons against the circular cone
// {theta < thetaMax} (for some thetaMax < pi/2) *in 3D, before projection* -
// this cone is convex, so clipping a convex polygon (e.g. a gate's corners)
// against it with Sutherland-Hodgman always yields another convex, non-self-
// intersecting polygon, which can then be projected directly with no risk of
// aliasing or self-intersection artifacts, even when the camera is extremely
// close to (or "inside") the polygon.
#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

#include "detect_gates/gates.hpp"

namespace detect_gates {

// Project camera-frame points to pixel coordinates.
//
// If `fisheye` is true, uses the OpenCV fisheye (equidistant) model
// (r = f * theta). Otherwise uses the standard pinhole model
// (r = f * tan(theta)), as used when rendering a rectified view.
std::vector<cv::Point2d> projectPoints(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix,
                                        const cv::Mat& distCoeffs, bool fisheye = true);

// Project a closed loop of camera-frame points (e.g. from `subdivideEdges`).
//
// The polygon is first clipped (in 3D) against the {theta < thetaMax} cone,
// then projected. Returns std::nullopt if nothing remains (the polygon is
// entirely outside the cone, e.g. behind the camera or out of the field of
// view).
//
// For the pinhole model (`fisheye=false`), `thetaMax` must be strictly less
// than pi/2 (r = f * tan(theta) is unbounded as theta -> pi/2).
std::optional<std::vector<cv::Point2d>> projectPolygon(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix,
                                                         const cv::Mat& distCoeffs,
                                                         double thetaMax = 89.0 * CV_PI / 180.0,
                                                         int nConePlanes = 32, bool fisheye = true);

// New pinhole camera matrix for a fisheye-rectified view of the same image size.
//
// Mirrors `cv::fisheye::estimateNewCameraMatrixForUndistortRectify` with
// identity rectification rotation and `balance=0.0` (no black borders).
cv::Mat rectifiedCameraMatrix(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, int imageWidth,
                               int imageHeight, double balance = 0.0);

// Max angle-from-optical-axis covered by a rectified image of this size.
//
// Used as the clipping cone's `thetaMax` for pinhole projection, since
// `r = f * tan(theta)` is unbounded as theta -> pi/2.
double rectifiedThetaMax(const cv::Mat& newCameraMatrix, int imageWidth, int imageHeight);

}  // namespace detect_gates
