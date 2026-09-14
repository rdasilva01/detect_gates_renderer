// Camera projection helpers.
//
// Two models are supported, and they need very different handling.
//
// Fisheye (equidistant): a camera-frame point maps to pixel radius
// r = f * theta, theta being the angle from the optical axis (+Z). This is
// well defined for every theta in [0, pi) and maps the whole sphere into a
// bounded disk (r <= f * pi), so nothing needs clipping in 3D -- geometry
// past theta = 90 deg is projected where it belongs, which matters because a
// >180 deg fisheye genuinely sees behind its own image plane (the calibration
// in `config/` reaches theta = 105.8 deg at the image corners). Note that
// `cv::fisheye::projectPoints` cannot be used for this: it computes theta as
// atan(|xy| / z), which is only correct for z > 0 and folds anything past
// 90 deg back across the principal point.
//
// Pinhole (`fisheye=false`, used for rectified views): r = f * tan(theta) is
// unbounded as theta -> pi/2, so polygons are clipped in 3D against the
// circular cone {theta < thetaMax} for some thetaMax < pi/2 *before*
// projection. The cone is convex there, so clipping a convex polygon (e.g. a
// gate face) against it with Sutherland-Hodgman yields another convex,
// non-self-intersecting polygon, safe to project directly. That construction
// is exactly why it cannot be reused for a wide fisheye: {theta < thetaMax}
// stops being convex once thetaMax exceeds 90 deg.
#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "detect_gates/gates.hpp"

namespace detect_gates {

// A face's projected outline, in pixel coordinates.
//
// `inverted` says how to read it: normally the face covers the *interior* of
// the outline, but on the fisheye path a face that spans the direction
// straight behind the camera has its image turned inside out (that direction
// maps to the rim of the projection disk), so the face covers everything
// *outside* the outline instead. This happens routinely at a gate crossing,
// where the frame the camera just passed wraps around behind it.
struct FacePixels {
    std::vector<cv::Point2d> points;  // empty if the face projects to nothing
    bool inverted = false;
};

// Project camera-frame points to pixel coordinates.
//
// If `fisheye` is true, uses the equidistant model (r = f * theta) over the
// full theta range. Otherwise uses the standard pinhole model
// (r = f * tan(theta)), as used when rendering a rectified view.
std::vector<cv::Point2d> projectPoints(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix,
                                        const cv::Mat& distCoeffs, bool fisheye = true);

// Project a closed loop of camera-frame points (e.g. from `subdivideEdges`)
// into an outline ready to rasterize.
//
// Fisheye: projected with adaptive refinement, and `inverted` set as described
// on `FacePixels`. The outline is left as it falls, off-canvas coordinates and
// all -- it is bounded by f * pi, and an inverted outline legitimately encloses
// the whole image. `thetaMax` and `nConePlanes` are unused.
//
// Pinhole: the polygon is first clipped in 3D against the {theta < thetaMax}
// cone (`thetaMax` must be strictly less than pi/2), then projected, hulled,
// and clipped to the image rectangle. Returns an empty outline if nothing
// remains.
FacePixels projectPolygon(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                           int imageWidth, int imageHeight, double thetaMax = 89.0 * CV_PI / 180.0,
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

// Max angle-from-optical-axis covered by a *fisheye* image of this size.
//
// The equidistant counterpart of `rectifiedThetaMax`: inverts the distortion
// polynomial at the image corners. Unlike the pinhole case this routinely
// exceeds 90 deg (105.8 deg for the calibration in `config/`), so it is not a
// clipping bound -- it is the real angular extent of the image, used to decide
// whether a point is inside the field of view.
double fisheyeThetaMax(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, int imageWidth, int imageHeight);

}  // namespace detect_gates
