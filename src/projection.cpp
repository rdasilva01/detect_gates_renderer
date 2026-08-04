#include "detect_gates/projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace detect_gates {

namespace {

std::array<double, 4> distortionCoeffs(const cv::Mat& distCoeffs) {
    std::array<double, 4> k{0.0, 0.0, 0.0, 0.0};
    const int n = std::min(4, static_cast<int>(distCoeffs.total()));
    for (int i = 0; i < n; ++i) {
        k[i] = distCoeffs.at<double>(i);
    }
    return k;
}

// Equidistant ("fisheye") projection over the full theta range.
//
// Identical to `cv::fisheye::projectPoints` for theta < 90 deg (verified to
// machine precision), but computes theta as atan2(|xy|, z) rather than
// atan(|xy| / z), so points at or behind the camera plane land on the correct
// side instead of being folded back across the principal point. See the note
// at the top of projection.hpp.
struct Equidistant {
    double fx, fy, cx, cy;
    std::array<double, 4> k;

    explicit Equidistant(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs)
        : fx(cameraMatrix.at<double>(0, 0)),
          fy(cameraMatrix.at<double>(1, 1)),
          cx(cameraMatrix.at<double>(0, 2)),
          cy(cameraMatrix.at<double>(1, 2)),
          k(distortionCoeffs(distCoeffs)) {}

    cv::Point2d operator()(const Eigen::Vector3d& p) const {
        const double r = std::hypot(p.x(), p.y());
        const double theta = std::atan2(r, p.z());
        const double t2 = theta * theta;
        const double thetaD = theta * (1.0 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
        // r == 0 is the optical axis: theta = 0 (the principal point) looking
        // forward, or theta = pi looking straight back, where the azimuth is
        // undefined and no finite pixel is correct. Both collapse to the
        // principal point; the latter is a measure-zero case that a subdivided
        // polygon vertex is not going to land on exactly.
        const double scale = r > 1e-12 ? thetaD / r : 0.0;
        return {fx * scale * p.x() + cx, fy * scale * p.y() + cy};
    }
};

std::vector<cv::Point2d> projectEquidistant(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix,
                                              const cv::Mat& distCoeffs) {
    const Equidistant project(cameraMatrix, distCoeffs);
    std::vector<cv::Point2d> imgPts;
    imgPts.reserve(ptsCam.size());
    for (const auto& p : ptsCam) {
        imgPts.push_back(project(p));
    }
    return imgPts;
}

// How far a projected midpoint may stray from the straight line between its
// neighbours before that span is bisected, and how many bisections a single
// input edge may take.
constexpr double kFlatnessPx = 0.5;
constexpr int kMaxRefineDepth = 10;

// Append the projection of the 3D segment a..b (excluding `a`, including `b`),
// bisecting until each span projects to something straight enough to fill with
// a line.
//
// Straight 3D edges project to curves, and how sharply they curve varies
// enormously: an edge metres away is near enough straight in pixels, while an
// edge that passes close to the direction straight behind the camera sweeps
// most of the azimuth over a few centimetres -- which is exactly what happens
// as the drone leaves a gate it has just crossed. A fixed subdivision cannot
// serve both: chosen for the first it cuts a chord across the second (drawing
// a large wedge of nonsense into the mask), and chosen for the second it costs
// hundreds of needless samples on every gate on the track.
void refineSegment(std::vector<cv::Point2d>& out, const Equidistant& project, const Eigen::Vector3d& a,
                    const cv::Point2d& pa, const Eigen::Vector3d& b, const cv::Point2d& pb, int depth) {
    if (depth > 0) {
        const Eigen::Vector3d mid = 0.5 * (a + b);
        const cv::Point2d pmid = project(mid);
        const cv::Point2d chordMid = 0.5 * (pa + pb);
        if (cv::norm(pmid - chordMid) > kFlatnessPx) {
            refineSegment(out, project, a, pa, mid, pmid, depth - 1);
            refineSegment(out, project, mid, pmid, b, pb, depth - 1);
            return;
        }
    }
    out.push_back(pb);
}

std::vector<cv::Point2d> projectEquidistantOutline(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix,
                                                     const cv::Mat& distCoeffs) {
    if (ptsCam.size() < 2) {
        return projectEquidistant(ptsCam, cameraMatrix, distCoeffs);
    }

    const Equidistant project(cameraMatrix, distCoeffs);
    std::vector<cv::Point2d> outline;
    outline.reserve(ptsCam.size());
    cv::Point2d prev = project(ptsCam.back());
    for (size_t i = 0; i < ptsCam.size(); ++i) {
        const cv::Point2d curr = project(ptsCam[i]);
        refineSegment(outline, project, ptsCam[(i + ptsCam.size() - 1) % ptsCam.size()], prev, ptsCam[i], curr,
                       kMaxRefineDepth);
        prev = curr;
    }
    return outline;
}

std::vector<Eigen::Vector3d> coneNormals(double thetaMax, int nPlanes) {
    const double cosTm = std::cos(thetaMax);
    const double sinTm = std::sin(thetaMax);
    std::vector<Eigen::Vector3d> normals;
    normals.reserve(nPlanes);
    for (int i = 0; i < nPlanes; ++i) {
        const double phi = i * (2.0 * CV_PI / nPlanes);
        normals.emplace_back(-cosTm * std::cos(phi), -cosTm * std::sin(phi), sinTm);
    }
    return normals;
}

// Sutherland-Hodgman clip of a closed polygon against the half-space {normal . p >= 0}.
Polygon3d clipAgainstPlane(const Polygon3d& points, const Eigen::Vector3d& normal) {
    if (points.empty()) {
        return points;
    }

    const size_t n = points.size();
    std::vector<double> dists(n);
    std::vector<bool> inside(n);
    for (size_t i = 0; i < n; ++i) {
        dists[i] = points[i].dot(normal);
        inside[i] = dists[i] >= 0;
    }

    Polygon3d output;
    for (size_t i = 0; i < n; ++i) {
        const size_t next = (i + 1) % n;
        const Eigen::Vector3d& curr = points[i];
        const Eigen::Vector3d& nxt = points[next];
        if (inside[i]) {
            output.push_back(curr);
        }
        if (inside[i] != inside[next]) {
            const double t = dists[i] / (dists[i] - dists[next]);
            output.push_back(curr + t * (nxt - curr));
        }
    }
    return output;
}

// Does this planar face span the direction straight behind the camera (-Z)?
//
// If it does, its equidistant image is inside out (see `FacePixels`): the -Z
// direction maps to the rim of the projection disk, so the face's outline
// bounds the part of the view the face does *not* cover.
bool coversBackwardAxis(const Polygon3d& face) {
    if (face.size() < 3) {
        return false;
    }

    // Newell's method: robust for the subdivided polygons this is called with,
    // where any three consecutive points may well be collinear.
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < face.size(); ++i) {
        normal += face[i].cross(face[(i + 1) % face.size()]);
    }
    if (normal.norm() < 1e-12) {
        return false;  // degenerate: the camera lies in the face's plane
    }
    normal.normalize();

    const Eigen::Vector3d back(0.0, 0.0, -1.0);
    const double denom = normal.dot(back);
    if (std::abs(denom) < 1e-12) {
        return false;  // the -Z ray runs parallel to the face
    }
    const double t = normal.dot(face[0]) / denom;
    if (t <= 0.0) {
        return false;  // the face's plane is crossed forwards, not backwards
    }

    // Point in (convex) polygon, in the face's own plane.
    const Eigen::Vector3d hit = t * back;
    bool anyPositive = false;
    bool anyNegative = false;
    for (size_t i = 0; i < face.size(); ++i) {
        const Eigen::Vector3d& a = face[i];
        const Eigen::Vector3d& b = face[(i + 1) % face.size()];
        const double side = (b - a).cross(hit - a).dot(normal);
        if (side > 1e-9) {
            anyPositive = true;
        } else if (side < -1e-9) {
            anyNegative = true;
        }
    }
    return !(anyPositive && anyNegative);
}

// A polygon that is convex in 3D can still project to a non-convex,
// inconsistently-wound 2D polygon under r = f * tan(theta), because the cone
// clip is a circular bound approximating a rectangular image (loose off the
// image diagonal, exact only at the 4 corners) and tan is highly nonlinear
// near the cone boundary. Normally harmless, but right at a gate crossing
// (camera within centimeters of the frame) some vertices swing from far below
// the canvas to far above it for a change of only a few centimeters in camera
// position -- exactly where a vertex's angle from the optical axis crosses the
// cone boundary in a non-diagonal direction. Filling such a polygon is
// ambiguous about which side to keep and can flip between covering the top or
// bottom (or left/right) half of the canvas.
//
// Fix: take the convex hull of the projected points. The polygon is convex in
// 3D by construction, so for any normal view the hull is a no-op -- it only
// changes anything in exactly this edge case, where it forces a single,
// unambiguous, correctly-wound convex shape.
//
// This is a pinhole-only remedy. The equidistant projection has no such
// blowup, and its outlines are legitimately non-convex once they span more
// than a hemisphere, so hulling them would be plain wrong.
std::vector<cv::Point2d> convexHull2d(const std::vector<cv::Point2d>& pts) {
    if (pts.size() < 3) {
        return pts;
    }
    std::vector<cv::Point2f> ptsF;
    ptsF.reserve(pts.size());
    for (const auto& p : pts) {
        ptsF.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }
    std::vector<cv::Point2f> hullF;
    cv::convexHull(ptsF, hullF);
    std::vector<cv::Point2d> hull;
    hull.reserve(hullF.size());
    for (const auto& p : hullF) {
        hull.emplace_back(p.x, p.y);
    }
    return hull;
}

// Clip a projected outline to the image rectangle, Sutherland-Hodgman against
// the 4 half-planes of [0, imageWidth] x [0, imageHeight].
//
// Pinhole only, and only after `convexHull2d`: the FOV cone bounds each point's
// angle from the optical axis, but that is a looser bound than the rectangular
// image, so a point right at the cone boundary can still land hundreds of
// pixels off-canvas at a gate crossing. Sutherland-Hodgman needs a convex
// subject to be trustworthy, which the hull guarantees here and which fisheye
// outlines are not.
std::vector<cv::Point2d> clipPolygonToRect(const std::vector<cv::Point2d>& pts, int imageWidth, int imageHeight) {
    static const auto insideLeft = [](const cv::Point2d& p, int, int) { return p.x; };
    static const auto insideRight = [](const cv::Point2d& p, int w, int) { return static_cast<double>(w) - p.x; };
    static const auto insideTop = [](const cv::Point2d& p, int, int) { return p.y; };
    static const auto insideBottom = [](const cv::Point2d& p, int, int h) { return static_cast<double>(h) - p.y; };
    using DistFn = double (*)(const cv::Point2d&, int, int);
    static const DistFn kEdges[4] = {insideLeft, insideRight, insideTop, insideBottom};

    std::vector<cv::Point2d> poly = pts;
    for (DistFn dist : kEdges) {
        if (poly.empty()) {
            break;
        }
        std::vector<cv::Point2d> clipped;
        const size_t n = poly.size();
        for (size_t i = 0; i < n; ++i) {
            const size_t next = (i + 1) % n;
            const cv::Point2d& curr = poly[i];
            const cv::Point2d& nxt = poly[next];
            const double dCurr = dist(curr, imageWidth, imageHeight);
            const double dNext = dist(nxt, imageWidth, imageHeight);
            const bool currIn = dCurr >= 0;
            const bool nextIn = dNext >= 0;
            if (currIn) {
                clipped.push_back(curr);
            }
            if (currIn != nextIn) {
                const double t = dCurr / (dCurr - dNext);
                clipped.push_back(curr + t * (nxt - curr));
            }
        }
        poly = std::move(clipped);
    }
    return poly;
}

}  // namespace

std::vector<cv::Point2d> projectPoints(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix,
                                        const cv::Mat& distCoeffs, bool fisheye) {
    if (fisheye) {
        return projectEquidistant(ptsCam, cameraMatrix, distCoeffs);
    }

    std::vector<cv::Point3d> objPts;
    objPts.reserve(ptsCam.size());
    for (const auto& p : ptsCam) {
        objPts.emplace_back(p.x(), p.y(), p.z());
    }

    static const cv::Mat kZero3 = cv::Mat::zeros(3, 1, CV_64F);

    std::vector<cv::Point2d> imgPts;
    cv::projectPoints(objPts, kZero3, kZero3, cameraMatrix, distCoeffs, imgPts);
    return imgPts;
}

FacePixels projectPolygon(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
                           int imageWidth, int imageHeight, double thetaMax, int nConePlanes, bool fisheye) {
    if (fisheye) {
        return FacePixels{projectEquidistantOutline(ptsCam, cameraMatrix, distCoeffs), coversBackwardAxis(ptsCam)};
    }

    Polygon3d clipped = ptsCam;
    for (const auto& normal : coneNormals(thetaMax, nConePlanes)) {
        clipped = clipAgainstPlane(clipped, normal);
        if (clipped.empty()) {
            return FacePixels{};
        }
    }

    const std::vector<cv::Point2d> hull =
        convexHull2d(projectPoints(clipped, cameraMatrix, distCoeffs, /*fisheye=*/false));
    std::vector<cv::Point2d> outline = clipPolygonToRect(hull, imageWidth, imageHeight);
    if (outline.size() < 3) {
        // Off-canvas, but not out of frame. An empty outline means "clipped out
        // of the field of view entirely", which `singleGateMask` skips; a face
        // that merely fell off the canvas has to keep contributing its (empty)
        // on-canvas region when the two apertures are intersected, or a gate
        // whose far aperture is off-screen would be given a hole it does not
        // visibly have. Hand back the unclipped hull instead -- convex and
        // bounded by the FOV cone, so fillPoly clips it away to nothing.
        outline = hull;
    }
    return FacePixels{std::move(outline), false};
}

cv::Mat rectifiedCameraMatrix(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, int imageWidth,
                               int imageHeight, double balance) {
    cv::Mat newCameraMatrix;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs,
                                                             cv::Size(imageWidth, imageHeight), cv::Mat::eye(3, 3, CV_64F),
                                                             newCameraMatrix, balance);
    return newCameraMatrix;
}

namespace {

// Largest normalized image radius over the 4 image corners. For the pinhole
// model that radius is tan(theta); for the equidistant model it is theta_d.
double maxCornerRadius(const cv::Mat& cameraMatrix, int imageWidth, int imageHeight) {
    const double fx = cameraMatrix.at<double>(0, 0);
    const double fy = cameraMatrix.at<double>(1, 1);
    const double cx = cameraMatrix.at<double>(0, 2);
    const double cy = cameraMatrix.at<double>(1, 2);

    const cv::Point2d corners[4] = {{0, 0}, {static_cast<double>(imageWidth), 0}, {0, static_cast<double>(imageHeight)},
                                     {static_cast<double>(imageWidth), static_cast<double>(imageHeight)}};

    double maxRadius = 0.0;
    for (const auto& corner : corners) {
        maxRadius = std::max(maxRadius, std::hypot((corner.x - cx) / fx, (corner.y - cy) / fy));
    }
    return maxRadius;
}

}  // namespace

double rectifiedThetaMax(const cv::Mat& newCameraMatrix, int imageWidth, int imageHeight) {
    return std::atan(maxCornerRadius(newCameraMatrix, imageWidth, imageHeight));
}

double fisheyeThetaMax(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, int imageWidth, int imageHeight) {
    const double thetaD = maxCornerRadius(cameraMatrix, imageWidth, imageHeight);
    const std::array<double, 4> k = distortionCoeffs(distCoeffs);

    // Invert theta_d(theta) by Newton. Undistorted (all k zero) this converges
    // on the first step, since theta_d == theta.
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
    return std::clamp(theta, 0.0, CV_PI);
}

}  // namespace detect_gates
