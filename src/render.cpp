#include "detect_gates/render.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace detect_gates {

namespace {

// The face polygons handed in here were clipped in *3D angle-space* against
// the FOV cone (projection.cpp), which keeps them convex there -- but the
// cone is a circular bound approximating a rectangular image (loose off the
// image diagonal, exact only at the 4 corners), and the tan(theta) pinhole
// projection is highly nonlinear near the cone boundary. So a polygon that's
// convex in 3D can project to a *non-convex, inconsistently-wound* 2D pixel
// polygon -- normally harmless (still roughly the right shape), but right at
// a gate crossing (camera within centimeters of the frame) some vertices
// swing from far below the canvas to far above it (or vice versa) for a
// change of only a few centimeters in camera position, since that's exactly
// where a vertex's angle from the optical axis crosses close to the cone
// boundary in a non-diagonal direction. Clipping such a non-convex polygon
// straight to the image rectangle is ambiguous about which side to keep,
// and can flip between filling the top or bottom (or left/right) half of
// the canvas -- observed as the mask suddenly covering the whole image.
//
// Fix: take the convex hull of the projected points first. The polygon is
// convex in 3D by construction, so for any normal (non-degenerate) view the
// hull is a no-op -- it only changes anything in exactly this edge case,
// where it forces a single, unambiguous, correctly-wound convex shape
// before clipping to the rectangle.
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

// The angular FOV-cone pre-clip (projection.cpp) bounds each point's angle
// from the optical axis, but that's a looser bound than the rectangular
// image: a point exactly at the cone boundary can still project far outside
// the canvas (its angle matches a corner's, but not its direction). That's
// harmless for points near the camera's usual operating distance, but right
// at a gate crossing -- camera within centimeters of the frame material --
// the clipped-but-still-off-canvas coordinates can reach hundreds of pixels
// beyond the image edges. Clip to the image rectangle in pixel space, after
// projection (and after convexHull2d above), so fillPoly only ever sees
// polygons that are actually within (or exactly on) the canvas.
std::vector<cv::Point2d> clipPolygonToRect(const std::vector<cv::Point2d>& pts, int imageWidth, int imageHeight) {
    // Sutherland-Hodgman against the 4 half-planes of [0, imageWidth] x [0, imageHeight].
    // Each entry is a "signed distance from the edge" function; >= 0 means "inside".
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

std::vector<cv::Point> toIntPoints(const std::vector<cv::Point2d>& pts) {
    std::vector<cv::Point> intPts;
    intPts.reserve(pts.size());
    for (const auto& p : pts) {
        intPts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }
    return intPts;
}

void fillClippedPoly(cv::Mat& canvas, const std::vector<cv::Point2d>& facePx, int imageWidth, int imageHeight) {
    const std::vector<cv::Point2d> hull = convexHull2d(facePx);
    const std::vector<cv::Point2d> clipped = clipPolygonToRect(hull, imageWidth, imageHeight);
    if (clipped.size() < 3) {
        return;
    }
    cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{toIntPoints(clipped)}, cv::Scalar(255));
}

}  // namespace

namespace {
// Above this fraction of the canvas, a single gate's silhouette is no
// longer a plausible view of its frame -- it means the camera is within
// centimeters of the gate plane (mid-crossing), where extreme perspective
// makes the front and back apertures project to non-overlapping regions
// (so the "hole" AND-subtraction finds nothing to cut out) and the tan(theta)
// pinhole projection's nonlinearity inflates the outer silhouette far beyond
// its true angular extent. A real, non-degenerate gate view tops out around
// 25-27% of the canvas (checked empirically across distances and off-axis
// angles); 50% leaves a wide margin. Past it, suppress the gate rather than
// paint a screen-filling white flash for the 1-2 frames the camera spends
// inside the frame's thickness.
constexpr double kMaxPlausibleGateCoverage = 0.5;
}  // namespace

cv::Mat singleGateMask(const GateFacesPx& gate, int imageWidth, int imageHeight) {
    cv::Mat outerMask = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
    for (const auto& facePx : gate.outerFacesPx) {
        if (!facePx) {
            continue;
        }
        fillClippedPoly(outerMask, *facePx, imageWidth, imageHeight);
    }

    // innerMask starts as "the whole image is potentially the see-through
    // hole" and each available inner face narrows it down via AND. A face
    // that's nullopt (fully outside the FOV cone -- routine right at a gate
    // crossing, when the camera is within centimeters of the frame and one
    // of the front/back inner squares ends up entirely behind it) used to
    // force the *whole* hole computation to "no hole" (i.e. render this
    // gate fully solid) -- exactly backwards for a crossing, where the
    // camera being unable to see one inner face is a symptom of being
    // embedded in the aperture, not evidence that there's no aperture.
    // Skipping the unavailable face instead means: with one face available,
    // its footprint alone approximates the hole; with neither available,
    // innerMask stays all-255 -> inverted to all-0 -> this gate contributes
    // nothing (hidden) rather than a solid white frame.
    cv::Mat innerMask(imageHeight, imageWidth, CV_8UC1, cv::Scalar(255));
    for (const auto& facePx : gate.innerFacesPx) {
        if (!facePx) {
            continue;
        }
        cv::Mat faceMask = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
        fillClippedPoly(faceMask, *facePx, imageWidth, imageHeight);
        cv::bitwise_and(innerMask, faceMask, innerMask);
    }

    cv::Mat gateMask;
    cv::bitwise_not(innerMask, innerMask);
    cv::bitwise_and(outerMask, innerMask, gateMask);

    const double coverage = cv::countNonZero(gateMask) / static_cast<double>(gateMask.total());
    if (coverage > kMaxPlausibleGateCoverage) {
        return cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
    }
    return gateMask;
}

cv::Mat renderSegmentation(const std::vector<GateFacesPx>& gatesPx, int imageWidth, int imageHeight) {
    cv::Mat canvas = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);

    for (const auto& gate : gatesPx) {
        cv::bitwise_or(canvas, singleGateMask(gate, imageWidth, imageHeight), canvas);
    }

    return canvas;
}

}  // namespace detect_gates
