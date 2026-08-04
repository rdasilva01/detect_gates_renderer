#include "detect_gates/render.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace detect_gates {

namespace {

std::vector<cv::Point> toIntPoints(const std::vector<cv::Point2d>& pts) {
    std::vector<cv::Point> intPts;
    intPts.reserve(pts.size());
    for (const auto& p : pts) {
        intPts.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
    }
    return intPts;
}

// Outlines arrive ready to rasterize (projection.cpp), off-canvas coordinates
// and all, and OpenCV clips them against the canvas itself. Nothing is
// pre-clipped here: a fisheye outline is legitimately non-convex once it spans
// more than a hemisphere, and around a gate crossing it encloses the whole
// canvas -- the one configuration a polygon pre-clip is least able to express.
void fillOutline(cv::Mat& canvas, const std::vector<cv::Point2d>& outline, const cv::Scalar& color) {
    if (outline.size() < 3) {
        return;
    }
    cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{toIntPoints(outline)}, color);
}

// Rasterize a face whose image is inside out (`FacePixels::inverted`): it
// covers everything *except* the outline's interior. Start solid and punch
// the outline out.
void fillInvertedOutline(cv::Mat& canvas, const std::vector<cv::Point2d>& outline) {
    canvas.setTo(cv::Scalar(255));
    fillOutline(canvas, outline, cv::Scalar(0));
}

}  // namespace

namespace {
// Above this fraction of the canvas, a single gate's silhouette is no longer a
// plausible view of its frame: what it takes is for the camera to be level with
// the frame and inside its outer footprint but outside the aperture -- flown
// into a gate post, not through it -- where the frame really does fill the view
// and there is no useful label to emit. A real, non-degenerate gate view tops
// out around 25-27% of the canvas (checked empirically across distances and
// off-axis angles, crossings included); 50% leaves a wide margin. Past it,
// suppress the gate rather than paint a screen-filling white flash.
constexpr double kMaxPlausibleGateCoverage = 0.5;
}  // namespace

cv::Mat singleGateMask(const GateFacesPx& gate, int imageWidth, int imageHeight) {
    cv::Mat outerMask = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
    cv::Mat scratch;  // only needed for inverted faces, which are the exception
    for (const auto& facePx : gate.outerFacesPx) {
        if (facePx.points.empty()) {
            continue;
        }
        if (!facePx.inverted) {
            // OR-ing is just "fill 255", so these go straight onto the mask.
            fillOutline(outerMask, facePx.points, cv::Scalar(255));
            continue;
        }
        if (scratch.empty()) {
            scratch.create(imageHeight, imageWidth, CV_8UC1);
        }
        fillInvertedOutline(scratch, facePx.points);
        cv::bitwise_or(outerMask, scratch, outerMask);
    }

    // The see-through hole is the set of rays that make it all the way through
    // the frame, and how the two inner faces combine into it depends on which
    // side of them the camera is.
    //
    // From outside the gate a ray has to clear *both* apertures, so the two
    // faces intersect. From inside the frame's thickness -- the couple of
    // frames a crossing lasts -- a ray escapes if it clears *either* aperture,
    // so they union. Intersecting them there instead is what used to erase the
    // gate mid-crossing: at that range the front and back apertures project to
    // disjoint regions, their intersection comes out empty, the gate renders
    // solid, and the coverage guard below then blanks it outright.
    //
    // A face with no outline was clipped out entirely and is simply skipped
    // (the identity for either operator). That happens on the pinhole path,
    // where an aperture can fall outside the necessarily-narrower-than-180-deg
    // rectified view; on the fisheye path nothing is clipped in 3D, so an
    // aperture that has wrapped behind the camera still projects (inside out,
    // see `FacePixels`) and still cuts the right hole.
    const uint8_t identity = gate.cameraInAperture ? 0 : 255;
    cv::Mat innerMask(imageHeight, imageWidth, CV_8UC1, cv::Scalar(identity));
    for (const auto& facePx : gate.innerFacesPx) {
        if (facePx.points.empty()) {
            continue;
        }
        cv::Mat faceMask = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
        if (facePx.inverted) {
            fillInvertedOutline(faceMask, facePx.points);
        } else {
            fillOutline(faceMask, facePx.points, cv::Scalar(255));
        }
        if (gate.cameraInAperture) {
            cv::bitwise_or(innerMask, faceMask, innerMask);
        } else {
            cv::bitwise_and(innerMask, faceMask, innerMask);
        }
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
