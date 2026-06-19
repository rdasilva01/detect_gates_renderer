// Render a gate-segmentation mask onto a grayscale canvas.
#pragma once

#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace detect_gates {

// A projected face polygon, or std::nullopt if it was clipped out entirely.
using FacePixels = std::optional<std::vector<cv::Point2d>>;

struct GateFacesPx {
    std::vector<FacePixels> outerFacesPx;
    std::vector<FacePixels> innerFacesPx;
};

// Render a single gate's silhouette mask: outer faces OR-ed together minus
// inner faces AND-ed together. This is the same per-gate footprint used by
// `renderSegmentation`, exposed separately so other callers (e.g. pose-mode
// occlusion) can test against the exact rendered silhouette rather than an
// approximation.
cv::Mat singleGateMask(const GateFacesPx& gate, int imageWidth, int imageHeight);

// Render a single-channel segmentation mask.
//
// `gatesPx` is a list of per-gate outer/inner face pixel polygons (already
// clipped/projected). Per gate, the outer faces are OR-ed together
// (approximating the extruded frame's silhouette) and the inner faces are
// AND-ed together (approximating the through-hole's visible opening) and
// subtracted. Gates are then OR-ed into the canvas, so a closer gate's hole
// never erases a farther gate's frame that's visible through it.
cv::Mat renderSegmentation(const std::vector<GateFacesPx>& gatesPx, int imageWidth, int imageHeight);

}  // namespace detect_gates
