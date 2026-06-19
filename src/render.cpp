#include "detect_gates/render.hpp"

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

}  // namespace

cv::Mat singleGateMask(const GateFacesPx& gate, int imageWidth, int imageHeight) {
    cv::Mat outerMask = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
    for (const auto& facePx : gate.outerFacesPx) {
        if (!facePx) {
            continue;
        }
        cv::fillPoly(outerMask, std::vector<std::vector<cv::Point>>{toIntPoints(*facePx)}, cv::Scalar(255));
    }

    cv::Mat innerMask(imageHeight, imageWidth, CV_8UC1, cv::Scalar(255));
    for (const auto& facePx : gate.innerFacesPx) {
        if (!facePx) {
            innerMask.setTo(cv::Scalar(0));
            break;
        }
        cv::Mat faceMask = cv::Mat::zeros(imageHeight, imageWidth, CV_8UC1);
        cv::fillPoly(faceMask, std::vector<std::vector<cv::Point>>{toIntPoints(*facePx)}, cv::Scalar(255));
        cv::bitwise_and(innerMask, faceMask, innerMask);
    }

    cv::Mat gateMask;
    cv::bitwise_not(innerMask, innerMask);
    cv::bitwise_and(outerMask, innerMask, gateMask);
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
