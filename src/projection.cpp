#include "detect_gates/projection.hpp"

#include <cmath>

#include <opencv2/calib3d.hpp>

namespace detect_gates {

namespace {

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

}  // namespace

std::vector<cv::Point2d> projectPoints(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix,
                                        const cv::Mat& distCoeffs, bool fisheye) {
    std::vector<cv::Point3d> objPts;
    objPts.reserve(ptsCam.size());
    for (const auto& p : ptsCam) {
        objPts.emplace_back(p.x(), p.y(), p.z());
    }

    static const cv::Mat kZero3 = cv::Mat::zeros(3, 1, CV_64F);

    std::vector<cv::Point2d> imgPts;
    if (fisheye) {
        cv::fisheye::projectPoints(objPts, imgPts, kZero3, kZero3, cameraMatrix, distCoeffs);
    } else {
        cv::projectPoints(objPts, kZero3, kZero3, cameraMatrix, distCoeffs, imgPts);
    }
    return imgPts;
}

std::optional<std::vector<cv::Point2d>> projectPolygon(const Polygon3d& ptsCam, const cv::Mat& cameraMatrix,
                                                         const cv::Mat& distCoeffs, double thetaMax,
                                                         int nConePlanes, bool fisheye) {
    Polygon3d clipped = ptsCam;
    for (const auto& normal : coneNormals(thetaMax, nConePlanes)) {
        clipped = clipAgainstPlane(clipped, normal);
        if (clipped.empty()) {
            return std::nullopt;
        }
    }

    return projectPoints(clipped, cameraMatrix, distCoeffs, fisheye);
}

cv::Mat rectifiedCameraMatrix(const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, int imageWidth,
                               int imageHeight, double balance) {
    cv::Mat newCameraMatrix;
    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs,
                                                             cv::Size(imageWidth, imageHeight), cv::Mat::eye(3, 3, CV_64F),
                                                             newCameraMatrix, balance);
    return newCameraMatrix;
}

double rectifiedThetaMax(const cv::Mat& newCameraMatrix, int imageWidth, int imageHeight) {
    const double fx = newCameraMatrix.at<double>(0, 0);
    const double fy = newCameraMatrix.at<double>(1, 1);
    const double cx = newCameraMatrix.at<double>(0, 2);
    const double cy = newCameraMatrix.at<double>(1, 2);

    const cv::Point2d corners[4] = {{0, 0}, {static_cast<double>(imageWidth), 0}, {0, static_cast<double>(imageHeight)},
                                     {static_cast<double>(imageWidth), static_cast<double>(imageHeight)}};

    double maxAngle = 0.0;
    for (const auto& corner : corners) {
        const double dx = (corner.x - cx) / fx;
        const double dy = (corner.y - cy) / fy;
        const double angle = std::atan(std::hypot(dx, dy));
        maxAngle = std::max(maxAngle, angle);
    }
    return maxAngle;
}

}  // namespace detect_gates
