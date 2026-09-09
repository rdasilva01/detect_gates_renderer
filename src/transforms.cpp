#include "detect_gates/transforms.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace detect_gates {

Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw) {
    const double cr = std::cos(roll), sr = std::sin(roll);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    const double cy = std::cos(yaw), sy = std::sin(yaw);

    Eigen::Matrix3d rx;
    rx << 1, 0, 0, 0, cr, -sr, 0, sr, cr;

    Eigen::Matrix3d ry;
    ry << cp, 0, sp, 0, 1, 0, -sp, 0, cp;

    Eigen::Matrix3d rz;
    rz << cy, -sy, 0, sy, cy, 0, 0, 0, 1;

    return rz * ry * rx;
}

Eigen::Vector3d matrixToRpy(const Eigen::Matrix3d& R) {
    const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
    const double cp = std::cos(pitch);

    double roll, yaw;
    if (std::abs(cp) > 1e-6) {
        roll = std::atan2(R(2, 1), R(2, 2));
        yaw = std::atan2(R(1, 0), R(0, 0));
    } else {
        roll = 0.0;
        yaw = std::atan2(-R(0, 1), R(1, 1));
    }
    return Eigen::Vector3d(roll, pitch, yaw);
}

Eigen::Matrix3d quaternionToMatrix(double w, double x, double y, double z) {
    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    if (norm < 1e-12) {
        throw std::invalid_argument("quaternionToMatrix: quaternion has zero norm");
    }
    w /= norm;
    x /= norm;
    y /= norm;
    z /= norm;

    Eigen::Matrix3d r;
    r << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w),
         2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
         2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y);
    return r;
}

Transform poseToTransform(double x, double y, double z, double roll, double pitch, double yaw) {
    return Transform{rpyToMatrix(roll, pitch, yaw), Eigen::Vector3d(x, y, z)};
}

Transform compose(const Transform& t1, const Transform& t2) {
    return Transform{t1.R * t2.R, t1.t + t1.R * t2.t};
}

Transform invert(const Transform& t) {
    const Eigen::Matrix3d rInv = t.R.transpose();
    return Transform{rInv, -rInv * t.t};
}

}  // namespace detect_gates
