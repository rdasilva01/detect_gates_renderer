// Rigid-body transform helpers using the REP-103 (ROS) roll-pitch-yaw convention.
//
// A pose is represented as a Transform{R, t}: R is a 3x3 rotation matrix, t is
// a length-3 translation vector, and a point p in the source frame maps to
// R * p + t in the destination frame.
#pragma once

#include <Eigen/Core>

namespace detect_gates {

struct Transform {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
};

// Build R = Rz(yaw) * Ry(pitch) * Rx(roll).
Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw);

// Inverse of rpyToMatrix: extract (roll, pitch, yaw) from R = Rz(yaw) * Ry(pitch) * Rx(roll).
Eigen::Vector3d matrixToRpy(const Eigen::Matrix3d& R);

// Rotation matrix from a quaternion, scalar first: (w, x, y, z).
//
// Same rotation `rpyToMatrix` describes and the same frames REP-103 uses --
// world ENU (x east, y north, z up), body FLU (x forward, y left, z up) -- so
// this is R_world_body, and quaternion and roll/pitch/yaw poses are
// interchangeable.
//
// Watch the ordering: this takes w first, but ROS's geometry_msgs/Quaternion
// lays its *fields* out x, y, z, w. The input need not be normalized.
Eigen::Matrix3d quaternionToMatrix(double w, double x, double y, double z);

// Build a transform from a 6-DOF pose.
Transform poseToTransform(double x, double y, double z, double roll, double pitch, double yaw);

// Compose two transforms: result maps points through t2 then t1.
Transform compose(const Transform& t1, const Transform& t2);

// Invert a transform.
Transform invert(const Transform& t);

}  // namespace detect_gates
