#pragma once
#include "pnp/geometry_profile.h"
namespace cuvslam::pnp {
struct JointGeometryReport {
  bool covariance_only = false;
  Eigen::Matrix<double, 6, 1> profile_gradient = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::MatrixXd navigation_covariance, point_pose_covariance;
  size_t historical_poses = 0, points = 0, missing_history = 0;
};
bool RefineJointGeometry(const map::UnifiedMap&, const camera::Rig&, const std::unordered_map<TrackId, Vector3T>&,
                         const std::vector<camera::Observation>&, size_t required, int64_t time_ns, Isometry3T&,
                         Matrix6T*, JointGeometryReport* = nullptr);
}  // namespace cuvslam::pnp
