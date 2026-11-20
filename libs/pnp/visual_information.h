#pragma once
#include <Eigen/Geometry>
namespace cuvslam::pnp {
// Relative IRLS attenuation of already calibrated measurement information.
// Solver-wide averaging and loss amplitude must not change sensor covariance.
inline double VisualInformationWeight(double squared_whitened_error, double c) {
  return 1. / (1. + c * squared_whitened_error);
}
inline Eigen::Matrix3d InformationSkew(const Eigen::Vector3d& p) {
  Eigen::Matrix3d S;
  S << 0., -p.z(), p.y(), p.z(), 0., -p.x(), -p.y(), p.x(), 0.;
  return S;
}
// PnP T_r_w uses a right world twist. Inertial pose uses body rotation
// and body translation: R'=R Exp(a), p'=p+R b. At zero, inversion gives
// delta_world = -Ad(T_w_i) [a,b]. The common minus cancels in information.
inline Eigen::Matrix<double, 6, 6> WorldTwistToBodyInformation(const Eigen::Matrix<double, 6, 6>& H,
                                                               const Eigen::Isometry3d& world_from_body) {
  Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
  A.topLeftCorner<3, 3>() = world_from_body.linear();
  A.bottomRightCorner<3, 3>() = world_from_body.linear();
  A.bottomLeftCorner<3, 3>() = InformationSkew(world_from_body.translation()) * world_from_body.linear();
  const Eigen::Matrix<double, 6, 6> B = A.transpose() * H * A;
  return .5 * (B + B.transpose());
}
}  // namespace cuvslam::pnp
