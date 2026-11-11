#pragma once
#include <cmath>
#include <vector>
#include "imu/imu_sba_problem.h"
namespace cuvslam::pipelines {
inline bool ValidInitRotation(const Matrix3T& r) {
  return r.allFinite() && (r.transpose() * r - Matrix3T::Identity()).norm() < 1e-3f &&
         std::abs(r.determinant() - 1.f) < 1e-3f;
}
inline bool ValidInitInterval(int64_t left, int64_t right, float integrated) {
  const double dt = (right - left) * 1e-9;
  return right > left && std::isfinite(integrated) && integrated > 0 &&
         std::abs(double(integrated) - dt) < std::max(1e-5, dt * 1e-5);
}
inline bool ValidInitCandidate(const std::vector<sba_imu::Pose>& poses, const Matrix3T& rg) {
  if (poses.size() < 2 || !ValidInitRotation(rg)) {
    return false;
  }
  for (const auto& p : poses) {
    if (!ValidInitRotation(p.w_from_imu.linear()) || !p.w_from_imu.translation().allFinite() ||
        !p.velocity.allFinite() || !p.gyro_bias.allFinite() || !p.acc_bias.allFinite()) {
      return false;
    }
  }
  return true;
}
inline bool InitVelocityAt(const sba_imu::Pose& anchor, const sba_imu::IMUPreintegration& pi, int64_t anchor_ns,
                           int64_t frame_ns, const Vector3T& gravity, Vector3T& velocity) {
  if (!gravity.allFinite() || !anchor.velocity.allFinite()) {
    return false;
  }
  if (frame_ns == anchor_ns) {
    velocity = anchor.velocity;
    return true;
  }
  if (!ValidInitInterval(anchor_ns, frame_ns, pi.dT_s) || !pi.dR.allFinite() || !pi.dV.allFinite()) {
    return false;
  }
  sba_imu::Pose predicted;
  if (!anchor.predict_pose(gravity, pi, predicted) || !predicted.velocity.allFinite()) {
    return false;
  }
  velocity = predicted.velocity;
  return true;
}

}  // namespace cuvslam::pipelines
