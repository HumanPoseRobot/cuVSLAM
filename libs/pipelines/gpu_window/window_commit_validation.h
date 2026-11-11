#pragma once
#include "map/keyframe.h"
namespace cuvslam::sba_imu {
inline bool SameCommitState(const map::State& a, const map::State& b) {
  return a.rig_from_w.matrix().allFinite() && a.velocity.allFinite() && a.gyro_bias.allFinite() &&
         a.acc_bias.allFinite() && a.rig_from_w.matrix().isApprox(b.rig_from_w.matrix(), 1e-6f) &&
         a.velocity.isApprox(b.velocity, 1e-6f) && a.gyro_bias.isApprox(b.gyro_bias, 1e-6f) &&
         a.acc_bias.isApprox(b.acc_bias, 1e-6f);
}
}  // namespace cuvslam::sba_imu
