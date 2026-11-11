#pragma once
#include "imu/soft_inertial_pnp.h"
namespace cuvslam::sba_imu {
void AuditGpuJoint(const imu::ImuCalibration&, const StereoPnPInput&, const Pose&, const Pose&, float);
bool SolveGpuJoint(const imu::ImuCalibration&, const StereoPnPInput&, Pose&, Pose&, float, bool reuse_lm = true);
}  // namespace cuvslam::sba_imu
