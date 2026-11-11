#pragma once
#include "gpu_joint_adapter.h"
namespace cuvslam::sba_imu {
bool SolveGpuBiasRW(const imu::ImuCalibration&, const StereoPnPInput&, Pose&, Pose&, float);
}
