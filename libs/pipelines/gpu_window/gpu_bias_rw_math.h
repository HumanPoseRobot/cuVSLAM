#pragma once
#include "gpu_bias_rw_types.h"
#include "gpu_joint_math_ad.h"
#ifdef __CUDACC__
#define RW_HD __host__ __device__
#else
#define RW_HD
#endif
namespace gpu_bias_rw {
using gpu_vi::imu_res;
using gpu_vi::Jet;
using gpu_vi::prior_res;
using gpu_vi::visual_res;
template <class T>
RW_HD void rw_res(const Data& d, const T* x, T* r) {
  T delta[6];
  for (int k = 0; k < 3; ++k) {
    delta[k] = d.right_bg[k] + x[24 + k] - d.bg[k] - x[18 + k];
    delta[3 + k] = d.right_ba[k] + x[27 + k] - d.ba[k] - x[21 + k];
  }
  for (int i = 0; i < 6; ++i) {
    r[i] = 0;
    for (int j = 0; j < 6; ++j) r[i] += d.rw_sqrt[6 * i + j] * delta[j];
  }
}
}  // namespace gpu_bias_rw
#undef RW_HD
