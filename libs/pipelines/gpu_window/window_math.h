#pragma once
#include "gpu_joint_math_ad.h"
#include "window_types.h"
#ifdef __CUDACC__
#define WHD __host__ __device__
#else
#define WHD
#endif
namespace gpu_window {
using namespace gpu_vi;
template <class T>
WHD void pose(const State& a, const T* x, T* R, T* p, T* v) {
  T inc[9], dp[3];
  exp3(x, inc);
  mul(a.R, inc, R);
  mv(a.R, x + 3, dp);
  for (int k = 0; k < 3; ++k) {
    p[k] = a.p[k] + dp[k];
    v[k] = a.v[k] + x[6 + k];
  }
}
template <class T>
WHD void difference(const State& a, const State& anchor, const T* x, T* r) {
  T R[9], p[3], v[3], at[9], dR[9], dp[3];
  pose(a, x, R, p, v);
  trans(anchor.R, at);
  mul(at, R, dR);
  log3(dR, r);
  for (int k = 0; k < 3; ++k) {
    dp[k] = p[k] - anchor.p[k];
    r[6 + k] = v[k] - anchor.v[k];
    r[9 + k] = a.bg[k] + x[9 + k] - anchor.bg[k];
    r[12 + k] = a.ba[k] + x[12 + k] - anchor.ba[k];
  }
  mtv(anchor.R, dp, r + 3);
}
template <class T>
WHD void edge_res(const Edge& e, const State& l, const State& r, const T* x, T* y) {
  // Data anchors are packed from the same states as the window; each endpoint has its own bias.
  T z[24];
  for (int k = 0; k < 9; ++k) {
    z[k] = x[k];
    z[9 + k] = x[15 + k];
  }
  for (int k = 0; k < 6; ++k) {
    z[18 + k] = x[9 + k];
  }
  imu_res(e.data, z, y);
  T b[6];
  for (int k = 0; k < 3; ++k) {
    b[k] = r.bg[k] + x[24 + k] - l.bg[k] - x[9 + k];
    b[3 + k] = r.ba[k] + x[27 + k] - l.ba[k] - x[12 + k];
  }
  for (int i = 0; i < 6; ++i) {
    y[9 + i] = 0;
    for (int j = 0; j < 6; ++j) {
      y[9 + i] += e.rw[6 * i + j] * b[j];
    }
  }
}
template <class T>
WHD void visual_res(const Visual& o, const State& a, const double* point, const T* x, T* r) {
  T R[9], p[3], v[3], dp[3], pi[3], pc[3];
  pose(a, x, R, p, v);
  for (int k = 0; k < 3; ++k) {
    dp[k] = point[k] + x[15 + k] - p[k];
  }
  mtv(R, dp, pi);
  mv(o.obs.Rci, pi, pc);
  for (int k = 0; k < 3; ++k) {
    pc[k] += o.obs.tci[k];
  }
  // Invalid-depth observations cannot silently vanish: host acceptance also checks visibility.
  T z = pc[2] <= o.obs.min_depth ? T(o.obs.min_depth) : pc[2];
  T err[2] = {pc[0] / z - o.obs.xy[0], pc[1] / z - o.obs.xy[1]};
  for (int i = 0; i < 2; ++i) {
    r[i] = o.obs.sqrt_info[2 * i] * err[0] + o.obs.sqrt_info[2 * i + 1] * err[1];
  }
}
}  // namespace gpu_window
#undef WHD
