#pragma once
#include <cmath>
#include "gpu_joint_jet.h"
#include "gpu_joint_types.h"
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
namespace gpu_vi {
template <class A, class B, class C>
HD inline void mul(const A* a, const B* b, C* c) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      c[i * 3 + j] = 0;
      for (int k = 0; k < 3; ++k) c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
    }
}
template <class A, class B, class C>
HD inline void mv(const A* a, const B* b, C* c) {
  for (int i = 0; i < 3; ++i) {
    c[i] = 0;
    for (int k = 0; k < 3; ++k) c[i] += a[i * 3 + k] * b[k];
  }
}
template <class A, class B, class C>
HD inline void mtv(const A* a, const B* b, C* c) {
  for (int i = 0; i < 3; ++i) {
    c[i] = 0;
    for (int k = 0; k < 3; ++k) c[i] += a[k * 3 + i] * b[k];
  }
}
template <class A, class B>
HD inline void trans(const A* a, B* b) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) b[3 * i + j] = a[3 * j + i];
}
template <class T>
HD inline void exp3(const T* x, T* r) {
  T s = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  T th = sqrt(s);
  T a = s < 1e-12 ? 1 - s / 6 : sin(th) / th, b = s < 1e-12 ? .5 - s / 24 : (1 - cos(th)) / s;
  T K[9] = {0, -x[2], x[1], x[2], 0, -x[0], -x[1], x[0], 0}, KK[9];
  mul(K, K, KK);
  for (int i = 0; i < 9; ++i) r[i] = (i % 4 == 0 ? 1. : 0.) + a * K[i] + b * KK[i];
}
template <class T>
HD inline void log3(const T* r, T* x) {
  T y[3] = {r[7] - r[5], r[2] - r[6], r[3] - r[1]};
  T sn = .5 * sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]), cs = .5 * (r[0] + r[4] + r[8] - 1);
  T theta = atan2(sn, cs);
  if (cs < -.999999) {
    int i = r[0] < r[4] ? 1 : 0;
    i = r[i * 4] < r[8] ? 2 : i;
    T axis[3] = {0, 0, 0};
    // Rodrigues: R_ii = cos(theta) + (1-cos(theta))*axis_i^2.
    // The former (R_ii+1)/2 approximation is exact only at pi and creates
    // a value/Jacobian discontinuity at the near-pi branch boundary.
    const T one_minus_cos = 1 - cs;
    axis[i] = sqrt((r[4 * i] - cs) / one_minus_cos);
    for (int j = 0; j < 3; ++j) {
      if (j != i) {
        axis[j] = (r[3 * i + j] + r[3 * j + i]) / (2 * one_minus_cos * axis[i]);
      }
    }
    T dot = axis[0] * y[0] + axis[1] * y[1] + axis[2] * y[2];
    if (dot < 0) {
      for (int j = 0; j < 3; ++j) axis[j] = -axis[j];
    }
    for (int j = 0; j < 3; ++j) x[j] = theta * axis[j];
    return;
  }
  T f = sn < 1e-10 ? .5 : theta / (2 * sn);
  for (int k = 0; k < 3; ++k) x[k] = f * y[k];
}
template <class T>
HD inline void pose(const Data& d, const T* x, int side, T* R, T* p, T* v) {
  T inc[9], dp[3];
  exp3(x, inc);
  mul(d.R[side], inc, R);
  mv(d.R[side], x + 3, dp);
  for (int k = 0; k < 3; ++k) {
    p[k] = d.p[side][k] + dp[k];
    v[k] = d.v[side][k] + x[6 + k];
  }
}
template <class T>
HD inline void raw_imu(const Data& d, const T* x, T* r) {
  T rl[9], rr[9], pl[3], pr[3], vl[3], vr[3];
  pose(d, x, 0, rl, pl, vl);
  pose(d, x + 9, 1, rr, pr, vr);
  T bg[3], ba[3];
  for (int k = 0; k < 3; ++k) {
    bg[k] = d.bg[k] + x[18 + k] - d.bg0[k];
    ba[k] = d.ba[k] + x[21 + k] - d.ba0[k];
  }
  T w[3], inc[9], dr[9], drt[9], rlt[9], tmp[9], err[9];
  mv(d.JRg, bg, w);
  exp3(w, inc);
  mul(d.dR, inc, dr);
  trans(dr, drt);
  trans(rl, rlt);
  mul(rlt, rr, tmp);
  mul(drt, tmp, err);
  log3(err, r);
  T dv[3], dp[3], jvg[3], jva[3], jpg[3], jpa[3], rv[3], rp[3];
  mv(d.JVg, bg, jvg);
  mv(d.JVa, ba, jva);
  mv(d.JPg, bg, jpg);
  mv(d.JPa, ba, jpa);
  for (int k = 0; k < 3; ++k) {
    dv[k] = vr[k] - vl[k] - d.gravity[k] * d.dt;
    dp[k] = pr[k] - pl[k] - vl[k] * d.dt - .5 * d.gravity[k] * d.dt * d.dt;
  }
  mtv(rl, dv, rv);
  mtv(rl, dp, rp);
  for (int k = 0; k < 3; ++k) {
    r[3 + k] = rv[k] - d.dV[k] - jvg[k] - jva[k];
    r[6 + k] = rp[k] - d.dP[k] - jpg[k] - jpa[k];
  }
}
template <class T>
HD inline void imu_res(const Data& d, const T* x, T* r) {
  T raw[9];
  raw_imu(d, x, raw);
  for (int i = 0; i < 9; ++i) {
    r[i] = 0;
    for (int j = 0; j < 9; ++j) r[i] += d.imu_sqrt[i * 9 + j] * raw[j];
  }
}
template <class T>
HD inline void prior_res(const Data& d, const T* x, T* r) {
  T raw[15];
  for (int i = 0; i < 9; ++i) raw[i] = x[i];
  for (int i = 0; i < 6; ++i) raw[9 + i] = x[18 + i];
  for (int i = 0; i < 15; ++i) {
    r[i] = 0;
    for (int j = 0; j < 15; ++j) r[i] += d.prior_sqrt[15 * i + j] * raw[j];
  }
}
template <class T>
HD inline void visual_res(const Data& d, const Observation& o, const T* x, T* r) {
  T R[9], p[3], v[3], delta[3], pi[3], pc[3];
  pose(d, x + 9, 1, R, p, v);
  for (int k = 0; k < 3; ++k) delta[k] = o.point[k] - p[k];
  mtv(R, delta, pi);
  mv(o.Rci, pi, pc);
  for (int k = 0; k < 3; ++k) pc[k] += o.tci[k];
  if (pc[2] <= o.min_depth) {
    r[0] = r[1] = 0;
    return;
  }
  T e[2] = {pc[0] / pc[2] - o.xy[0], pc[1] / pc[2] - o.xy[1]};
  r[0] = o.sqrt_info[0] * e[0] + o.sqrt_info[1] * e[1];
  r[1] = o.sqrt_info[2] * e[0] + o.sqrt_info[3] * e[1];
}
}  // namespace gpu_vi
#undef HD
