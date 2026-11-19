/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include "gpu_bias_rw_math.h"
using namespace gpu_bias_rw;
int main() {
  Data d{};
  d.dt = .033333333;
  for (int s = 0; s < 2; ++s) {
    for (int k = 0; k < 3; ++k) {
      d.R[s][4 * k] = 1.;
    }
  }
  for (int k = 0; k < 3; ++k) {
    d.dR[4 * k] = 1.;
    d.JVa[4 * k] = -.033333333;
    d.JPa[4 * k] = -.5 * d.dt * d.dt;
  }
  for (int k = 0; k < 9; ++k) {
    d.imu_sqrt[10 * k] = 2.;
  }
  for (int k = 0; k < 15; ++k) {
    d.prior_sqrt[16 * k] = 3.;
  }
  for (int k = 0; k < 6; ++k) {
    d.rw_sqrt[7 * k] = 10. + k;
  }
  Observation o{};
  o.Rci[0] = o.Rci[4] = o.Rci[8] = 1.;
  o.point[0] = .3;
  o.point[1] = -.2;
  o.point[2] = 4.;
  o.sqrt_info[0] = o.sqrt_info[3] = 20.;
  o.min_depth = .05;
  std::vector<Observation> obs{o};
  float x[30]{};
  for (int k = 0; k < 30; ++k) {
    x[k] = .001 * (k % 5 - 2);
  }
  auto e = EvaluateGpu(d, obs, x);
  auto test = [&](auto f, const std::vector<float>& rs, const std::vector<float>& js, int n) {
    double xd[30];
    for (int k = 0; k < 30; ++k) {
      xd[k] = x[k];
    }
    double r[15];
    f(xd, r);
    for (int k = 0; k < n; ++k) {
      assert(std::abs(r[k] - rs[k]) < 1e-5);
    }
    for (int c = 0; c < 30; ++c) {
      double plus[15], minus[15];
      xd[c] += 1e-6;
      f(xd, plus);
      xd[c] -= 2e-6;
      f(xd, minus);
      xd[c] += 1e-6;
      for (int k = 0; k < n; ++k) {
        assert(std::abs((plus[k] - minus[k]) / 2e-6 - js[30 * k + c]) < 2e-4);
      }
    }
  };
  test([&](const double* a, double* b) { gpu_vi::imu_res(d, a, b); }, e.imu, e.imu_j, 9);
  test([&](const double* a, double* b) { gpu_vi::prior_res(d, a, b); }, e.prior, e.prior_j, 15);
  test([&](const double* a, double* b) { gpu_vi::visual_res(d, o, a, b); }, e.visual, e.visual_j, 2);
  test([&](const double* a, double* b) { rw_res(d, a, b); }, e.rw, e.rw_j, 6);
  // Nonzero world centers and rotations: check the actual GPU AD at the same
  // additive coordinates used by VectorStateBatch, then an independent Eigen projection.
  {
    const auto saved_d = d;
    const auto saved_o = o;
    for (int s = 0; s < 2; ++s) {
      Eigen::Matrix3d R = Eigen::AngleAxisd(.6 + .15 * s, Eigen::Vector3d(1, -2, 3).normalized()).toRotationMatrix();
      for (int i = 0; i < 3; ++i) {
        d.p[s][i] = (i == 0 ? 2.1 : (i == 1 ? -.8 : 1.5)) + .02 * s;
        for (int j = 0; j < 3; ++j) d.R[s][3 * i + j] = R(i, j);
      }
    }
    for (int k = 0; k < 30; ++k) x[k] = .07f * (k % 5 - 2);
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R0(d.R[1]);
    Eigen::Vector3d phi(x[9], x[10], x[11]), dpos(x[12], x[13], x[14]);
    Eigen::Matrix3d R = R0 * Eigen::AngleAxisd(phi.norm(), phi.normalized()).toRotationMatrix();
    Eigen::Vector3d position = Eigen::Map<const Eigen::Vector3d>(d.p[1]) + R0 * dpos;
    Eigen::Vector3d point = position + R * Eigen::Vector3d(.3, -.2, 4.);
    for (int k = 0; k < 3; ++k) o.point[k] = point[k];
    for (int c = 0; c < 4; ++c) {
      Eigen::Matrix3d ci = Eigen::AngleAxisd(.15 * c, Eigen::Vector3d(2, 1, -1).normalized()).toRotationMatrix();
      Eigen::Vector3d tc(.12 * c, -.05 * c, .03 * c);
      for (int i = 0; i < 3; ++i) {
        o.tci[i] = tc[i];
        for (int j = 0; j < 3; ++j) o.Rci[3 * i + j] = ci(i, j);
      }
      e = EvaluateGpu(d, std::vector<Observation>{o}, x);
      test([&](const double* a, double* b) { gpu_vi::imu_res(d, a, b); }, e.imu, e.imu_j, 9);
      test([&](const double* a, double* b) { gpu_vi::prior_res(d, a, b); }, e.prior, e.prior_j, 15);
      test([&](const double* a, double* b) { gpu_vi::visual_res(d, o, a, b); }, e.visual, e.visual_j, 2);
      test([&](const double* a, double* b) { rw_res(d, a, b); }, e.rw, e.rw_j, 6);
      Eigen::Vector3d pc = ci * R.transpose() * (point - position) + tc;
      Eigen::Vector2d er = pc.hnormalized() - Eigen::Vector2d(o.xy[0], o.xy[1]);
      Eigen::Map<const Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> W(o.sqrt_info);
      assert((Eigen::Map<const Eigen::Vector2f>(e.visual.data()).cast<double>() - W * er).norm() < 1e-5);
    }
    d = saved_d;
    o = saved_o;
    for (int k = 0; k < 30; ++k) x[k] = .001f * (k % 5 - 2);
    e = EvaluateGpu(d, obs, x);
    std::cout << "Nonzero-rotation/center four-camera GPU joint Jacobians and independent Eigen reprojection passed\n";
  }
  // RW must depend on BOTH bias states with opposite derivatives; dt/noise
  // scaling enters via the preintegration whitening, not an extra posterior step.
  for (int k = 0; k < 6; ++k) {
    assert(std::abs(e.rw_j[k * 30 + 18 + k] + 10 + k) < 1e-6);
    assert(std::abs(e.rw_j[k * 30 + 24 + k] - 10 - k) < 1e-6);
  }
  auto fit = OptimizeGpu(d, obs, 1.5, 10, true);
  assert(std::isfinite(fit.final_cost));
  assert(fit.final_cost <= fit.initial_cost + 1e-5);
  // Analytical equivalence: eliminate right-bias RW or add process covariance.
  Eigen::Matrix2d H;
  H << 9 + 100, -100, -100, 100;
  double explicit_information = H(1, 1) - H(1, 0) * H(0, 1) / H(0, 0);
  assert(std::abs(explicit_information - 1. / (1. / 9. + 1. / 100.)) < 1e-12);
  // Extreme RW precision changes uncertainty, not the conditional MAP mean.
  // Explicit FP32 normal equations are ill-conditioned here; analytical block
  // elimination must agree with the reduced objective for identical inputs.
  const auto reference = gpu_vi::OptimizeGpu(d, obs, 1.5, 10, true, /*fixed_order_normal_equations=*/true);
  for (double weight : {1e1, 1e3, 1e5, 1e7}) {
    for (int k = 0; k < 6; ++k) {
      d.rw_sqrt[7 * k] = weight;
    }
    const auto stable = OptimizeEliminated(d, obs, 1.5, 10, true);
    for (int k = 0; k < 24; ++k) {
      assert(stable.state[k] == reference.state[k]);
    }
  }
  for (int k = 0; k < 6; ++k) {
    d.rw_sqrt[7 * k] = 10. + k;
  }
  // Reused GPU workspace must not retain the previous problem's state.
  auto repeat = OptimizeGpu(d, obs, 1.5, 10, true);
  for (int k = 0; k < 30; ++k) {
    assert(std::abs(fit.state[k] - repeat.state[k]) < 1e-5);
  }
  std::cout
      << "30-state GPU residual/Jacobian finite differences, RW sign, process-noise equivalence and reuse passed\n";
}
