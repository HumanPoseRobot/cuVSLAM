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
  const auto reference = gpu_vi::OptimizeGpu(d, obs, 1.5, 10, true);
  for (double weight : {1e1, 1e3, 1e5, 1e7}) {
    for (int k = 0; k < 6; ++k) {
      d.rw_sqrt[7 * k] = weight;
    }
    const auto stable = OptimizeEliminated(d, obs, 1.5, 10, true);
    for (int k = 0; k < 24; ++k) {
      assert(std::abs(stable.state[k] - reference.state[k]) < 1e-5);
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
