#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "window_fp64.h"
int main(int argc, char** argv) {
  using namespace gpu_window;
  using Clock = std::chrono::steady_clock;
  std::cout << "file,states,points,obs,cpu_cost,gpu_cost,max_state_difference,max_point_difference,cpu_linear_residual,"
               "gpu_linear_residual,cpu_ms,gpu_ms,cpu_iters,gpu_iters,cpu_failed,gpu_failed,legacy_cpu_cost,legacy_gpu_"
               "cost,legacy_state_difference,legacy_gpu_repeat_difference,first_step_max_difference,first_cost_"
               "difference,legacy_last_dp_difference_m,legacy_last_dv_difference_m_s,legacy_last_dbg_difference,legacy_"
               "last_dba_difference,legacy_cpu_dh_m,legacy_gpu_dh_m,fp64_cpu_dh_m,fp64_gpu_dh_m,legacy_gpu_repeat_last_"
               "dp_m,legacy_gpu_repeat_last_dh_m,fp64_gpu_repeat_max_difference\n";
  for (int a = 1; a < argc; ++a) {
    auto p = fp64::LoadProblem(argv[a]);
    auto start = Clock::now();
    auto c = fp64::MakeCpu(p);
    auto cr = fp64::Minimize(p, *c);
    double ct = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    start = Clock::now();
    auto g = fp64::MakeGpu(p);
    auto gr = fp64::Minimize(p, *g);
    double gt = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    auto lc = OptimizeCpu(p);
    auto lg = OptimizeSchur(p);
    auto repeat = OptimizeSchur(p);
    double legacy_difference = 0, repeat_difference = 0;
    for (int k = 0; k < 15 * p.count; ++k) {
      legacy_difference = std::max(legacy_difference, double(std::abs(lc.x[k] - lg.x[k])));
      repeat_difference = std::max(repeat_difference, double(std::abs(repeat.x[k] - lg.x[k])));
    }
    std::vector<double> zero(15 * p.count + 3 * p.points.size());
    double c0 = c->Evaluate(zero, true), g0 = g->Evaluate(zero, true);
    auto cs = c->Solve(1e-3), gs = g->Solve(1e-3);
    double first_diff = 0;
    if (!cs.valid || !gs.valid) {
      first_diff = 1e99;
    } else {
      for (size_t k = 0; k < cs.dx.size(); ++k) {
        first_diff = std::max(first_diff, std::abs(cs.dx[k] - gs.dx[k]));
      }
    }
    double state = 0, point = 0;
    for (size_t k = 0; k < cr.x.size(); ++k) {
      auto& d = k < size_t(15 * p.count) ? state : point;
      d = std::max(d, std::abs(cr.x[k] - gr.x[k]));
    }
    auto gr_repeat = fp64::Minimize(p, *g);
    double gpu64_repeat = 0;
    for (size_t k = 0; k < gr.x.size(); ++k) {
      gpu64_repeat = std::max(gpu64_repeat, std::abs(gr.x[k] - gr_repeat.x[k]));
    }
    int last = 15 * (p.count - 1);
    double grouped[4]{};
    for (int k = 0; k < 4; ++k) {
      for (int j = 0; j < 3; ++j) {
        double d = lc.x[last + 3 + 3 * k + j] - lg.x[last + 3 + 3 * k + j];
        grouped[k] += d * d;
      }
      grouped[k] = std::sqrt(grouped[k]);
    }
    double repeat_dp = 0;
    for (int j = 0; j < 3; ++j) {
      double d = repeat.x[last + 3 + j] - lg.x[last + 3 + j];
      repeat_dp += d * d;
    }
    repeat_dp = std::sqrt(repeat_dp);
    auto height = [&](const auto& x) {
      if (p.edges.empty()) {
        return 0.;
      }
      const auto& gravity = p.edges.front().data.gravity;
      const auto& R = p.nodes[p.count - 1].R;
      double norm = 0, h = 0;
      for (int i = 0; i < 3; ++i) {
        norm += gravity[i] * gravity[i];
        for (int j = 0; j < 3; ++j) {
          h -= gravity[i] * R[3 * i + j] * x[last + 3 + j];
        }
      }
      return h / std::sqrt(norm);
    };
    std::cout << std::setprecision(14) << argv[a] << ',' << p.count << ',' << p.points.size() << ',' << p.visuals.size()
              << ',' << cr.after << ',' << gr.after << ',' << state << ',' << point << ',' << cr.max_linear_residual
              << ',' << gr.max_linear_residual << ',' << ct << ',' << gt << ',' << cr.iterations << ',' << gr.iterations
              << ',' << cr.failed << ',' << gr.failed << ',' << lc.after << ',' << lg.after << ',' << legacy_difference
              << ',' << repeat_difference << ',' << first_diff << ',' << std::abs(c0 - g0) << ',' << grouped[0] << ','
              << grouped[1] << ',' << grouped[2] << ',' << grouped[3] << ',' << height(lc.x) << ',' << height(lg.x)
              << ',' << height(cr.x) << ',' << height(gr.x) << ',' << repeat_dp << ','
              << std::abs(height(repeat.x) - height(lg.x)) << ',' << gpu64_repeat << std::endl;
  }
}
