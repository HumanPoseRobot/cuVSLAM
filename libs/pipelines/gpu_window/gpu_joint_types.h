#pragma once
#include <vector>
namespace gpu_vi {
// Row-major matrices, SI units, world_from_imu. Three local state blocks:
// left pose+velocity (9), right pose+velocity (9), shared gyro+acc bias (6).
struct Data {
  double R[2][9], p[2][3], v[2][3], bg[3], ba[3], bg0[3], ba0[3];
  double dR[9], dV[3], dP[3], JRg[9], JVg[9], JVa[9], JPg[9], JPa[9];
  double gravity[3], dt, imu_sqrt[81], prior_sqrt[225];
};
struct Observation {
  double point[3], Rci[9], tci[3], xy[2], sqrt_info[4], min_depth;
};
struct Evaluation {
  std::vector<float> imu, prior, visual, imu_j, prior_j, visual_j;
};
struct Result {
  float state[24]{};
  float initial_cost = 0, final_cost = 0;
  int iterations = 0;
  double buffer_ms = 0, graph_ms = 0, lm_ms = 0, copy_ms = 0;
};
Evaluation EvaluateGpu(const Data&, const std::vector<Observation>&, const float state[24]);
Result OptimizeGpu(const Data&, const std::vector<Observation>&, float huber, int iterations, bool reuse_lm,
                   bool fixed_order_normal_equations = false);
}  // namespace gpu_vi
