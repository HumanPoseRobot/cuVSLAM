#pragma once
#include "gpu_joint_types.h"
namespace gpu_bias_rw {
// Four blocks: two pose/velocity blocks and two independent bias blocks.
struct Data : gpu_vi::Data {
  double right_bg[3]{}, right_ba[3]{}, rw_sqrt[36]{};
};
using Observation = gpu_vi::Observation;
struct Evaluation {
  std::vector<float> imu, prior, visual, rw, imu_j, prior_j, visual_j, rw_j;
};
struct Result {
  float state[30]{};
  float initial_cost = 0, final_cost = 0;
  int iterations = 0;
  double buffer_ms = 0, graph_ms = 0, lm_ms = 0, copy_ms = 0;
};
Evaluation EvaluateGpu(const Data&, const std::vector<Observation>&, const float state[30]);
Result OptimizeEliminated(const Data&, const std::vector<Observation>&, float, int, bool);
Result OptimizeGpu(const Data&, const std::vector<Observation>&, float, int, bool);
}  // namespace gpu_bias_rw
