#pragma once
#include <memory>
#include "window_types.h"

namespace gpu_window::fp64 {
// Internal FP64 path. Float conversion occurs only when returning the legacy
// backend Result; residuals, Jacobians, normal equations and LM states are double.
struct Evaluation {
  std::vector<double> prior, prior_j, edge, edge_j, visual, visual_j;
};
struct Step {
  std::vector<double> dx;
  bool valid = false;
  double predicted = 0, relative_residual = 0;
};
class Backend {
public:
  virtual ~Backend() = default;
  virtual double Evaluate(const std::vector<double>& x, bool jacobian) = 0;
  virtual Step Solve(double lambda) = 0;
};
std::unique_ptr<Backend> MakeCpu(const Problem&);
std::unique_ptr<Backend> MakeGpu(const Problem&);
Evaluation EvaluateHost(const Problem&, const std::vector<double>&, bool);
struct Report {
  std::vector<double> x;
  double before = 0, after = 0, max_linear_residual = 0;
  int iterations = 0, rejected = 0, failed = 0;
};
Report Minimize(const Problem&, Backend&);
Result Optimize(const Problem&, bool gpu);
void SaveProblem(const Problem&, const char* path);
Problem LoadProblem(const char* path);
}  // namespace gpu_window::fp64
