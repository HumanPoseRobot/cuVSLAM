#pragma once
#include <array>
#include "gpu_joint_types.h"
namespace gpu_window {
constexpr int Slots = 6, Dim = Slots * 15;  // five retained keyframes plus current image
struct State {
  double R[9]{1, 0, 0, 0, 1, 0, 0, 0, 1}, p[3]{}, v[3]{}, bg[3]{}, ba[3]{};
};
struct Edge {
  int left = 0, right = 0;
  gpu_vi::Data data{};
  double rw[36]{};
};
struct Visual {
  int node = 0, point = 0;
  gpu_vi::Observation obs{};
};
// A fixed source-camera coordinate chart for one optimization call. Bearing
// offsets remain variables: noisy source pixels are not made exact constraints.
struct SourceRay {
  double Rwc[9]{1, 0, 0, 0, 1, 0, 0, 0, 1}, ray[2]{}, source_xy[2]{}, center[3]{};
  double inverse_depth = 0;
  int node = -1;
};
struct Problem {
  std::array<State, Slots> nodes{}, anchors{};
  std::vector<std::array<double, 3>> points;
  std::vector<Edge> edges;
  std::vector<Visual> visuals;
  std::array<double, Dim * Dim> W{};
  std::array<double, Dim> offset{};
  std::vector<SourceRay> source_rays;
  int count = 0, iterations = 7;
  double huber = 1;
};
struct Evaluation {
  std::vector<float> edge, edge_j, visual, visual_j, prior, prior_j;
};
struct Result {
  std::array<float, Dim> x{};
  std::vector<float> points;
  float before = 0, after = 0;
  int iterations = 0;
  double landmark_ms = 0;
};
Result Optimize(const Problem&);
Result OptimizeSchur(const Problem&);
Evaluation Evaluate(const Problem&, const Result&);
Evaluation EvaluateCpu(const Problem&, const Result&);
Result OptimizeCpu(const Problem&);
}  // namespace gpu_window
