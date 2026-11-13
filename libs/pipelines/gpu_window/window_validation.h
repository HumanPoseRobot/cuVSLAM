#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include "window_types.h"

namespace gpu_window {
// Never saturate a solver result: rejecting the candidate preserves the last
// valid state, whereas clipping silently changes the estimated motion.
inline bool float_representable(double v) {
  return std::isfinite(v) && std::abs(v) <= std::numeric_limits<float>::max();
}
template <class Range>
bool finite_range(const Range& values) {
  for (auto v : values) {
    if (!std::isfinite(v)) {
      return false;
    }
  }
  return true;
}
inline bool valid_rotation(const double* r) {
  for (int i = 0; i < 9; ++i) {
    if (!std::isfinite(r[i])) {
      return false;
    }
  }
  double error = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double dot = 0;
      for (int k = 0; k < 3; ++k) {
        dot += r[3 * k + i] * r[3 * k + j];
      }
      error = std::max(error, std::abs(dot - (i == j ? 1. : 0.)));
    }
  }
  const double det =
      r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
  // Inputs originate in float pose storage. Accept rounding, not reflections,
  // scaling or arbitrary matrices. Do not silently repair calibration here.
  return error <= 1e-4 && std::abs(det - 1.) <= 1e-4;
}
inline bool valid_state(const State& s) {
  return valid_rotation(s.R) && finite_range(s.p) && finite_range(s.v) && finite_range(s.bg) && finite_range(s.ba);
}
inline const char* invalid_problem(const Problem& p) {
  if (p.count < 1 || p.count > Slots || p.iterations < 0 || p.iterations > 1000 || !std::isfinite(p.huber) ||
      p.huber <= 0) {
    return "shape_or_settings";
  }
  // Kernels use signed int offsets, including the dense normal matrix index.
  const size_t n = 15 * p.count + 3 * p.points.size();
  if (p.points.size() > 1000000 || n > 46340 || p.visuals.size() > size_t(std::numeric_limits<int>::max() / 36) ||
      p.edges.size() > size_t(std::numeric_limits<int>::max() / 450)) {
    return "index_overflow";
  }
  if (!finite_range(p.W) || !finite_range(p.offset)) {
    return "prior_nonfinite";
  }
  for (int i = 0; i < Slots; ++i) {
    if (!valid_state(p.nodes[i]) || !valid_state(p.anchors[i])) {
      return "state_or_rotation";
    }
  }
  for (const auto& point : p.points) {
    if (!finite_range(point)) {
      return "point_nonfinite";
    }
  }
  for (const auto& e : p.edges) {
    if (e.left < 0 || e.right < 0 || e.left >= p.count || e.right >= p.count || e.left == e.right) {
      return "edge_index";
    }
    const auto& d = e.data;
    if (!std::isfinite(d.dt) || d.dt <= 0 || !valid_rotation(d.R[0]) || !valid_rotation(d.R[1]) ||
        !valid_rotation(d.dR)) {
      return "imu_rotation_or_time";
    }
    if (!finite_range(d.p[0]) || !finite_range(d.p[1]) || !finite_range(d.v[0]) || !finite_range(d.v[1]) ||
        !finite_range(d.bg) || !finite_range(d.ba) || !finite_range(d.bg0) || !finite_range(d.ba0) ||
        !finite_range(d.dV) || !finite_range(d.dP) || !finite_range(d.JRg) || !finite_range(d.JVg) ||
        !finite_range(d.JVa) || !finite_range(d.JPg) || !finite_range(d.JPa) || !finite_range(d.gravity) ||
        !finite_range(d.imu_sqrt) || !finite_range(e.rw)) {
      return "imu_nonfinite";
    }
  }
  for (const auto& v : p.visuals) {
    if (v.node < -1 || v.node >= p.count || v.point < 0 || size_t(v.point) >= p.points.size()) {
      return "visual_index";
    }
    if (v.node < 0 && !valid_state(v.fixed)) {
      return "fixed_pose";
    }
    if (!valid_rotation(v.obs.Rci) || !finite_range(v.obs.tci) || !finite_range(v.obs.xy) ||
        !finite_range(v.obs.sqrt_info) || !std::isfinite(v.obs.min_depth) || v.obs.min_depth <= 0) {
      return "visual_input";
    }
  }
  return nullptr;
}
inline bool valid_evaluation(const Evaluation& e) {
  return finite_range(e.edge) && finite_range(e.edge_j) && finite_range(e.visual) && finite_range(e.visual_j) &&
         finite_range(e.prior) && finite_range(e.prior_j);
}
inline bool valid_result(const Result& r, size_t points) {
  return r.points.size() == 3 * points && finite_range(r.x) && finite_range(r.points) && std::isfinite(r.before) &&
         std::isfinite(r.after);
}
}  // namespace gpu_window
