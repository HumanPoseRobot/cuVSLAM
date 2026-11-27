#pragma once

#include <set>
#include <stdexcept>
#include "joint_posterior.h"

namespace gpu_window::posterior::history {
inline std::set<int64_t> ObservedPointIds(const Graph& graph) {
  std::set<int64_t> ids;
  for (const auto& visual : graph.visuals) {
    if (visual.point < 0 || visual.point >= int(graph.ids.size())) {
      throw std::logic_error("invalid raw observation index for historical query");
    }
    ids.insert(graph.ids[visual.point]);
  }
  return ids;
}
}  // namespace gpu_window::posterior::history
