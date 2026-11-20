
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

#pragma once

#include <functional>
#include <vector>
#include "camera/observation.h"
#include "camera/rig.h"
#include "pipelines/gpu_window/frontend_guard_adapter.h"

namespace cuvslam::pipelines {
// Triangulation may correct arbitrarily inconsistent epipolar observations.
// Verify the ORIGINAL measurements using the same visual residual policy as
// tracking; no depth cap, scene-specific parallax threshold or relaxed gate.
inline bool TriangulationObservationsConsistent(
    const camera::Rig& rig, const Isometry3T& world_from_rig, const Vector3T& point,
    const std::vector<std::reference_wrapper<const camera::Observation>>& observations) {
  if (!point.allFinite() || !world_from_rig.matrix().allFinite()) {
    return false;
  }
  const auto rig_from_world = world_from_rig.inverse();
  for (const auto& ref : observations) {
    const auto& o = ref.get();
    if (o.cam_id >= rig.num_cameras) {
      return false;
    }
    const Vector3T pc = rig.camera_from_rig[o.cam_id] * rig_from_world * point;
    Vector2T pixel;
    if (!pc.allFinite() || pc.z() <= .05f ||
        !frontend_guard::actual_pixel_residual(*rig.intrinsics[o.cam_id], pc.hnormalized(), o.xy, pixel)) {
      return false;
    }
    const Vector2T e = pc.hnormalized() - o.xy;
    if (!frontend_guard::observation_consistent(9. * o.xy_info.cast<double>(), e.cast<double>(),
                                                pixel.cast<double>())) {
      return false;
    }
  }
  return observations.size() >= 2;
}
}  // namespace cuvslam::pipelines
