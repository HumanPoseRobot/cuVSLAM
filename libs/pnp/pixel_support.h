#pragma once
#include "camera/observation.h"
#include "camera/rig.h"
#include "common/vector_3t.h"
namespace cuvslam::pnp {
// Normalized-coordinate information and actual lens pixel reprojection are
// separate checks. Multiplication by focal length is exact only for pinhole.
inline bool PixelObservationSupported(const camera::Rig& rig, const camera::Observation& observation,
                                      const Vector3T& pc, float min_depth = .001f) {
  if (observation.cam_id >= rig.num_cameras || !pc.allFinite() || pc.z() <= min_depth) {
    return false;
  }
  const auto* camera = rig.intrinsics[observation.cam_id];
  if (!camera || !observation.xy.allFinite() || !observation.xy_info.allFinite()) {
    return false;
  }
  Vector2T predicted, measured;
  if (!camera->denormalizePoint(pc.hnormalized(), predicted) || !camera->denormalizePoint(observation.xy, measured)) {
    return false;
  }
  const Vector2T residual = pc.hnormalized() - observation.xy;
  const double chi = 9. * residual.cast<double>().dot(observation.xy_info.cast<double>() * residual.cast<double>());
  return std::isfinite(chi) && chi <= 9.21 && (predicted - measured).squaredNorm() <= 64.f;
}
}  // namespace cuvslam::pnp
