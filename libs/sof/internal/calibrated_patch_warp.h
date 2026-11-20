
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
#include <cmath>
#include "camera/camera.h"
#include "common/isometry.h"
#include "common/vector_3t.h"
namespace cuvslam::sof {
// Jacobian from primary raster offsets to secondary raster offsets under the
// calibrated camera rotation. Used only to sample the small matching patch:
// original images, field of view, timestamps and calibration remain unchanged.
inline bool CalibratedPatchWarp(const camera::ICameraModel& primary, const camera::ICameraModel& secondary,
                                const Matrix3T& secondary_from_primary_rotation, const Vector2T& primary_uv,
                                Matrix2T& map) {
  auto project = [&](const Vector2T& uv, Vector2T& result) {
    Vector2T xy;
    if (!primary.normalizePoint(uv, xy)) {
      return false;
    }
    const Vector3T ray = secondary_from_primary_rotation * xy.homogeneous();
    return ray.allFinite() && ray.z() > 1e-8f && secondary.denormalizePoint(ray.hnormalized(), result) &&
           result.allFinite();
  };
  Matrix2T candidate;
  for (int axis = 0; axis < 2; ++axis) {
    Vector2T offset = Vector2T::Zero();
    offset[axis] = .5f;
    Vector2T a, b;
    if (!project(primary_uv + offset, a) || !project(primary_uv - offset, b)) {
      return false;
    }
    candidate.col(axis) = a - b;
  }
  if (!candidate.allFinite() || std::abs(candidate.determinant()) <= 1e-8f) {
    return false;
  }
  map = candidate;
  return true;
}
// Translation-only LK linearizes corresponding image gradients without a patch
// rotation. A non-positive symmetric part can reverse a descent direction even
// for an ideal image. This is a camera-model dispatch check, not a VIO gate or
// a threshold on scene depth, speed, time or an observed trajectory error.
inline bool TranslationPatchCompatible(const Matrix2T& map) {
  const float off = .5f * (map(0, 1) + map(1, 0));
  return map.allFinite() && map(0, 0) > 0.f && map(1, 1) > 0.f && map(0, 0) * map(1, 1) - off * off > 0.f;
}
inline bool PatchWarpRequired(const Matrix2T& map) {
  // Compare omission error over the actual 9x9 patch with the existing ST
  // subpixel convergence accuracy (0.05 px), not a scene or VIO speed limit.
  const Matrix2T delta = map - Matrix2T::Identity();
  for (float x : {-4.f, 4.f}) {
    for (float y : {-4.f, 4.f}) {
      if ((delta * Vector2T(x, y)).norm() > .05f) {
        return true;
      }
    }
  }
  return false;
}
}  // namespace cuvslam::sof
