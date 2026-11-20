
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
#include <Eigen/Cholesky>
#include <cmath>
#include "common/isometry.h"
namespace cuvslam::sof {
// Validate ONLY a newly searched secondary-camera observation before it can
// enter multi-camera triangulation. Use the measurement covariance and the same
// 9x information / 9.21 residual threshold as visual publication, with no depth,
// motion, scene, camera-ID, or time-window limits.
inline bool ConsistentCalibratedStereo(const Isometry3T& secondary_from_primary, const Vector2T& primary,
                                       const Matrix2T& primary_info, const Vector2T& secondary,
                                       const Matrix2T& secondary_info) {
  if (!primary.allFinite() || !secondary.allFinite() || !primary_info.allFinite() || !secondary_info.allFinite() ||
      !secondary_from_primary.matrix().allFinite())
    return false;
  const Eigen::Vector3d t = secondary_from_primary.translation().cast<double>();
  Eigen::Matrix3d tx;
  tx << 0., -t.z(), t.y(), t.z(), 0., -t.x(), -t.y(), t.x(), 0.;
  const Eigen::Matrix3d E = tx * secondary_from_primary.linear().cast<double>();
  const Eigen::Vector3d a = primary.cast<double>().homogeneous(), b = secondary.cast<double>().homogeneous();
  const Eigen::Vector3d eb = E.transpose() * b, ea = E * a;
  const Eigen::Vector2d jp = eb.head<2>(), js = ea.head<2>();
  Eigen::LDLT<Eigen::Matrix2d> lp(primary_info.cast<double>()), ls(secondary_info.cast<double>());
  if (lp.info() != Eigen::Success || ls.info() != Eigen::Success || !lp.isPositive() || !ls.isPositive()) return false;
  const double variance = jp.dot(lp.solve(jp)) + js.dot(ls.solve(js));
  const double error = b.dot(ea);
  return std::isfinite(variance) && variance > 0. && std::isfinite(error) && 9. * error * error <= 9.21 * variance;
}
}  // namespace cuvslam::sof
