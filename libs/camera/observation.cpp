
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

#include "camera/observation.h"

#include <Eigen/Cholesky>

namespace {

bool ObservationsCompareById(const cuvslam::camera::Observation& lhs, const cuvslam::camera::Observation& rhs) {
  return lhs.id < rhs.id;
}

}  // namespace

namespace cuvslam::camera {

bool FindObservation(const std::vector<Observation>& observations, TrackId track_id, size_t* index) {
  Observation x;
  x.id = track_id;
  const auto res_iter = std::lower_bound(std::begin(observations), std::end(observations), x, ObservationsCompareById);
  if (res_iter == std::end(observations) || res_iter->id != track_id) {
    return false;
  }
  if (index != NULL) {
    *index = res_iter - std::begin(observations);
  }
  return true;
}

// Creates a physically meaningful information matrix for 2D observations.
Matrix2T GetDefaultObservationInfoUV() {
  // conservative estimate for the tracking noise on undistorted frame
  constexpr auto stdPx = float{3};

  // information matrix in pixel space
  Matrix2T invSigma;
  invSigma << 1.f / (stdPx * stdPx), 0.f, 0.f, 1.f / (stdPx * stdPx);

  return invSigma;
}

Matrix2T ObservationInfoUVToNormUV(const ICameraModel& intrinsics, const Matrix2T& info_uv) {
  const Vector2T& focal = intrinsics.getFocal();
  const Matrix2T invJacobian = focal.asDiagonal();

  // if y = f(x), j^-1 = f'(x), then
  // cov(y) ~ j^-1 cov(x) j^-t
  // info(y) ~ j^t info(x) j

  // invJacobian - diagonal matrix => invJacobian^T = invJacobian
  return invJacobian * info_uv * invJacobian;
}

bool TryObservationInfoUVToXY(const ICameraModel& intrinsics, const Vector2T& uv,
                              const Matrix2T& info_uv, Vector2T& xy, Matrix2T& info_xy) {
  if (!intrinsics.normalizePoint(uv, xy) || !info_uv.allFinite()) {
    return false;
  }
  const Eigen::Matrix2d w = .5 * (info_uv.cast<double>() + info_uv.transpose().cast<double>());
  const Eigen::LLT<Eigen::Matrix2d> pixel_factor(w);
  if (pixel_factor.info() != Eigen::Success) {
    return false;
  }
  const Eigen::Matrix2d axes = pixel_factor.matrixU().solve(Eigen::Matrix2d::Identity());
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (int axis = 0; axis < 2; ++axis) {
    // Use a centered approximation of the local derivative. When the pixel is
    // near the model boundary, use a shorter valid stencil along this axis.
    double scale = 1.;
    bool usable = false;
    for (int i = 0; i < 5; ++i, scale *= .5) {
      const Vector2T delta = (scale * axes.col(axis)).cast<float>();
      Vector2T plus, minus;
      if (intrinsics.normalizePoint(uv + delta, plus) && intrinsics.normalizePoint(uv - delta, minus)) {
        const Eigen::Vector2d j = (plus.cast<double>() - minus.cast<double>()) / (2 * scale);
        cov.noalias() += j * j.transpose();
        usable = true;
        break;
      }
    }
    if (!usable) {
      return false;
    }
  }
  const Eigen::LLT<Eigen::Matrix2d> normalized_factor(cov);
  if (!cov.allFinite() || normalized_factor.info() != Eigen::Success) {
    return false;
  }
  info_xy = normalized_factor.solve(Eigen::Matrix2d::Identity()).cast<float>();
  return info_xy.allFinite() && info_xy(0, 0) > 0 &&
         info_xy.cast<double>().determinant() > 0;
}

}  // namespace cuvslam::camera
