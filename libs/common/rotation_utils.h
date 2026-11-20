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

#ifndef ROTATION_UTILS_H
#define ROTATION_UTILS_H

#include <limits>
#include <type_traits>
#include "include_eigen.h"

namespace cuvslam {
namespace common {

/**
 * @brief Calculate rotation matrix from a transformation matrix using SVD decomposition
 *
 * This function uses SVD decomposition to extract the rotation part from a transformation matrix
 *
 * @tparam Scalar The scalar type (float, double, etc.)
 * @tparam Dim The dimension of the space (typically 3 for 3D)
 * @param transform_matrix The transformation matrix to extract rotation from
 * @return The rotation matrix extracted from the transformation
 */
template <typename Scalar, int Dim>
EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, Dim, Dim> CalculateRotationFromSVD(
    const Eigen::Matrix<Scalar, Dim, Dim>& transform_matrix) {
  using WorkScalar = typename std::conditional<std::is_same<Scalar, float>::value, double, Scalar>::type;
  using MatrixType = Eigen::Matrix<WorkScalar, Dim, Dim>;
  if (!transform_matrix.allFinite()) {
    return Eigen::Matrix<Scalar, Dim, Dim>::Constant(std::numeric_limits<Scalar>::quiet_NaN());
  }
  // Float JacobiSVD can return det errors above 1e-6 for ordinary finite
  // scaled/reflected inputs. Project in FP64 and round once at the API boundary.
  const MatrixType input = transform_matrix.template cast<WorkScalar>();
  Eigen::JacobiSVD<MatrixType> svd(input, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // U * V^T is the closest orthogonal matrix, but may have det = -1 (reflection).
  // When that happens, negating U's last column (smallest singular value) yields
  // the closest proper rotation -- equivalent to subtracting 2 * u_last * v_last^T
  // from U * V^T, which avoids a second full matmul.
  MatrixType uvt = svd.matrixU() * svd.matrixV().adjoint();
  if (uvt.determinant() < WorkScalar(0)) {
    uvt -= WorkScalar(2) * svd.matrixU().col(Dim - 1) * svd.matrixV().col(Dim - 1).adjoint();
  }
  return uvt.template cast<Scalar>();
}

/**
 * @brief Calculate rotation matrix from a 4x4 transformation matrix
 *
 * This function extracts the 3x3 rotation part from a 4x4 transformation matrix
 * and applies the SVD-based rotation extraction.
 *
 * @tparam Scalar The scalar type (float, double, etc.)
 * @param transform_matrix The 4x4 transformation matrix
 * @return The 3x3 rotation matrix
 */
template <typename Scalar>
EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 3, 3> CalculateRotationFromSVD(
    const Eigen::Matrix<Scalar, 4, 4>& transform_matrix) {
  // Extract the 3x3 linear part
  Eigen::Matrix<Scalar, 3, 3> linear_part = transform_matrix.template block<3, 3>(0, 0);

  // Apply the SVD-based rotation extraction
  return CalculateRotationFromSVD<Scalar, 3>(linear_part);
}

/**
 * @brief Calculate rotation matrix from a 3x4 transformation matrix (AffineCompact)
 *
 * This function extracts the 3x3 rotation part from a 3x4 transformation matrix
 * and applies the SVD-based rotation extraction.
 *
 * @tparam Scalar The scalar type (float, double, etc.)
 * @param transform_matrix The 3x4 transformation matrix
 * @return The 3x3 rotation matrix
 */
template <typename Scalar>
EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 3, 3> CalculateRotationFromSVD(
    const Eigen::Matrix<Scalar, 3, 4>& transform_matrix) {
  // Extract the 3x3 linear part
  Eigen::Matrix<Scalar, 3, 3> linear_part = transform_matrix.template block<3, 3>(0, 0);

  // Apply the SVD-based rotation extraction
  return CalculateRotationFromSVD<Scalar, 3>(linear_part);
}

}  // namespace common
}  // namespace cuvslam

#endif  // ROTATION_UTILS_H
