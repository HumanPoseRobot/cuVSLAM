#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
#include "joint_lapack.h"
#include "pipelines/gpu_window/window_linear.h"

namespace cuvslam::pnp::joint_geometry {
// A square-root factor over explicitly ordered pose AND landmark coordinates.
// No block-diagonal or independent-map approximation is applied at a boundary.
struct Factor {
  Eigen::MatrixXd rows;
  Eigen::VectorXd residual;
  std::vector<uint64_t> evidence;
  bool valid() const { return rows.rows() == residual.rows() && rows.allFinite() && residual.allFinite(); }
};
inline Factor combine(const std::vector<Factor>& factors, int dimension) {
  Factor result;
  int count = 0;
  std::set<uint64_t> seen;
  for (const auto& factor : factors) {
    if (!factor.valid() || factor.rows.cols() != dimension) {
      throw std::invalid_argument("joint factor shape or finite check");
    }
    for (auto id : factor.evidence) {
      if (!seen.insert(id).second) {
        throw std::invalid_argument("joint evidence counted twice");
      }
    }
    count += factor.rows.rows();
  }
  result.rows.resize(count, dimension);
  result.residual.resize(count);
  int row = 0;
  for (const auto& factor : factors) {
    result.rows.middleRows(row, factor.rows.rows()) = factor.rows;
    result.residual.segment(row, factor.residual.size()) = factor.residual;
    row += factor.rows.rows();
  }
  result.evidence.assign(seen.begin(), seen.end());
  return result;
}
inline Factor eliminate(const Factor& factor, const std::vector<int>& kept) {
  if (!factor.valid()) {
    throw std::invalid_argument("invalid joint factor");
  }
  std::vector<bool> selected(factor.rows.cols(), false);
  for (int i : kept) {
    if (i < 0 || i >= factor.rows.cols() || selected[i]) {
      throw std::invalid_argument("invalid joint retained columns");
    }
    selected[i] = true;
  }
  Eigen::MatrixXd retain(factor.rows.rows(), kept.size());
  Eigen::MatrixXd nuisance(factor.rows.rows(), factor.rows.cols() - kept.size());
  for (size_t i = 0; i < kept.size(); ++i) {
    retain.col(i) = factor.rows.col(kept[i]);
  }
  int at = 0;
  for (int i = 0; i < factor.rows.cols(); ++i) {
    if (!selected[i]) {
      nuisance.col(at++) = factor.rows.col(i);
    }
  }
  Factor result;
  dense::project(nuisance, retain, factor.residual, result.rows, result.residual);
  result.evidence = factor.evidence;
  return result;
}
inline Factor compress(const Factor& factor) {
  if (!factor.valid()) {
    throw std::invalid_argument("invalid joint compression");
  }
  Factor result;
  dense::compress(factor.rows, factor.residual, result.rows, result.residual);
  result.evidence = factor.evidence;
  return result;
}
struct Marginal {
  bool observable = false;
  Eigen::MatrixXd covariance, information;
  Eigen::VectorXd gradient;
  int rank = 0;
};
inline Marginal marginal(const Factor& factor, const std::vector<int>& kept,
                         const Eigen::VectorXd* original_column_norms = nullptr) {
  const auto projected = eliminate(factor, kept);
  Marginal result;
  result.information = projected.rows.transpose() * projected.rows;
  result.gradient = projected.rows.transpose() * projected.residual;
  const int n = kept.size();
  if (!n || projected.rows.rows() < n) {
    return result;
  }
  // The absolute reference is the unprojected kept Jacobian, not the tiny
  // remainder after eliminating an unanchored point. Normalizing numerical
  // null-space noise must never turn an unseen map point into pose evidence.
  if (original_column_norms && original_column_norms->size() != n) {
    throw std::invalid_argument("joint reference scale shape");
  }
  Eigen::VectorXd scale(n);
  for (int i = 0; i < n; ++i) {
    scale[i] = original_column_norms ? (*original_column_norms)[i] : factor.rows.col(kept[i]).norm();
    if (!(scale[i] > 0)) {
      return result;
    }
  }
  const Eigen::MatrixXd normalized = projected.rows * scale.cwiseInverse().asDiagonal();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(normalized, Eigen::ComputeThinV);
  if (svd.info() != Eigen::Success) {
    return result;
  }
  const auto singular = svd.singularValues();
  const double tolerance = 256. * std::numeric_limits<double>::epsilon() *
                           std::max(factor.rows.rows(), factor.rows.cols()) * std::max(1., singular[0]);
  result.rank = (singular.array() > tolerance).count();
  if (result.rank != n) {
    return result;  // No zero covariance is returned for an unobservable state.
  }
  const auto inverse = singular.array().square().inverse().matrix().eval();
  result.covariance = scale.cwiseInverse().asDiagonal() * svd.matrixV() * inverse.asDiagonal() *
                      svd.matrixV().transpose() * scale.cwiseInverse().asDiagonal();
  result.observable = result.covariance.allFinite();
  return result;
}
}  // namespace cuvslam::pnp::joint_geometry
