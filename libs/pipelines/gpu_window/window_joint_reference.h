#pragma once
#include <map>
#include <stdexcept>
#include "camera/rig.h"
#include "common/imu_calibration.h"
#include "map/map.h"
#include "pnp/joint_sqrt_prior.h"
#include "window_fp64.h"
#include "window_math.h"

namespace gpu_window::joint_reference {
using cuvslam::pnp::joint_geometry::Factor;
struct Prior {
  Factor factor;
  std::vector<int64_t> times;
  std::vector<State> states;
  std::vector<int64_t> ids;
  std::vector<Eigen::Vector3d> points;
  int64_t absorbed_through_ns = -1;
  int64_t visual_absorbed_through_ns = -1;
  bool empty() const { return factor.rows.cols() == 0; }
};

inline Factor Reanchor(const Factor& raw, const std::vector<State>& states) {
  Factor factor = raw;
  if (raw.rows.cols() < 15 * static_cast<int>(states.size())) {
    throw std::invalid_argument("joint reanchor shape");
  }
  for (size_t node = 0; node < states.size(); ++node) {
    Eigen::Matrix<double, 15, 15> derivative;
    Eigen::Matrix<double, 15, 1> origin;
    for (int col = 0; col < 15; ++col) {
      gpu_vi::Jet z[15], r[15];
      for (int k = 0; k < 15; ++k) {
        z[k] = {0., k == col ? 1. : 0.};
      }
      gpu_window::difference(states[node], states[node], z, r);
      for (int row = 0; row < 15; ++row) {
        derivative(row, col) = r[row].d;
        origin[row] = r[row].v;
      }
    }
    Eigen::FullPivLU<Eigen::Matrix<double, 15, 15>> inverse(derivative);
    if (!inverse.isInvertible()) {
      throw std::runtime_error("singular joint reanchor chart");
    }
    const Eigen::MatrixXd rows = raw.rows.middleCols(15 * node, 15) * inverse.inverse();
    factor.rows.middleCols(15 * node, 15) = rows;
    factor.residual -= rows * origin;
  }
  return factor;
}
bool SeedHistoricalGeometry(const cuvslam::map::UnifiedMap&, const cuvslam::camera::Rig&,
                            const cuvslam::imu::ImuCalibration&, int64_t, const State&,
                            const std::map<cuvslam::TrackId, cuvslam::Vector3T>&, Prior&);

struct Linearization {
  Factor factor;
  double cost = 0.;
};
inline Linearization linearize(const Problem& p, const fp64::Evaluation& e, const std::vector<double>& x,
                               const Prior& prior, const std::vector<int64_t>& times, const std::vector<int64_t>& ids,
                               bool boundary = false) {
  if (!p.source_rays.empty()) {
    throw std::invalid_argument("joint reference requires Cartesian point coordinates");
  }
  const int navigation = 15 * p.count, dimension = navigation + 3 * p.points.size();
  if (times.size() != size_t(p.count) || ids.size() != p.points.size() ||
      x.size() != size_t(15 * p.count + 3 * p.points.size())) {
    throw std::invalid_argument("joint reference labels or state shape");
  }
  std::vector<Factor> blocks;
  Linearization result;
  if (!prior.empty()) {
    const int old_navigation = 15 * prior.times.size(), old_dimension = old_navigation + 3 * prior.ids.size();
    if (!prior.factor.valid() || prior.factor.rows.cols() != old_dimension ||
        prior.states.size() != prior.times.size() || prior.ids.size() != prior.points.size()) {
      throw std::invalid_argument("joint retained prior shape");
    }
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(old_dimension);
    Eigen::MatrixXd tangent = Eigen::MatrixXd::Zero(old_dimension, dimension);
    for (size_t i = 0; i < prior.times.size(); ++i) {
      const auto found = std::find(times.begin(), times.end(), prior.times[i]);
      if (found == times.end()) {
        throw std::invalid_argument("retained navigation state disappeared before elimination");
      }
      const int node = found - times.begin();
      for (int col = 0; col < 15; ++col) {
        gpu_vi::Jet z[15], r[15];
        for (int k = 0; k < 15; ++k) {
          z[k] = {x[15 * node + k], k == col ? 1. : 0.};
        }
        gpu_window::difference(p.nodes[node], prior.states[i], z, r);
        for (int row = 0; row < 15; ++row) {
          delta[15 * i + row] = r[row].v;
          tangent(15 * i + row, 15 * node + col) = r[row].d;
        }
      }
    }
    for (size_t i = 0; i < prior.ids.size(); ++i) {
      const auto found = std::find(ids.begin(), ids.end(), prior.ids[i]);
      if (found == ids.end()) {
        throw std::invalid_argument("correlated landmark disappeared before elimination");
      }
      const int point = found - ids.begin();
      for (int k = 0; k < 3; ++k) {
        delta[old_navigation + 3 * i + k] = p.points[point][k] + x[navigation + 3 * point + k] - prior.points[i][k];
        tangent(old_navigation + 3 * i + k, navigation + 3 * point + k) = 1.;
      }
    }
    Factor block;
    // The chart derivative is block diagonal up to label permutations.
    // Keep this exact sparsity; a dense NxN product needlessly multiplies
    // billions of structural zeros when landmarks retain correlations.
    block.rows = Eigen::MatrixXd::Zero(prior.factor.rows.rows(), dimension);
    for (size_t i = 0; i < prior.times.size(); ++i) {
      const int node = std::find(times.begin(), times.end(), prior.times[i]) - times.begin();
      block.rows.middleCols(15 * node, 15).noalias() =
          prior.factor.rows.middleCols(15 * i, 15) * tangent.block(15 * i, 15 * node, 15, 15);
    }
    for (size_t i = 0; i < prior.ids.size(); ++i) {
      const int point = std::find(ids.begin(), ids.end(), prior.ids[i]) - ids.begin();
      block.rows.middleCols(navigation + 3 * point, 3) = prior.factor.rows.middleCols(old_navigation + 3 * i, 3);
    }
    block.residual = prior.factor.rows * delta + prior.factor.residual;
    result.cost += .5 * block.residual.squaredNorm();
    blocks.push_back(std::move(block));
  }
  const int prior_rows = e.prior.size();
  if (e.prior_j.size() != size_t(prior_rows) * prior_rows) {
    throw std::invalid_argument("joint navigation seed shape");
  }
  if (prior_rows) {
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(
        e.prior_j.data(), prior_rows, prior_rows);
    Factor block;
    block.rows = Eigen::MatrixXd::Zero(prior_rows, dimension);
    const int seeded = std::min(navigation, prior_rows);
    block.rows.leftCols(seeded) = J.leftCols(seeded);
    block.residual = Eigen::Map<const Eigen::VectorXd>(e.prior.data(), prior_rows);
    result.cost += .5 * block.residual.squaredNorm();
    blocks.push_back(std::move(block));
  }
  for (size_t i = 0; i < p.edges.size(); ++i) {
    const auto& edge = p.edges[i];
    if (boundary && edge.left != 0 && edge.right != 0) {
      continue;
    }
    if (!prior.empty() && times[edge.left] <= prior.absorbed_through_ns) {
      throw std::invalid_argument("IMU evidence already absorbed");
    }
    const Eigen::Map<const Eigen::Matrix<double, 15, 30, Eigen::RowMajor>> J(e.edge_j.data() + 450 * i);
    Factor block;
    block.rows = Eigen::MatrixXd::Zero(15, dimension);
    block.rows.block(0, 15 * edge.left, 15, 15) = J.leftCols(15);
    block.rows.block(0, 15 * edge.right, 15, 15) = J.rightCols(15);
    block.residual = Eigen::Map<const Eigen::Matrix<double, 15, 1>>(e.edge.data() + 15 * i);
    result.cost += .5 * block.residual.squaredNorm();
    blocks.push_back(std::move(block));
  }
  for (size_t i = 0; i < p.visuals.size(); ++i) {
    const auto& visual = p.visuals[i];
    if (boundary && visual.node != 0) {
      continue;
    }
    if (!prior.empty() && times[visual.node] <= std::max(prior.absorbed_through_ns, prior.visual_absorbed_through_ns)) {
      throw std::invalid_argument("visual evidence already absorbed");
    }
    const Eigen::Map<const Eigen::Matrix<double, 2, 18, Eigen::RowMajor>> J(e.visual_j.data() + 36 * i);
    const Eigen::Map<const Eigen::Vector2d> residual(e.visual.data() + 2 * i);
    const double norm = residual.norm();
    const double weight = norm > p.huber ? std::sqrt(p.huber / norm) : 1.;
    Factor block;
    block.rows = Eigen::MatrixXd::Zero(2, dimension);
    block.rows.block(0, 15 * visual.node, 2, 15) = weight * J.leftCols(15);
    block.rows.block(0, navigation + 3 * visual.point, 2, 3) = weight * J.rightCols(3);
    block.residual = weight * residual;
    result.cost += norm > p.huber ? p.huber * norm - .5 * p.huber * p.huber : .5 * norm * norm;
    blocks.push_back(std::move(block));
  }
  result.factor = cuvslam::pnp::joint_geometry::combine(blocks, dimension);
  return result;
}
inline Result optimize(const Problem& p, const Prior& prior, const std::vector<int64_t>& times,
                       const std::vector<int64_t>& ids) {
  const int navigation = 15 * p.count, dimension = navigation + 3 * p.points.size();
  std::vector<double> x(15 * p.count + 3 * p.points.size(), 0.);
  double lambda = 1e-3;
  auto evaluation = fp64::EvaluateHost(p, x, true);
  auto current = linearize(p, evaluation, x, prior, times, ids);
  const double before = current.cost;
  Result result;
  result.points.resize(3 * p.points.size());
  result.before = before;
  if (!std::isfinite(before)) {
    throw std::runtime_error("invalid full joint objective");
  }
  for (int iter = 0; iter < p.iterations && current.cost > 1e-10; ++iter) {
    ++result.iterations;
    const auto& A = current.factor.rows;
    const auto& residual = current.factor.residual;
    const Eigen::MatrixXd H = cuvslam::pnp::joint_geometry::dense::gram(A);
    const Eigen::VectorXd gradient = A.transpose() * residual;
    const Eigen::VectorXd diagonal = H.diagonal().cwiseMax(1e-12);
    if (!H.allFinite() || !gradient.allFinite()) {
      throw std::runtime_error("joint normal equations overflow");
    }
    const Eigen::VectorXd scale = diagonal.cwiseSqrt().cwiseInverse();
    Eigen::MatrixXd damped = scale.asDiagonal() * H * scale.asDiagonal();
    damped.diagonal().array() += lambda;
    const Eigen::VectorXd rhs = -(scale.asDiagonal() * gradient);
    Eigen::VectorXd normalized;
    if (!cuvslam::pnp::joint_geometry::dense::solve_spd(damped, rhs, normalized)) {
      lambda = std::min(1e12, lambda * 10.);
      continue;
    }
    const Eigen::VectorXd step = scale.asDiagonal() * normalized;
    const double relative = (damped * normalized - rhs).norm() / std::max(1e-30, rhs.norm());
    if (!step.allFinite() || !std::isfinite(relative) || relative > 1e-8) {
      lambda = std::min(1e12, lambda * 10.);
      continue;
    }
    const double prediction = -gradient.dot(step) - .5 * step.dot(H * step);
    if (!(prediction > 0)) {
      lambda = std::min(1e12, lambda * 10.);
      continue;
    }
    auto trial = x;
    for (int k = 0; k < navigation; ++k) {
      trial[k] += step[k];
    }
    for (size_t k = 0; k < 3 * p.points.size(); ++k) {
      trial[navigation + k] += step[navigation + k];
    }
    auto next_evaluation = fp64::EvaluateHost(p, trial, true);
    auto next = linearize(p, next_evaluation, trial, prior, times, ids);
    const double gain = (current.cost - next.cost) / prediction;
    if (std::isfinite(next.cost) && next.cost <= current.cost && gain >= .25) {
      x = std::move(trial);
      evaluation = std::move(next_evaluation);
      current = std::move(next);
      if (gain > .75) {
        lambda = std::max(1e-8, lambda * .3);
      }
      if (step.squaredNorm() < 1e-10 || prediction / std::max(1., current.cost) < 1e-6) {
        break;
      }
    } else {
      lambda = std::min(1e12, lambda * 10.);
    }
  }
  for (int k = 0; k < navigation; ++k) {
    result.x[k] = x[k];
  }
  for (size_t k = 0; k < result.points.size(); ++k) {
    result.points[k] = x[navigation + k];
  }
  result.after = current.cost;
  return result;
}
inline std::vector<double> unpack(const Problem& p, const Result& result) {
  const int navigation = 15 * p.count;
  std::vector<double> x(navigation + 3 * p.points.size(), 0.);
  for (int k = 0; k < navigation; ++k) {
    x[k] = result.x[k];
  }
  for (size_t k = 0; k < result.points.size(); ++k) {
    x[navigation + k] = result.points[k];
  }
  return x;
}
}  // namespace gpu_window::joint_reference
