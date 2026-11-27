#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>
#include <string_view>
#include <vector>
#include "joint_posterior.h"

namespace gpu_window::posterior::finite_geometry {
// This is a seed/finite-domain test, not a new measurement. Accepted points
// remain free Cartesian variables jointly constrained by the original noisy
// source AND target observations and variable poses. Deferred observations
// remain in Online::raw while their source state is retained. No fixed-pose
// pseudo-prior or finite replacement for an infinite direction is introduced.
enum class Domain { Finite, Boundary, Unobservable, Numerical };
struct Fit {
  Domain domain = Domain::Numerical;
  Eigen::Vector3d coordinates = Eigen::Vector3d::Zero();  // alpha,beta,inverse z
  Eigen::Vector3d world = Eigen::Vector3d::Zero();
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  double cost = 0., condition = 0., roundoff_rho = 0., stationarity_rho = 0.;
  int iterations = 0;
};
struct Observation {
  Eigen::Matrix3d M;
  Eigen::Vector3d baseline;
  Eigen::Vector2d xy;
  Eigen::Matrix2d W;
};
struct Evaluation {
  Eigen::VectorXd residual;
  Eigen::MatrixXd jacobian;
  double cost = 0.;
  bool valid = false;
};
inline bool Enabled() {
  const char* value = std::getenv("VIO_FINITE_GEOMETRY");
  if (!value) { return false; }
  if (std::string_view(value) != "1") { throw std::invalid_argument("VIO_FINITE_GEOMETRY must be 1 or unset"); }
  return true;
}
inline Evaluation Evaluate(const std::vector<Observation>& observations, const Eigen::Vector3d& q, double huber) {
  Evaluation out;
  out.residual.resize(2 * observations.size());
  out.jacobian.resize(2 * observations.size(), 3);
  const Eigen::Vector3d ray(q.x(), q.y(), 1.);
  for (size_t i = 0; i < observations.size(); ++i) {
    const auto& o = observations[i];
    // Signed inverse depth is allowed ONLY in this independent domain test.
    // A negative optimum never becomes an accepted physical landmark.
    const Eigen::Vector3d pc = o.M * ray + q.z() * o.baseline;
    if (!pc.allFinite() || !(pc.z() > 0.)) { return out; }
    const Eigen::Vector2d xy = pc.hnormalized();
    const Eigen::Vector2d residual = o.W * (xy - o.xy);
    Eigen::Matrix<double, 2, 3> projection;
    projection << 1. / pc.z(), 0., -xy.x() / pc.z(), 0., 1. / pc.z(), -xy.y() / pc.z();
    Eigen::Matrix3d derivative;
    derivative.leftCols<2>() = o.M.leftCols<2>();
    derivative.col(2) = o.baseline;
    const double norm = residual.norm();
    const double weight = norm > huber ? std::sqrt(huber / norm) : 1.;
    out.residual.segment<2>(2 * i) = weight * residual;
    out.jacobian.middleRows<2>(2 * i) = weight * o.W * projection * derivative;
    out.cost += norm > huber ? huber * norm - .5 * huber * huber : .5 * residual.squaredNorm();
  }
  out.valid = std::isfinite(out.cost) && out.residual.allFinite() && out.jacobian.allFinite();
  return out;
}
inline Fit Solve(const std::vector<Observation>& observations, Eigen::Vector3d q, double huber) {
  Fit fit;
  if (observations.size() < 2) { fit.domain = Domain::Unobservable; return fit; }
  auto current = Evaluate(observations, q, huber);
  if (!current.valid) { return fit; }
  double lambda = 1e-3;
  bool converged = false;
  const double eps = std::numeric_limits<double>::epsilon();
  for (int iteration = 0; iteration < 80; ++iteration) {
    fit.iterations = iteration + 1;
    const Eigen::Matrix3d H = current.jacobian.transpose() * current.jacobian;
    const Eigen::Vector3d g = current.jacobian.transpose() * current.residual;
    if ((H.diagonal().array() <= 0.).any()) { fit.domain = Domain::Unobservable; return fit; }
    const Eigen::Vector3d scale = H.diagonal().array().sqrt().inverse();
    const Eigen::Vector3d gradient = scale.asDiagonal() * g;
    // Dimensionless numerical convergence, not a depth/parallax/scene cutoff.
    if (gradient.lpNorm<Eigen::Infinity>() <= 64. * std::pow(eps, 2. / 3.) * std::max(1., std::sqrt(2. * current.cost))) {
      converged = true; break;
    }
    const Eigen::MatrixXd normalized_J = current.jacobian * scale.asDiagonal();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(normalized_J);
    qr.setThreshold(256. * eps * std::max(normalized_J.rows(), normalized_J.cols()));
    if (qr.rank() != 3) { fit.domain = Domain::Unobservable; return fit; }
    const Eigen::Vector3d free_step = qr.solve(-current.residual);
    const Eigen::VectorXd projected = normalized_J * free_step;
    // The unconstrained GN model cannot resolve an improvement below the
    // objective's floating-point evaluation error. Do NOT call such a
    // well-optimized noisy point "Numerical" merely because LM gain rounds to
    // zero. Use the UNDAMPED model, not an artificially tiny large-lambda step.
    if (.5 * projected.squaredNorm() <= 128. * eps * std::max(1., current.cost)) {
      converged = true; break;
    }
    Eigen::MatrixXd augmented(normalized_J.rows() + 3, 3);
    augmented.topRows(normalized_J.rows()) = normalized_J;
    augmented.bottomRows<3>() = std::sqrt(lambda) * Eigen::Matrix3d::Identity();
    Eigen::VectorXd rhs(augmented.rows()); rhs.head(current.residual.size()) = -current.residual; rhs.tail<3>().setZero();
    // Solve the augmented least-squares system directly; no squared condition
    // number from the three-variable normal equations.
    const Eigen::Vector3d step = scale.asDiagonal() * augmented.colPivHouseholderQr().solve(rhs);
    const double prediction = -g.dot(step) - .5 * step.dot(H * step);
    if (!step.allFinite() || !(prediction > 0.)) { break; }
    auto next = Evaluate(observations, q + step, huber);
    const double gain = next.valid ? (current.cost - next.cost) / prediction : -1.;
    if (next.valid && next.cost <= current.cost && gain >= .25) {
      q += step;
      const double decrease = current.cost - next.cost;
      current = std::move(next);
      lambda = std::max(1e-12, .3 * lambda);
      if (decrease <= 128. * eps * std::max(1., current.cost)) { converged = true; break; }
    } else {
      lambda = std::min(1e12, 10. * lambda);
    }
  }
  if (!converged) { return fit; }
  const Eigen::Vector3d norms = current.jacobian.colwise().norm();
  if ((norms.array() <= 0.).any()) { fit.domain = Domain::Unobservable; return fit; }
  const Eigen::MatrixXd normalized = current.jacobian * norms.cwiseInverse().asDiagonal();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(normalized, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto singular = svd.singularValues();
  const double tolerance = 256. * eps * std::max(normalized.rows(), normalized.cols()) * std::max(1., singular[0]);
  if (svd.info() != Eigen::Success || singular.size() != 3 || singular[2] <= tolerance) {
    fit.domain = Domain::Unobservable;
    return fit;
  }
  fit.condition = singular[0] / singular[2];
  fit.covariance = norms.cwiseInverse().asDiagonal() * svd.matrixV() *
                   singular.array().square().inverse().matrix().asDiagonal() * svd.matrixV().transpose() *
                   norms.cwiseInverse().asDiagonal();
  // Floating-point sign uncertainty expressed in inverse-depth units via the
  // local metric, so changing metres to millimetres does not add a depth gate.
  fit.roundoff_rho = 256. * eps * std::max(normalized.rows(), normalized.cols()) * fit.condition *
                     std::sqrt(fit.covariance(2, 2)) * std::max(1., std::sqrt(2. * current.cost));
  const Eigen::Vector3d remaining_step = norms.cwiseInverse().asDiagonal() * svd.solve(-current.residual);
  fit.stationarity_rho = std::abs(remaining_step.z());
  fit.coordinates = q;
  fit.cost = current.cost;
  fit.domain = q.z() > fit.roundoff_rho + fit.stationarity_rho ? Domain::Finite : Domain::Boundary;
  return fit;
}
inline Fit Profile(const Graph& graph, int point, const std::vector<const Visual*>& views) {
  Fit result;
  if (views.size() < 2) { result.domain = Domain::Unobservable; return result; }
  const auto& first = *views.front();
  const auto& source = graph.nodes[first.node];
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R0(source.R), C0(first.obs.Rci);
  const Eigen::Map<const Eigen::Vector3d> p0(source.p), t0(first.obs.tci);
  const Eigen::Matrix3d A0 = C0 * R0.transpose();
  const Eigen::Matrix3d inverse = A0.inverse();
  const Eigen::Vector3d center = p0 - inverse * t0;
  std::vector<Observation> data;
  for (const auto* view : views) {
    const auto& node = graph.nodes[view->node];
    const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(node.R), C(view->obs.Rci);
    const Eigen::Map<const Eigen::Vector3d> p(node.p), t(view->obs.tci);
    Observation o;
    if (view->node == first.node) {
      // Same-rig relative geometry is independent of its uncertain world pose.
      o.M = C * C0.inverse();
      o.baseline = t - o.M * t0;
    } else {
      o.M = C * R.transpose() * inverse;
      o.baseline = C * R.transpose() * (center - p) + t;
    }
    o.xy = Eigen::Map<const Eigen::Vector2d>(view->obs.xy);
    o.W = Eigen::Map<const Eigen::Matrix<double, 2, 2, Eigen::RowMajor>>(view->obs.sqrt_info);
    data.push_back(o);
  }
  data.front().M.setIdentity();
  data.front().baseline.setZero();
  const Eigen::Vector3d pc = A0 * (graph.points[point] - p0) + t0;
  Eigen::Vector3d q;
  if (pc.allFinite() && pc.z() > 0.) { q << pc.hnormalized(), 1. / pc.z(); }
  else { q << data.front().xy, 0.; }
  result = Solve(data, q, graph.huber);
  if (result.domain == Domain::Finite) {
    result.world = center + inverse * Eigen::Vector3d(result.coordinates.x(), result.coordinates.y(), 1.) /
                                result.coordinates.z();
    if (!result.world.allFinite()) { result.domain = Domain::Numerical; }
  }
  return result;
}
struct Stats {
  int seed_passes = 0, finite = 0, boundary = 0, unobservable = 0, numerical = 0;
  double geometry_ms = 0.;
  std::vector<std::pair<int64_t, Domain>> deferred;
};
inline std::vector<bool> Inspect(Graph& graph, bool seed, Stats& stats) {
  std::vector<std::vector<const Visual*>> groups(graph.points.size());
  for (const auto& o : graph.visuals) { groups[o.point].push_back(&o); }
  std::set<int64_t> protected_ids(graph.prior.ids.begin(), graph.prior.ids.end());
  std::vector<bool> remove(graph.points.size(), false);
  for (size_t i = 0; i < groups.size(); ++i) {
    // A current pair cannot override an existing correlated historical factor.
    if (protected_ids.count(graph.ids[i])) { continue; }
    const auto fit = Profile(graph, i, groups[i]);
    if (fit.domain == Domain::Finite) {
      ++stats.finite;
      if (seed) { graph.points[i] = fit.world; }
    } else if (fit.domain == Domain::Numerical) {
      // Failure to certify is NOT evidence of invalid geometry. Keep original
      // means and noisy factors; never cascade numerical stagnation into
      // deletion of otherwise valid observations.
      ++stats.numerical;
    } else {
      remove[i] = true;
      stats.deferred.emplace_back(graph.ids[i], fit.domain);
      if (fit.domain == Domain::Boundary) { ++stats.boundary; }
      if (fit.domain == Domain::Unobservable) { ++stats.unobservable; }
    }
  }
  return remove;
}
inline void Compact(Graph& graph, const std::vector<bool>& remove) {
  std::vector<int> indices(graph.points.size(), -1);
  Graph result = graph;  // prior remains shared immutable and bitwise unchanged
  result.ids.clear(); result.points.clear(); result.visuals.clear(); result.visual_cameras.clear();
  for (size_t i = 0; i < graph.points.size(); ++i) {
    if (!remove[i]) {
      indices[i] = result.points.size(); result.ids.push_back(graph.ids[i]); result.points.push_back(graph.points[i]);
    }
  }
  for (size_t i = 0; i < graph.visuals.size(); ++i) {
    auto o = graph.visuals[i];
    if (indices[o.point] >= 0) {
      o.point = indices[o.point]; result.visuals.push_back(o); result.visual_cameras.push_back(graph.visual_cameras[i]);
    }
  }
  graph = std::move(result);
}
inline Report OptimizeFinite(Graph& graph, Linearization* final, Stats& stats) {
  if (!Enabled()) { return Optimize(graph, final); }
  Graph initial = graph;
  auto begin = std::chrono::steady_clock::now();
  const auto remove = Inspect(initial, true, stats);
  Compact(initial, remove);
  stats.geometry_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  for (;;) {
    Graph trial = initial;
    auto report = Optimize(trial, final);
    if (!report.success) { return report; }
    begin = std::chrono::steady_clock::now();
    const auto invalid = Inspect(trial, false, stats);
    stats.geometry_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (std::none_of(invalid.begin(), invalid.end(), [](bool x) { return x; })) {
      graph = std::move(trial);
      return report;
    }
    // Restart from the same valid initial state, never from a rejected NLS
    // result. Every retry removes at least one previously unconsumed point.
    Compact(initial, invalid);
    ++stats.seed_passes;
  }
}
}  // namespace gpu_window::posterior::finite_geometry
