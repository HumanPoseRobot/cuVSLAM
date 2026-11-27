#include "joint_posterior.h"
#include "consistent_prior_chart.h"
#include <chrono>
#include <cstdlib>
#include <optional>
#include "block_lm_metric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace gpu_window::posterior {
namespace {
namespace gaussian = cuvslam::pnp::joint_geometry;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

int StateDimension(const Graph& graph) { return 15 * graph.nodes.size() + 2; }
int Dimension(const Graph& graph) { return 15 * graph.nodes.size() + 3 * graph.points.size() + 2; }

template <class T>
int Index(const std::vector<T>& labels, T label) {
  const auto found = std::find(labels.begin(), labels.end(), label);
  if (found == labels.end()) {
    throw std::invalid_argument("joint posterior variable missing from live graph");
  }
  return static_cast<int>(found - labels.begin());
}

void Validate(const Graph& graph) {
  if (graph.nodes.size() != graph.times.size() || graph.points.size() != graph.ids.size() ||
      graph.visuals.size() != graph.visual_cameras.size() || !graph.gravity.allFinite() ||
      !(graph.gravity.norm() > 0.) || !(graph.huber > 0.) ||
      std::set<int64_t>(graph.times.begin(), graph.times.end()).size() != graph.times.size() ||
      std::set<int64_t>(graph.ids.begin(), graph.ids.end()).size() != graph.ids.size()) {
    throw std::invalid_argument("invalid joint posterior graph metadata");
  }
  if (!std::is_sorted(graph.times.begin(), graph.times.end())) {
    throw std::invalid_argument("joint posterior clocks are not ordered");
  }
  std::set<std::tuple<int64_t, int64_t, int>> evidence;
  for (size_t i = 0; i < graph.visuals.size(); ++i) {
    const auto& visual = graph.visuals[i];
    if (visual.node < 0 || visual.node >= int(graph.times.size()) || visual.point < 0 ||
        visual.point >= int(graph.ids.size()) || graph.visual_cameras[i] < 0 ||
        !evidence.emplace(graph.times[visual.node], graph.ids[visual.point], graph.visual_cameras[i]).second) {
      throw std::invalid_argument("invalid or duplicate raw visual evidence");
    }
  }
  std::set<std::pair<int, int>> inertial;
  for (const auto& edge : graph.edges) {
    if (edge.left < 0 || edge.right <= edge.left || edge.right >= int(graph.nodes.size()) ||
        !inertial.emplace(edge.left, edge.right).second) {
      throw std::invalid_argument("invalid or duplicate raw IMU evidence");
    }
  }
}

template <int Rows, int Cols, class Function>
void Derivative(Function function, Eigen::Matrix<double, Rows, 1>& residual,
                Eigen::Matrix<double, Rows, Cols>& jacobian) {
  for (int col = 0; col < Cols; ++col) {
    gpu_vi::Jet x[Cols], y[Rows];
    for (int k = 0; k < Cols; ++k) {
      x[k] = {0., k == col ? 1. : 0.};
    }
    function(x, y);
    for (int row = 0; row < Rows; ++row) {
      residual[row] = y[row].v;
      jacobian(row, col) = y[row].d;
    }
  }
}

// Conditional experiment: hold the initialized gravity mean exact. This drops
// its uncertainty and must not be confused with the full posterior model.
bool DiagnosticFixedGravity(const Graph& graph) {
  return !graph.prior.empty() && std::getenv("VIO_POSTERIOR_DIAG_FIXED_GRAVITY");
}

void FixGauge(const Graph& graph, Matrix& state) {
  if (DiagnosticFixedGravity(graph)) {
    state.rightCols<2>().setZero();
  }
  const auto root = std::find(graph.times.begin(), graph.times.end(), graph.fixed_pose_time);
  if (root != graph.times.end()) {
    state.middleCols(15 * (root - graph.times.begin()), 6).setZero();
  }
}

Factor PriorRows(const Graph& graph) {
  const auto& prior = graph.prior;
  const int old_navigation = 15 * prior.nodes.size();
  const int old_dimension = old_navigation + 3 * prior.points.size() + 2;
  const int navigation = 15 * graph.nodes.size(), dimension = Dimension(graph);
  if (!prior.factor().valid() || prior.factor().rows.cols() != old_dimension ||
      prior.nodes.size() != prior.times.size() || prior.points.size() != prior.ids.size()) {
    throw std::invalid_argument("invalid initial joint posterior shape");
  }
  Vector delta = Vector::Zero(old_dimension);
  Factor result;
  result.rows = Matrix::Zero(prior.factor().rows.rows(), dimension);
  for (size_t i = 0; i < prior.times.size(); ++i) {
    const int node = Index(graph.times, prior.times[i]);
    Eigen::Matrix<double, 15, 1> value;
    Eigen::Matrix<double, 15, 15> derivative;
    Derivative<15, 15>([&](auto* x, auto* y) { PriorChartDifference(graph.nodes[node], prior.nodes[i], x, y); }, value,
                       derivative);
    delta.segment<15>(15 * i) = value;
    result.rows.middleCols(15 * node, 15).noalias() = prior.factor().rows.middleCols(15 * i, 15) * derivative;
  }
  for (size_t i = 0; i < prior.ids.size(); ++i) {
    const int point = Index(graph.ids, prior.ids[i]);
    delta.segment<3>(old_navigation + 3 * i) = graph.points[point] - prior.points[i];
    result.rows.middleCols(navigation + 3 * point, 3) = prior.factor().rows.middleCols(old_navigation + 3 * i, 3);
  }
  delta.tail<2>() = GravityDifference(graph.gravity, prior.gravity);
  result.rows.rightCols<2>().noalias() =
      prior.factor().rows.rightCols<2>() * GravityDifferenceJacobian(graph.gravity, prior.gravity);
  result.residual = prior.factor().rows * delta + prior.factor().residual;
  result.evidence = prior.factor().evidence;
  FixGauge(graph, result.rows);
  return result;
}

// Only raw factors have independent point blocks. The correlated prior is
// transformed in full; its off-diagonal map/pose/gravity entries are preserved.
void AddRawNormal(const Graph& graph, const Linearization& linearization, Matrix& hessian, Vector& gradient) {
  const int navigation = 15 * graph.nodes.size(), dimension = Dimension(graph);
  for (const auto& block : linearization.blocks) {
    const Matrix ss = block.state.transpose() * block.state;
    const Vector sg = block.state.transpose() * block.residual;
    hessian.topLeftCorner(navigation, navigation) += ss.topLeftCorner(navigation, navigation);
    hessian.topRightCorner(navigation, 2) += ss.topRightCorner(navigation, 2);
    hessian.bottomLeftCorner(2, navigation) += ss.bottomLeftCorner(2, navigation);
    hessian.bottomRightCorner<2, 2>() += ss.bottomRightCorner<2, 2>();
    gradient.head(navigation) += sg.head(navigation);
    gradient.tail<2>() += sg.tail<2>();
    if (block.point_index >= 0) {
      const int column = navigation + 3 * block.point_index;
      const Matrix cross = block.state.transpose() * block.point;
      hessian.block(column, column, 3, 3).noalias() += block.point.transpose() * block.point;
      hessian.block(0, column, navigation, 3) += cross.topRows(navigation);
      hessian.block(column, 0, 3, navigation) += cross.topRows(navigation).transpose();
      hessian.block(dimension - 2, column, 2, 3) += cross.bottomRows<2>();
      hessian.block(column, dimension - 2, 3, 2) += cross.bottomRows<2>().transpose();
      gradient.segment<3>(column).noalias() += block.point.transpose() * block.residual;
    }
  }
}
struct PriorNormalCache {
  Matrix information;
  Vector gradient;
};
PriorNormalCache CachePrior(const Graph& graph) {
  const auto& factor = graph.prior.factor();
  if (!factor.valid()) {
    throw std::invalid_argument("invalid prior factor cache input");
  }
  return {gaussian::dense::gram(factor.rows), factor.rows.transpose() * factor.residual};
}
Vector PriorDelta(const Graph& graph) {
  const auto& prior = graph.prior;
  const int expected = 15 * prior.nodes.size() + 3 * prior.points.size() + 2;
  if (!prior.factor().valid() || prior.factor().rows.cols() != expected || prior.nodes.size() != prior.times.size() ||
      prior.points.size() != prior.ids.size() ||
      std::set<int64_t>(prior.times.begin(), prior.times.end()).size() != prior.times.size() ||
      std::set<int64_t>(prior.ids.begin(), prior.ids.end()).size() != prior.ids.size()) {
    throw std::invalid_argument("invalid prior metadata for cached chart");
  }

  const int navigation = 15 * prior.nodes.size();
  Vector delta = Vector::Zero(navigation + 3 * prior.points.size() + 2);
  for (size_t i = 0; i < prior.nodes.size(); ++i) {
    const int node = Index(graph.times, prior.times[i]);
    Eigen::Matrix<double, 15, 1> value;
    Eigen::Matrix<double, 15, 15> derivative;
    Derivative<15, 15>([&](auto* x, auto* y) { PriorChartDifference(graph.nodes[node], prior.nodes[i], x, y); }, value,
                       derivative);
    delta.segment<15>(15 * i) = value;
  }
  for (size_t i = 0; i < prior.points.size(); ++i) {
    delta.segment<3>(navigation + 3 * i) = graph.points[Index(graph.ids, prior.ids[i])] - prior.points[i];
  }
  delta.tail<2>() = GravityDifference(graph.gravity, prior.gravity);
  return delta;
}
void CachedNormalSystem(const Graph& graph, const Linearization& linearization, const PriorNormalCache& cache,
                        Matrix& hessian, Vector& gradient, Matrix* reusable_prior = nullptr) {
  const auto& prior = graph.prior;
  const int old_navigation = 15 * prior.nodes.size(), navigation = 15 * graph.nodes.size();
  const int old_dimension = old_navigation + 3 * prior.points.size() + 2, dimension = Dimension(graph);
  const Vector delta = PriorDelta(graph);
  Matrix local_prior;
  Matrix& old_hessian = reusable_prior ? *reusable_prior : local_prior;
  old_hessian = cache.information;
  Vector old_gradient = cache.information * delta + cache.gradient;
  std::vector<Eigen::Matrix<double, 15, 15>> derivatives;
  std::vector<int> columns(old_dimension);
  for (size_t i = 0; i < prior.nodes.size(); ++i) {
    const int node = Index(graph.times, prior.times[i]);
    Eigen::Matrix<double, 15, 1> value;
    Eigen::Matrix<double, 15, 15> derivative;
    Derivative<15, 15>([&](auto* x, auto* y) { PriorChartDifference(graph.nodes[node], prior.nodes[i], x, y); }, value,
                       derivative);
    if (graph.times[node] == graph.fixed_pose_time) {
      derivative.leftCols<6>().setZero();
    }
    derivatives.push_back(derivative);
    old_hessian.middleCols(15 * i, 15) = (old_hessian.middleCols(15 * i, 15) * derivative).eval();
    for (int k = 0; k < 15; ++k) {
      columns[15 * i + k] = 15 * node + k;
    }
  }
  Eigen::Matrix2d gravity = GravityDifferenceJacobian(graph.gravity, prior.gravity);
  if (DiagnosticFixedGravity(graph)) {
    gravity.setZero();
  }
  old_hessian.rightCols<2>() = (old_hessian.rightCols<2>() * gravity).eval();
  for (size_t i = 0; i < derivatives.size(); ++i) {
    old_hessian.middleRows(15 * i, 15) = (derivatives[i].transpose() * old_hessian.middleRows(15 * i, 15)).eval();
    old_gradient.segment<15>(15 * i) = (derivatives[i].transpose() * old_gradient.segment<15>(15 * i)).eval();
  }
  old_hessian.bottomRows<2>() = (gravity.transpose() * old_hessian.bottomRows<2>()).eval();
  old_gradient.tail<2>() = (gravity.transpose() * old_gradient.tail<2>()).eval();
  for (size_t i = 0; i < prior.ids.size(); ++i) {
    const int point = Index(graph.ids, prior.ids[i]);
    for (int k = 0; k < 3; ++k) {
      columns[old_navigation + 3 * i + k] = navigation + 3 * point + k;
    }
  }
  columns[old_dimension - 2] = dimension - 2;
  columns[old_dimension - 1] = dimension - 1;
  hessian = Matrix::Zero(dimension, dimension);
  gradient = Vector::Zero(dimension);
  for (int col = 0; col < old_dimension; ++col) {
    gradient[columns[col]] = old_gradient[col];
    for (int row = 0; row < old_dimension; ++row) {
      hessian(columns[row], columns[col]) = old_hessian(row, col);
    }
  }
  AddRawNormal(graph, linearization, hessian, gradient);
}
// Unlike the older compression helper, retain the state-independent residual
// energy as a single zero-Jacobian row. H, gradient AND the profiled cost survive.
Factor CompressWithConstant(const Factor& factor) {
  Factor result;
  gaussian::dense::compress(factor.rows, factor.residual, result.rows, result.residual);
  const double discarded = factor.residual.squaredNorm() - result.residual.squaredNorm();
  const double roundoff = 128. * std::numeric_limits<double>::epsilon() *
                          std::max(factor.rows.rows(), factor.rows.cols()) * factor.residual.squaredNorm();
  if (discarded < -roundoff) {
    throw std::runtime_error("joint posterior compression changed residual energy");
  }
  const int rows = result.rows.rows();
  result.rows.conservativeResize(rows + 1, result.rows.cols());
  result.rows.bottomRows(1).setZero();
  result.residual.conservativeResize(rows + 1);
  result.residual[rows] = std::sqrt(std::max(0., discarded));
  result.evidence = factor.evidence;
  return result;
}

struct Step {
  Vector value;
  double prediction = 0., linear_residual = 0.;
  bool valid = false;
  double assemble_ms = 0., solve_ms = 0.;
};

struct DenseWorkspace {
  Matrix hessian, normalized, prior;
  Vector gradient, scale, rhs, solution;
};
bool SolveScaled(const Matrix& hessian, const Vector& gradient, Vector& step, double& error,
                 DenseWorkspace* workspace = nullptr) {
  DenseWorkspace local;
  auto& work = workspace ? *workspace : local;
  work.scale = hessian.diagonal().cwiseMax(1e-30).cwiseSqrt().cwiseInverse();
  work.normalized.noalias() = work.scale.asDiagonal() * hessian * work.scale.asDiagonal();
  work.rhs = -(work.scale.asDiagonal() * gradient);
  if (!gaussian::dense::solve_spd(work.normalized, work.rhs, work.solution)) {
    return false;
  }
  error = (work.normalized * work.solution - work.rhs).norm() / std::max(1e-30, work.rhs.norm());
  step = work.scale.asDiagonal() * work.solution;
  return step.allFinite() && std::isfinite(error) && error <= 1e-8;
}
Step DenseStep(const Graph& graph, const Linearization& linearization, double lambda, const PriorNormalCache* cache,
               DenseWorkspace* workspace) {
  const auto begin = std::chrono::steady_clock::now();
  Matrix local_hessian;
  Vector local_gradient;
  Matrix& hessian = workspace ? workspace->hessian : local_hessian;
  Vector& gradient = workspace ? workspace->gradient : local_gradient;
  if (std::getenv("VIO_POSTERIOR_BLOCK_NORMAL")) {
    if (cache) {
      CachedNormalSystem(graph, linearization, *cache, hessian, gradient, workspace ? &workspace->prior : nullptr);
    } else {
      NormalEquations(graph, linearization, hessian, gradient);
    }
  } else {
    const auto factor = Assemble(graph, linearization);
    hessian = gaussian::dense::gram(factor.rows);
    gradient = factor.rows.transpose() * factor.residual;
  }
  Matrix local_damped;
  const Vector original_diagonal = hessian.diagonal();
  if (!workspace) {
    local_damped = hessian;
  }
  Matrix& damped = workspace ? hessian : local_damped;
  const Vector metric_diagonal = block_lm_metric::Diagonal(original_diagonal);
  damped.diagonal().array() += lambda * metric_diagonal.array().max(1e-12);
  Step result;
  const auto solving = std::chrono::steady_clock::now();
  result.assemble_ms = std::chrono::duration<double, std::milli>(solving - begin).count();
  const bool success = SolveScaled(damped, gradient, result.value, result.linear_residual, workspace);
  // Restore exact original bits, not a subtract-damping round trip. Posterior
  // information and predicted gain must never contain the LM damping.
  if (workspace) {
    hessian.diagonal() = original_diagonal;
  }
  if (!success) {
    return result;
  }
  result.prediction = -gradient.dot(result.value) - .5 * result.value.dot(hessian * result.value);
  result.valid = std::isfinite(result.prediction) && result.prediction > 0.;
  result.solve_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - solving).count();
  return result;
}

Step IndependentPointStep(const Graph& graph, const Linearization& linearization, double lambda) {
  const int dimension = StateDimension(graph);
  Vector diagonal = Vector::Zero(dimension);
  std::vector<Matrix> rows;
  std::vector<Vector> residuals;
  for (const auto& block : linearization.blocks) {
    diagonal += block.state.colwise().squaredNorm().transpose();
    if (block.point_index < 0) {
      rows.push_back(block.state);
      residuals.push_back(block.residual);
      continue;
    }
    Matrix point = block.point, state = block.state;
    Vector residual = block.residual;
    const int n = point.rows();
    const Eigen::Vector3d point_diagonal = block_lm_metric::Point(point.colwise().squaredNorm().transpose());
    point.conservativeResize(n + 3, 3);
    point.bottomRows<3>() = (lambda * point_diagonal.array().max(1e-12)).sqrt().matrix().asDiagonal();
    state.conservativeResize(n + 3, dimension);
    state.bottomRows<3>().setZero();
    residual.conservativeResize(n + 3);
    residual.tail<3>().setZero();
    Matrix projected;
    Vector projected_r;
    gpu_window::project_out(point, state, residual, projected, projected_r);
    rows.push_back(std::move(projected));
    residuals.push_back(std::move(projected_r));
  }
  Factor reduced;
  int count = dimension;
  for (const auto& row : rows) {
    count += row.rows();
  }
  reduced.rows = Matrix::Zero(count, dimension);
  reduced.residual = Vector::Zero(count);
  int at = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    reduced.rows.middleRows(at, rows[i].rows()) = rows[i];
    reduced.residual.segment(at, residuals[i].size()) = residuals[i];
    at += rows[i].rows();
  }
  reduced.rows.bottomRows(dimension).diagonal() =
      (lambda * block_lm_metric::Diagonal(diagonal).array().max(1e-12)).sqrt().matrix();
  const Matrix hessian = gaussian::dense::gram(reduced.rows);
  const Vector gradient = reduced.rows.transpose() * reduced.residual;
  Vector state_step;
  Step result;
  if (!SolveScaled(hessian, gradient, state_step, result.linear_residual)) {
    return result;
  }
  const int navigation = 15 * graph.nodes.size();
  result.value = Vector::Zero(Dimension(graph));
  result.value.head(navigation) = state_step.head(navigation);
  result.value.tail<2>() = state_step.tail<2>();
  for (const auto& block : linearization.blocks) {
    Eigen::VectorXd change = block.state * state_step;
    if (block.point_index >= 0) {
      Matrix point = block.point;
      const Eigen::Vector3d point_diagonal = block_lm_metric::Point(point.colwise().squaredNorm().transpose());
      const int n = point.rows();
      point.conservativeResize(n + 3, 3);
      point.bottomRows<3>() = (lambda * point_diagonal.array().max(1e-12)).sqrt().matrix().asDiagonal();
      Vector rhs(n + 3);
      rhs.head(n) = -(block.residual + change);
      rhs.tail<3>().setZero();
      const Eigen::Vector3d step = point.colPivHouseholderQr().solve(rhs);
      result.value.segment<3>(navigation + 3 * block.point_index) = step;
      change += block.point * step;
    }
    result.prediction -= block.residual.dot(change) + .5 * change.squaredNorm();
  }
  result.valid = result.value.allFinite() && std::isfinite(result.prediction) && result.prediction > 0.;
  return result;
}
}  // namespace

Eigen::Matrix<double, 3, 2> GravityBasis(const Eigen::Vector3d& gravity) {
  if (!gravity.allFinite() || !(gravity.norm() > 0.)) {
    throw std::invalid_argument("invalid gravity chart");
  }
  const Eigen::Vector3d axis = gravity.normalized();
  Eigen::Index least;
  axis.cwiseAbs().minCoeff(&least);
  const Eigen::Vector3d first = axis.cross(Eigen::Vector3d::Unit(least)).normalized();
  Eigen::Matrix<double, 3, 2> basis;
  basis.col(0) = first;
  basis.col(1) = axis.cross(first);
  return basis;
}
Eigen::Vector3d RetractGravity(const Eigen::Vector3d& anchor, const Eigen::Vector2d& delta) {
  const Eigen::Vector3d value = anchor + GravityBasis(anchor) * delta;
  return anchor.norm() * value.normalized();
}
Eigen::Vector2d GravityDifference(const Eigen::Vector3d& value, const Eigen::Vector3d& anchor) {
  const double denominator = anchor.normalized().dot(value);
  if (!(denominator > 0.)) {
    throw std::domain_error("gravity left local hemisphere");
  }
  return anchor.norm() / denominator * GravityBasis(anchor).transpose() * value;
}
Eigen::Matrix2d GravityDifferenceJacobian(const Eigen::Vector3d& value, const Eigen::Vector3d& anchor) {
  const Eigen::Vector3d axis = anchor.normalized();
  const double denominator = axis.dot(value);
  if (!(denominator > 0.)) {
    throw std::domain_error("gravity left local hemisphere");
  }
  const auto a = GravityBasis(anchor), b = GravityBasis(value);
  return anchor.norm() / denominator * a.transpose() * b -
         anchor.norm() / (denominator * denominator) * (a.transpose() * value) * (axis.transpose() * b);
}

void AnalyticVisualForTest(const Visual& visual, const State& node, const Eigen::Vector3d& point,
                           Eigen::Vector2d& residual, Eigen::Matrix<double, 2, 18>& jacobian) {
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(node.R), C(visual.obs.Rci);
  const Eigen::Vector3d q = R.transpose() * (point - Eigen::Map<const Eigen::Vector3d>(node.p));
  const Eigen::Vector3d pc = C * q + Eigen::Map<const Eigen::Vector3d>(visual.obs.tci);
  if (!pc.allFinite() || !(pc.z() > visual.obs.min_depth)) {
    throw std::domain_error("analytic visual projection domain");
  }
  double x[18] = {}, y[2];
  visual_res(visual, node, point.data(), x, y);
  residual = Eigen::Map<const Eigen::Vector2d>(y);
  Eigen::Matrix<double, 2, 3> projection;
  projection << 1. / pc.z(), 0., -pc.x() / (pc.z() * pc.z()), 0., 1. / pc.z(), -pc.y() / (pc.z() * pc.z());
  const Eigen::Map<const Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> W(visual.obs.sqrt_info);
  const Eigen::Matrix<double, 2, 3> image = W * projection * C;
  Eigen::Matrix3d skew;
  skew << 0., -q.z(), q.y(), q.z(), 0., -q.x(), -q.y(), q.x(), 0.;
  jacobian.setZero();
  jacobian.leftCols<3>() = image * skew;
  // R^T R is intentional: inherited FP32 rotations need not be exactly
  // orthogonal, and the reference retraction translates by R*dt.
  jacobian.middleCols<3>(3) = -image * (R.transpose() * R);
  jacobian.rightCols<3>() = image * R.transpose();
}
Linearization Linearize(const Graph& graph, bool boundary) {
  Validate(graph);
  const int dimension = StateDimension(graph);
  Linearization result;
  if (!graph.prior.empty()) {
    const Vector prior_residual = std::getenv("VIO_POSTERIOR_CACHE_PRIOR")
                                      ? graph.prior.factor().rows * PriorDelta(graph) + graph.prior.factor().residual
                                      : PriorRows(graph).residual;
    // A correlated prior is already dense in landmark variables. Store it as a
    // full factor during Assemble; only raw measurements use independent blocks.
    result.cost += .5 * prior_residual.squaredNorm();
  }
  for (const auto& original : graph.edges) {
    const auto left = original.left, right = original.right;
    if (left < 0 || right < 0 || left >= int(graph.nodes.size()) || right >= int(graph.nodes.size())) {
      throw std::invalid_argument("invalid joint IMU endpoints");
    }
    if (boundary && left != 0 && right != 0) {
      continue;
    }
    if (!graph.prior.empty() && graph.times[left] <= graph.prior.imu_left_through) {
      throw std::invalid_argument("IMU factor already owned by marginal prior");
    }
    Edge edge = original;
    const auto& l = graph.nodes[left];
    const auto& r = graph.nodes[right];
    std::copy_n(l.R, 9, edge.data.R[0]);
    std::copy_n(r.R, 9, edge.data.R[1]);
    std::copy_n(l.p, 3, edge.data.p[0]);
    std::copy_n(r.p, 3, edge.data.p[1]);
    std::copy_n(l.v, 3, edge.data.v[0]);
    std::copy_n(r.v, 3, edge.data.v[1]);
    std::copy_n(l.bg, 3, edge.data.bg);
    std::copy_n(l.ba, 3, edge.data.ba);
    std::copy_n(graph.gravity.data(), 3, edge.data.gravity);
    Eigen::Matrix<double, 15, 1> residual;
    Eigen::Matrix<double, 15, 30> jacobian;
    Derivative<15, 30>([&](auto* x, auto* y) { edge_res(edge, l, r, x, y); }, residual, jacobian);
    Block block;
    block.state = Matrix::Zero(15, dimension);
    block.state.middleCols(15 * left, 15) = jacobian.leftCols<15>();
    block.state.middleCols(15 * right, 15) = jacobian.rightCols<15>();
    Eigen::Matrix<double, 9, 2> gravity = Eigen::Matrix<double, 9, 2>::Zero();
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(l.R);
    gravity.middleRows<3>(3) = -edge.data.dt * rotation.transpose() * GravityBasis(graph.gravity);
    gravity.bottomRows<3>() = -.5 * edge.data.dt * edge.data.dt * rotation.transpose() * GravityBasis(graph.gravity);
    const Eigen::Map<const Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> whiten(edge.data.imu_sqrt);
    block.state.topRightCorner<9, 2>() = whiten * gravity;
    FixGauge(graph, block.state);
    block.residual = residual;
    result.cost += .5 * residual.squaredNorm();
    result.blocks.push_back(std::move(block));
  }
  std::vector<std::vector<const Visual*>> groups(graph.points.size());
  for (size_t vi = 0; vi < graph.visuals.size(); ++vi) {
    const auto& visual = graph.visuals[vi];
    if (visual.node < 0 || visual.node >= int(graph.nodes.size()) || visual.point < 0 ||
        visual.point >= int(groups.size())) {
      throw std::invalid_argument("invalid joint visual variable index");
    }
    if (boundary && visual.node != 0) {
      continue;
    }
    const VisualKey key{graph.times[visual.node], graph.ids[visual.point], graph.visual_cameras[vi]};
    if (!graph.prior.empty() &&
        (key.time <= graph.prior.visual_through ||
         std::binary_search(graph.prior.consumed_visual.begin(), graph.prior.consumed_visual.end(), key))) {
      throw std::invalid_argument("visual factor already owned by marginal prior");
    }
    result.visual_evidence.push_back(key);
    groups[visual.point].push_back(&visual);
  }
  for (size_t point = 0; point < groups.size(); ++point) {
    if (groups[point].empty()) {
      continue;
    }
    Block block;
    block.point_index = point;
    block.state = Matrix::Zero(2 * groups[point].size(), dimension);
    block.point = Matrix::Zero(2 * groups[point].size(), 3);
    block.residual.resize(2 * groups[point].size());
    for (size_t k = 0; k < groups[point].size(); ++k) {
      const auto& visual = *groups[point][k];
      const auto& node = graph.nodes[visual.node];
      const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(node.R);
      const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> camera(visual.obs.Rci);
      const Eigen::Vector3d pc =
          camera * rotation.transpose() * (graph.points[point] - Eigen::Map<const Eigen::Vector3d>(node.p)) +
          Eigen::Map<const Eigen::Vector3d>(visual.obs.tci);
      if (!pc.allFinite() || !(pc.z() > visual.obs.min_depth)) {
        throw std::domain_error("joint visual factor outside projection domain");
      }
      Eigen::Vector2d residual;
      Eigen::Matrix<double, 2, 18> jacobian;
      if (std::getenv("VIO_POSTERIOR_ANALYTIC_VISUAL")) {
        AnalyticVisualForTest(visual, node, graph.points[point], residual, jacobian);
      } else {
        Derivative<2, 18>(
            [&](auto* x, auto* y) { visual_res(visual, graph.nodes[visual.node], graph.points[point].data(), x, y); },
            residual, jacobian);
      }
      const double norm = residual.norm();
      const double weight = norm > graph.huber ? std::sqrt(graph.huber / norm) : 1.;
      block.state.block(2 * k, 15 * visual.node, 2, 15) = weight * jacobian.leftCols<15>();
      block.point.middleRows<2>(2 * k) = weight * jacobian.rightCols<3>();
      block.residual.segment<2>(2 * k) = weight * residual;
      const double cost =
          norm > graph.huber ? graph.huber * norm - .5 * graph.huber * graph.huber : .5 * residual.squaredNorm();
      result.cost += cost;
      result.constant += cost - .5 * weight * weight * residual.squaredNorm();
    }
    FixGauge(graph, block.state);
    result.blocks.push_back(std::move(block));
  }
  return result;
}

Factor AssembleReferenceForTest(const Graph& graph, const Linearization& linearization) {
  const int navigation = 15 * graph.nodes.size(), dimension = Dimension(graph);
  std::vector<Factor> factors;
  if (!graph.prior.empty()) {
    factors.push_back(PriorRows(graph));
  }
  for (const auto& block : linearization.blocks) {
    Factor factor;
    factor.rows = Matrix::Zero(block.state.rows(), dimension);
    factor.rows.leftCols(navigation) = block.state.leftCols(navigation);
    factor.rows.rightCols<2>() = block.state.rightCols<2>();
    if (block.point_index >= 0) {
      factor.rows.middleCols(navigation + 3 * block.point_index, 3) = block.point;
    }
    factor.residual = block.residual;
    factors.push_back(std::move(factor));
  }
  if (linearization.constant > 0.) {
    Factor constant;
    constant.rows = Matrix::Zero(1, dimension);
    constant.residual = Vector::Constant(1, std::sqrt(2. * linearization.constant));
    factors.push_back(std::move(constant));
  }
  return gaussian::combine(factors, dimension);
}

Factor Assemble(const Graph& graph, const Linearization& linearization) {
  if (!std::getenv("VIO_POSTERIOR_SINGLE_ASSEMBLY")) {
    return AssembleReferenceForTest(graph, linearization);
  }
  const int navigation = 15 * graph.nodes.size(), dimension = Dimension(graph);
  std::optional<Factor> prior;
  Eigen::Index count = linearization.constant > 0. ? 1 : 0;
  if (!graph.prior.empty()) {
    prior = PriorRows(graph);
    count += prior->rows.rows();
  }
  for (const auto& block : linearization.blocks) {
    if (block.state.rows() != block.residual.size() || block.state.cols() != navigation + 2 ||
        (block.point_index >= 0 && (block.point.rows() != block.state.rows() || block.point.cols() != 3 ||
                                    block.point_index >= int(graph.points.size())))) {
      throw std::invalid_argument("invalid direct factor block");
    }
    count += block.state.rows();
    if (count > std::numeric_limits<int>::max()) {
      throw std::overflow_error("direct factor row count overflow");
    }
  }
  Factor result;
  result.rows = Matrix::Zero(count, dimension);
  result.residual = Vector::Zero(count);
  Eigen::Index at = 0;
  if (prior) {
    result.rows.topRows(prior->rows.rows()) = prior->rows;
    result.residual.head(prior->residual.size()) = prior->residual;
    at = prior->rows.rows();
    const std::set<uint64_t> evidence(prior->evidence.begin(), prior->evidence.end());
    if (evidence.size() != prior->evidence.size()) {
      throw std::invalid_argument("duplicate direct factor evidence");
    }
    result.evidence.assign(evidence.begin(), evidence.end());
  }
  for (const auto& block : linearization.blocks) {
    const int n = block.state.rows();
    result.rows.block(at, 0, n, navigation) = block.state.leftCols(navigation);
    result.rows.block(at, dimension - 2, n, 2) = block.state.rightCols<2>();
    if (block.point_index >= 0) {
      result.rows.block(at, navigation + 3 * block.point_index, n, 3) = block.point;
    }
    result.residual.segment(at, n) = block.residual;
    at += n;
  }
  if (linearization.constant > 0.) {
    result.residual[at] = std::sqrt(2. * linearization.constant);
  }
  if (!result.valid()) {
    throw std::invalid_argument("nonfinite direct factor");
  }
  return result;
}

void NormalEquations(const Graph& graph, const Linearization& linearization, Matrix& hessian, Vector& gradient) {
  hessian = Matrix::Zero(Dimension(graph), Dimension(graph));
  gradient = Vector::Zero(Dimension(graph));
  if (!graph.prior.empty()) {
    const auto prior = PriorRows(graph);
    hessian = gaussian::dense::gram(prior.rows);
    gradient = prior.rows.transpose() * prior.residual;
  }
  AddRawNormal(graph, linearization, hessian, gradient);
}
void CachedNormalEquationsForTest(const Graph& graph, const Linearization& linearization, Matrix& hessian,
                                  Vector& gradient) {
  if (graph.prior.empty()) {
    NormalEquations(graph, linearization, hessian, gradient);
    return;
  }
  CachedNormalSystem(graph, linearization, CachePrior(graph), hessian, gradient);
}

Graph Retract(const Graph& graph, const Vector& increment) {
  if (increment.size() != Dimension(graph) || !increment.allFinite()) {
    throw std::invalid_argument("invalid joint posterior update");
  }
  Graph result = graph;
  const int navigation = 15 * graph.nodes.size();
  for (size_t i = 0; i < result.nodes.size(); ++i) {
    auto& node = result.nodes[i];
    Vector delta = increment.segment<15>(15 * i);
    if (graph.times[i] == graph.fixed_pose_time) {
      delta.head<6>().setZero();
    }
    pose(graph.nodes[i], delta.data(), node.R, node.p, node.v);
    for (int k = 0; k < 3; ++k) {
      node.bg[k] += delta[9 + k];
      node.ba[k] += delta[12 + k];
    }
  }
  for (size_t i = 0; i < result.points.size(); ++i) {
    result.points[i] += increment.segment<3>(navigation + 3 * i);
  }
  result.gravity = DiagnosticFixedGravity(graph) ? graph.gravity : RetractGravity(graph.gravity, increment.tail<2>());
  return result;
}

Report Optimize(Graph& graph, Linearization* final_linearization) {
  const auto linear_start = std::chrono::steady_clock::now();
  auto current = Linearize(graph, false);
  Report report;
  report.linearize_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - linear_start).count();
  report.before = current.cost;
  // Marginalization preserves irreducible historical energy in explicit
  // zero-Jacobian rows. That energy is useful for reporting, but adding a
  // constant must not change LM acceptance or convergence. Sum the variable
  // objective directly (rather than subtracting two large energies), retaining
  // the complete prior and all its correlations for Jacobians/covariances.
  std::vector<int> variable_prior_rows;
  bool has_fixed_prior_rows = false;
  if (!graph.prior.empty()) {
    const Vector row_max = graph.prior.factor().rows.cwiseAbs().rowwise().maxCoeff();
    for (int row = 0; row < row_max.size(); ++row) {
      if (row_max[row] == 0.)
        has_fixed_prior_rows = true;
      else
        variable_prior_rows.push_back(row);
    }
  }
  auto solver_cost = [&](const Graph& value, const Linearization& linear) {
    if (!has_fixed_prior_rows) return linear.cost;
    const auto start = std::chrono::steady_clock::now();
    const Vector residual = value.prior.factor().rows * PriorDelta(value) + value.prior.factor().residual;
    double cost = 0.;
    for (int row : variable_prior_rows) cost += .5 * residual[row] * residual[row];
    for (const auto& block : linear.blocks) cost += .5 * block.residual.squaredNorm();
    cost += linear.constant;  // Huber's state-dependent correction is retained.
    report.linearize_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return cost;
  };
  double current_solver_cost = solver_cost(graph, current);
  const double initial_solver_cost = current_solver_cost;
  double lambda = 1e-3;
  if (!std::isfinite(current.cost)) {
    return report;
  }
  std::optional<DenseWorkspace> workspace;
  if (std::getenv("VIO_POSTERIOR_CPU_WORKSPACE")) {
    workspace.emplace();
  }
  std::optional<PriorNormalCache> cache;
  if (!graph.prior.empty() && std::getenv("VIO_POSTERIOR_CACHE_PRIOR")) {
    const auto start = std::chrono::steady_clock::now();
    cache = CachePrior(graph);
    report.assemble_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
  for (int i = 0; i < graph.iterations && current_solver_cost > 1e-12; ++i) {
    ++report.iterations;
    const auto step = graph.prior.empty() ? IndependentPointStep(graph, current, lambda)
                                          : DenseStep(graph, current, lambda, cache ? &*cache : nullptr,
                                                      workspace ? &*workspace : nullptr);
    report.assemble_ms += step.assemble_ms;
    report.linear_solve_ms += step.solve_ms;
    if (!step.valid) {
      lambda = std::min(1e12, 10. * lambda);
      continue;
    }
    report.max_linear_residual = std::max(report.max_linear_residual, step.linear_residual);
    const auto retract_start = std::chrono::steady_clock::now();
    auto trial = Retract(graph, step.value);
    report.retract_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - retract_start).count();
    const auto trial_start = std::chrono::steady_clock::now();
    Linearization next;
    try {
      next = Linearize(trial, false);
      report.linearize_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trial_start).count();
    } catch (const std::domain_error&) {
      lambda = std::min(1e12, 10. * lambda);
      continue;
    }
    const double next_solver_cost = solver_cost(trial, next);
    const double gain = (current_solver_cost - next_solver_cost) / step.prediction;
    if (std::isfinite(next.cost) && std::isfinite(next_solver_cost) && next_solver_cost <= current_solver_cost &&
        gain >= .25) {
      graph = std::move(trial);
      current = std::move(next);
      current_solver_cost = next_solver_cost;
      if (gain > .75) {
        lambda = std::max(1e-8, .3 * lambda);
      }
      if (step.value.squaredNorm() < 1e-10 || step.prediction / std::max(1., current_solver_cost) < 1e-6) {
        break;
      }
    } else {
      lambda = std::min(1e12, 10. * lambda);
    }
  }
  report.after = current.cost;
  report.success =
      std::isfinite(current.cost) && std::isfinite(current_solver_cost) && current_solver_cost <= initial_solver_cost;
  if (final_linearization) {
    *final_linearization = std::move(current);
  }
  return report;
}

Prior StoreLinearizedPrior(Prior prior, Factor local) {
  // The rows above differentiate a retraction at the current estimate. The
  // stored prior is evaluated using PriorChartDifference(current, anchor), whose local
  // derivative is not exactly identity for inherited FP32 rotations. Convert
  // rows AND the nonzero residual into that exact chart before retaining them.
  const Matrix original_rows = local.rows;
  for (size_t i = 0; i < prior.nodes.size(); ++i) {
    Eigen::Matrix<double, 15, 1> origin;
    Eigen::Matrix<double, 15, 15> derivative;
    Derivative<15, 15>([&](auto* x, auto* y) { PriorChartDifference(prior.nodes[i], prior.nodes[i], x, y); }, origin, derivative);
    const Eigen::FullPivLU<Eigen::Matrix<double, 15, 15>> inverse(derivative);
    if (!inverse.isInvertible()) {
      throw std::runtime_error("singular stored navigation chart");
    }
    const Matrix rows = original_rows.middleCols(15 * i, 15) * inverse.inverse();
    local.rows.middleCols(15 * i, 15) = rows;
    local.residual -= rows * origin;
  }
  const Eigen::Matrix2d derivative = GravityDifferenceJacobian(prior.gravity, prior.gravity);
  local.rows.rightCols<2>() = original_rows.rightCols<2>() * derivative.inverse();
  local.residual -= local.rows.rightCols<2>() * GravityDifference(prior.gravity, prior.gravity);
  prior.SetFactor(std::move(local));
  return prior;
}

Factor CompressLocalFactor(const Factor& factor) { return CompressWithConstant(factor); }

Prior Marginalize(const Graph& graph, const Linearization& consumed, const std::vector<int64_t>& keep_times,
                  const std::vector<int64_t>& keep_ids, int64_t visual_through, int64_t imu_left_through) {
  // Eliminate unused points separately BEFORE allocating the dense retained
  // factor. Initial histories can contain thousands of independent landmarks.
  const int navigation = 15 * graph.nodes.size();
  const int dimension = navigation + 3 * keep_ids.size() + 2;
  std::vector<Factor> factors;
  if (!graph.prior.empty()) {
    const auto all = Assemble(graph, consumed);
    std::vector<int> columns;
    for (int k = 0; k < navigation; ++k) {
      columns.push_back(k);
    }
    for (auto id : keep_ids) {
      const int point = Index(graph.ids, id);
      for (int k = 0; k < 3; ++k) {
        columns.push_back(navigation + 3 * point + k);
      }
    }
    columns.push_back(Dimension(graph) - 2);
    columns.push_back(Dimension(graph) - 1);
    factors.push_back(gaussian::eliminate(all, columns));
  } else {
    for (const auto& block : consumed.blocks) {
      const auto found = block.point_index < 0
                             ? keep_ids.end()
                             : std::find(keep_ids.begin(), keep_ids.end(), graph.ids[block.point_index]);
      Matrix state = block.state;
      Vector residual = block.residual;
      if (block.point_index >= 0 && found == keep_ids.end()) {
        gpu_window::project_out(block.point, block.state, block.residual, state, residual);
      }
      Factor factor;
      factor.rows = Matrix::Zero(state.rows(), dimension);
      factor.rows.leftCols(navigation) = state.leftCols(navigation);
      factor.rows.rightCols<2>() = state.rightCols<2>();
      if (found != keep_ids.end()) {
        factor.rows.middleCols(navigation + 3 * (found - keep_ids.begin()), 3) = block.point;
      }
      factor.residual = residual;
      factors.push_back(std::move(factor));
    }
    if (consumed.constant > 0.) {
      Factor constant;
      constant.rows = Matrix::Zero(1, dimension);
      constant.residual = Vector::Constant(1, std::sqrt(2. * consumed.constant));
      factors.push_back(std::move(constant));
    }
  }
  const auto all = gaussian::combine(factors, dimension);
  std::vector<int> columns;
  for (auto time : keep_times) {
    const int node = Index(graph.times, time);
    for (int k = 0; k < 15; ++k) {
      columns.push_back(15 * node + k);
    }
  }
  for (int k = navigation; k < dimension; ++k) {
    columns.push_back(k);
  }
  auto compact = CompressWithConstant(gaussian::eliminate(all, columns));
  Prior prior;

  prior.times = keep_times;
  prior.ids = keep_ids;
  for (auto time : keep_times) {
    prior.nodes.push_back(graph.nodes[Index(graph.times, time)]);
  }
  for (auto id : keep_ids) {
    prior.points.push_back(graph.points[Index(graph.ids, id)]);
  }
  prior.gravity = graph.gravity;
  prior.fixed_pose_time = graph.fixed_pose_time;
  prior.visual_through = visual_through;
  prior.imu_left_through = imu_left_through;
  if (!keep_times.empty() && visual_through >= *std::min_element(keep_times.begin(), keep_times.end())) {
    throw std::invalid_argument("visual retirement cutoff overlaps retained frames");
  }
  for (const auto& key : graph.prior.consumed_visual) {
    if (key.time > visual_through) {
      prior.consumed_visual.push_back(key);
    }
  }
  for (const auto& key : consumed.visual_evidence) {
    if (key.time > visual_through) {
      prior.consumed_visual.push_back(key);
    }
  }
  std::sort(prior.consumed_visual.begin(), prior.consumed_visual.end());
  if (std::adjacent_find(prior.consumed_visual.begin(), prior.consumed_visual.end()) != prior.consumed_visual.end()) {
    throw std::invalid_argument("visual evidence absorbed twice at a retained frame");
  }
  return StoreLinearizedPrior(std::move(prior), std::move(compact));
}

bool CurrentMarginal(const Graph& graph, int64_t time, Matrix& covariance, Vector& gradient,
                     const Linearization* final_linearization) {
  const auto factor =
      final_linearization ? Assemble(graph, *final_linearization) : Assemble(graph, Linearize(graph, false));
  const int node = Index(graph.times, time);
  std::vector<int> columns;
  for (int k = 0; k < 15; ++k) {
    if (graph.fixed_pose_time != time || k >= 6) {
      columns.push_back(15 * node + k);
    }
  }
  const auto marginal = gaussian::marginal(factor, columns);
  if (!marginal.observable) {
    return false;
  }
  covariance = marginal.covariance;
  gradient = marginal.gradient;
  return true;
}
}  // namespace gpu_window::posterior
