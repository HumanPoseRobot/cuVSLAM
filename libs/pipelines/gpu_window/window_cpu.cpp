#include "window_math.h"

#include <Eigen/Cholesky>
#include <algorithm>
#include <chrono>

namespace gpu_window {
namespace {
using Clock = std::chrono::steady_clock;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using gpu_vi::Jet;

template<int Rows, int Columns, class Factor>
void linearize(const float* state, Factor factor, float* residual, float* jacobian) {
  if (!jacobian) {
    double x[Columns], y[Rows];
    std::copy_n(state, Columns, x);
    factor(x, y);
    std::copy_n(y, Rows, residual);
    return;
  }
  for (int c = 0; c < Columns; ++c) {
    Jet x[Columns], y[Rows];
    for (int k = 0; k < Columns; ++k) {
      x[k] = Jet(state[k], k == c);
    }
    factor(x, y);
    for (int k = 0; k < Rows; ++k) {
      residual[k] = y[k].v;
      jacobian[k * Columns + c] = y[k].d;
    }
  }
}

Evaluation EvaluateHost(const Problem& p, const Result& r, bool jacobian) {
  Evaluation out;
  out.edge.resize(15 * p.edges.size());
  out.visual.resize(2 * p.visuals.size());
  out.prior.resize(Dim);
  if (jacobian) {
    out.edge_j.resize(450 * p.edges.size());
    out.visual_j.resize(36 * p.visuals.size());
    out.prior_j.resize(Dim * Dim);
  }
  for (size_t i = 0; i < p.edges.size(); ++i) {
    const auto& edge = p.edges[i];
    float x[30];
    std::copy_n(r.x.data() + 15 * edge.left, 15, x);
    std::copy_n(r.x.data() + 15 * edge.right, 15, x + 15);
    linearize<15, 30>(x, [&](auto* a, auto* b) { edge_res(edge, p.nodes[edge.left], p.nodes[edge.right], a, b); },
                      out.edge.data() + 15 * i, jacobian ? out.edge_j.data() + 450 * i : nullptr);
  }
  for (size_t i = 0; i < p.visuals.size(); ++i) {
    const auto& v = p.visuals[i];
    float x[18];
    std::copy_n(r.x.data() + 15 * v.node, 15, x);
    std::copy_n(r.points.data() + 3 * v.point, 3, x + 15);
    linearize<2, 18>(x, [&](auto* a, auto* b) { visual_res(v, p.nodes[v.node], p.points[v.point].data(), a, b); },
                      out.visual.data() + 2 * i, jacobian ? out.visual_j.data() + 36 * i : nullptr);
  }
  const int n = p.count * 15;
  Eigen::Map<const Eigen::Matrix<double, Dim, Dim, Eigen::RowMajor>> weights(p.W.data());
  Eigen::Matrix<double, Dim, 1> delta = Eigen::Matrix<double, Dim, 1>::Zero();
  Matrix jac;
  if (jacobian) {
    jac = Matrix::Zero(n, n);
  }
  for (int i = 0; i < p.count; ++i) {
    float y[15], j[225];
    linearize<15, 15>(r.x.data() + 15 * i,
                      [&](auto* a, auto* b) { difference(p.nodes[i], p.anchors[i], a, b); },
                      y, jacobian ? j : nullptr);
    delta.segment<15>(15 * i) = Eigen::Map<Eigen::Vector<float, 15>>(y).cast<double>();
    if (jacobian) {
      jac.block<15, 15>(15 * i, 15 * i) =
          Eigen::Map<Eigen::Matrix<float, 15, 15, Eigen::RowMajor>>(j).cast<double>();
    }
  }
  Eigen::Map<Eigen::VectorXf> prior(out.prior.data(), Dim);
  prior = (weights * delta + Eigen::Map<const Eigen::Vector<double, Dim>>(p.offset.data())).cast<float>();
  if (jacobian) {
    Eigen::Map<Eigen::Matrix<float, Dim, Dim, Eigen::RowMajor>> prior_j(out.prior_j.data());
    prior_j.topLeftCorner(n, n) = (weights.topLeftCorner(n, n) * jac).cast<float>();
  }
  return out;
}

double cost(const Evaluation& e, double huber) {
  double total = 0;
  for (float x : e.edge) {
    total += .5 * double(x) * x;
  }
  for (float x : e.prior) {
    total += .5 * double(x) * x;
  }
  for (size_t i = 0; i < e.visual.size(); i += 2) {
    const double norm = std::hypot(double(e.visual[i]), double(e.visual[i + 1]));
    total += norm > huber ? huber * (norm - .5 * huber) : .5 * norm * norm;
  }
  return total;
}

struct Landmark {
  explicit Landmark(int n) : cross(Matrix::Zero(n, 3)) {}
  Matrix cross;
  Eigen::Matrix3d h = Eigen::Matrix3d::Zero();
  Eigen::Vector3d g = Eigen::Vector3d::Zero();
  Eigen::LDLT<Eigen::Matrix3d> factor;
};

struct LinearSystem {
  explicit LinearSystem(int n, size_t points) : h(Matrix::Zero(n, n)), g(Vector::Zero(n)) {
    landmarks.reserve(points);
    for (size_t i = 0; i < points; ++i) {
      landmarks.emplace_back(n);
    }
  }
  Matrix h;
  Vector g;
  std::vector<Landmark> landmarks;
};

LinearSystem assembleSystem(const Problem& p, const Evaluation& e) {
  const int n = p.count * 15;
  LinearSystem system(n, p.points.size());
  const Matrix prior = Eigen::Map<const Eigen::Matrix<float, Dim, Dim, Eigen::RowMajor>>(e.prior_j.data())
                           .topLeftCorner(n, n).cast<double>();
  system.h.noalias() = prior.transpose() * prior;
  system.g.noalias() = prior.transpose() * Eigen::Map<const Eigen::VectorXf>(e.prior.data(), n).cast<double>();
  for (size_t i = 0; i < p.edges.size(); ++i) {
    const auto& edge = p.edges[i];
    const Matrix j = Eigen::Map<const Eigen::Matrix<float, 15, 30, Eigen::RowMajor>>(e.edge_j.data() + 450 * i)
                         .cast<double>();
    const Vector residual = Eigen::Map<const Eigen::VectorXf>(e.edge.data() + 15 * i, 15).cast<double>();
    const int left = edge.left * 15, right = edge.right * 15;
    const Matrix h = j.transpose() * j;
    system.h.block<15, 15>(left, left) += h.topLeftCorner<15, 15>();
    system.h.block<15, 15>(left, right) += h.topRightCorner<15, 15>();
    system.h.block<15, 15>(right, left) += h.bottomLeftCorner<15, 15>();
    system.h.block<15, 15>(right, right) += h.bottomRightCorner<15, 15>();
    system.g.segment<15>(left) += j.leftCols(15).transpose() * residual;
    system.g.segment<15>(right) += j.rightCols(15).transpose() * residual;
  }
  for (size_t i = 0; i < p.visuals.size(); ++i) {
    const auto& observation = p.visuals[i];
    const int row = observation.node * 15;
    auto& point = system.landmarks[observation.point];
    const auto j = Eigen::Map<const Eigen::Matrix<float, 2, 18, Eigen::RowMajor>>(e.visual_j.data() + 36 * i);
    const Eigen::Vector2d residual = Eigen::Map<const Eigen::Vector2f>(e.visual.data() + 2 * i).cast<double>();
    const double norm = residual.norm();
    const double weight = norm > p.huber ? p.huber / norm : 1.;
    const Eigen::Matrix<double, 2, 15> a = j.leftCols<15>().cast<double>();
    const Eigen::Matrix<double, 2, 3> b = j.rightCols<3>().cast<double>();
    system.h.block<15, 15>(row, row).noalias() += weight * a.transpose() * a;
    system.g.segment<15>(row).noalias() += weight * a.transpose() * residual;
    point.h.noalias() += weight * b.transpose() * b;
    point.g.noalias() += weight * b.transpose() * residual;
    point.cross.middleRows(row, 15).noalias() += weight * a.transpose() * b;
  }
  return system;
}

}  // namespace

Evaluation EvaluateCpu(const Problem& p, const Result& r) {
  return EvaluateHost(p, r, true);
}

Result OptimizeCpu(const Problem& p) {
  const int n = p.count * 15;
  Result result;
  result.points.resize(3 * p.points.size());
  double current = cost(EvaluateHost(p, result, false), p.huber);
  result.before = current;
  double lambda = 1e-3;
  int rejected = 0;
  for (; result.iterations < p.iterations && current >= 1e-7; ++result.iterations) {
    auto system = assembleSystem(p, EvaluateCpu(p, result));
    Matrix h = system.h;
    h.diagonal().array() += lambda * system.h.diagonal().array().max(1e-12);
    Vector g = system.g;
    const auto eliminate_start = Clock::now();
    for (auto& point : system.landmarks) {
      Eigen::Matrix3d local = point.h;
      local.diagonal().array() += lambda * point.h.diagonal().array().max(1e-12);
      point.factor.compute(local);
      h.noalias() -= point.cross * point.factor.solve(point.cross.transpose());
      g.noalias() -= point.cross * point.factor.solve(point.g);
    }
    result.landmark_ms += std::chrono::duration<double, std::milli>(Clock::now() - eliminate_start).count();
    const Eigen::LDLT<Matrix> factor(h);
    if (factor.info() != Eigen::Success || !factor.isPositive()) {
      break;
    }
    const Vector dx = factor.solve(-g);
    if (!dx.allFinite()) {
      break;
    }
    Result trial = result;
    for (int i = 0; i < n; ++i) {
      trial.x[i] += dx[i];
    }
    const auto recover_start = Clock::now();
    double step = dx.squaredNorm();
    double damped_step = dx.dot(system.h.diagonal().array().max(1e-12).matrix().cwiseProduct(dx));
    double predicted = dx.dot(system.h * dx);
    for (size_t i = 0; i < system.landmarks.size(); ++i) {
      const auto& point = system.landmarks[i];
      const Eigen::Vector3d cross = point.cross.transpose() * dx;
      const Eigen::Vector3d dp = point.factor.solve(-point.g - cross);
      predicted += dp.dot(point.h * dp + 2 * cross);
      step += dp.squaredNorm();
      damped_step += dp.dot(point.h.diagonal().array().max(1e-12).matrix().cwiseProduct(dp));
      for (int j = 0; j < 3; ++j) {
        trial.points[3 * i + j] += dp[j];
      }
    }
    result.landmark_ms += std::chrono::duration<double, std::milli>(Clock::now() - recover_start).count();
    const double updated = cost(EvaluateHost(p, trial, false), p.huber);
    const double relative = (predicted + 2 * lambda * damped_step) / current;
    const bool converged = step < 1e-10 || relative < 1e-6 || updated < 1e-7;
    const double rho = (1. - updated / current) / relative;
    if ((converged && updated <= current) || (!converged && rho >= .25)) {
      result.x = trial.x;
      result.points.swap(trial.points);
      current = updated;
      rejected = 0;
      if (rho > .75) {
        lambda = std::clamp(.3 * lambda, 1e-6, 1e6);
      }
    } else if (!converged) {
      lambda *= 10;
      if (++rejected >= 5) {
        break;
      }
    }
    if (converged) {
      break;
    }
  }
  result.after = current;
  return result;
}
}  // namespace gpu_window
