#include "window_fp64.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include "common/log.h"
#include "window_math.h"
#include "window_validation.h"

namespace gpu_window::fp64 {
namespace {
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::VectorXd;
template <int Rows, int Cols, class F>
void Linearize(const double* x, F f, double* r, double* j) {
  if (!j) {
    f(x, r);
    return;
  }
  for (int col = 0; col < Cols; ++col) {
    gpu_vi::Jet a[Cols], b[Rows];
    for (int k = 0; k < Cols; ++k) {
      a[k] = {x[k], k == col ? 1. : 0.};
    }
    f(a, b);
    for (int k = 0; k < Rows; ++k) {
      r[k] = b[k].v;
      j[k * Cols + col] = b[k].d;
    }
  }
}
double Cost(const Evaluation& e, double huber) {
  double c = 0;
  for (double v : e.prior) {
    c += .5 * v * v;
  }
  for (double v : e.edge) {
    c += .5 * v * v;
  }
  for (size_t i = 0; i < e.visual.size(); i += 2) {
    double r = std::hypot(e.visual[i], e.visual[i + 1]);
    c += r > huber ? huber * (r - .5 * huber) : .5 * r * r;
  }
  return c;
}
struct Point {
  Matrix B;
  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  Eigen::Vector3d g = Eigen::Vector3d::Zero();
  explicit Point(int n) : B(Matrix::Zero(n, 3)) {}
};
class Cpu final : public Backend {
  const Problem& p;
  Matrix H;
  Vector g;
  std::vector<Point> points;

public:
  explicit Cpu(const Problem& problem) : p(problem) {}
  double Evaluate(const std::vector<double>& x, bool jacobian) override {
    const auto e = EvaluateHost(p, x, jacobian);
    if (!jacobian) {
      return Cost(e, p.huber);
    }
    const int n = 15 * p.count;
    Eigen::Map<const Matrix> J(e.prior_j.data(), n, n);
    H = J.transpose() * J;
    g = J.transpose() * Eigen::Map<const Vector>(e.prior.data(), n);
    points.clear();
    for (size_t i = 0; i < p.points.size(); ++i) {
      points.emplace_back(n);
    }
    for (size_t z = 0; z < p.edges.size(); ++z) {
      Matrix j = Matrix::Zero(15, n);
      Eigen::Map<const Matrix> a(e.edge_j.data() + 450 * z, 15, 30);
      j.middleCols(15 * p.edges[z].left, 15) = a.leftCols(15);
      j.middleCols(15 * p.edges[z].right, 15) = a.rightCols(15);
      H.noalias() += j.transpose() * j;
      g.noalias() += j.transpose() * Eigen::Map<const Vector>(e.edge.data() + 15 * z, 15);
    }
    for (size_t z = 0; z < p.visuals.size(); ++z) {
      const auto& v = p.visuals[z];
      Eigen::Map<const Matrix> j(e.visual_j.data() + 36 * z, 2, 18);
      Eigen::Map<const Eigen::Vector2d> r(e.visual.data() + 2 * z);
      double w = r.norm() > p.huber ? p.huber / r.norm() : 1.;
      int k = 15 * v.node;
      H.block(k, k, 15, 15).noalias() += w * j.leftCols(15).transpose() * j.leftCols(15);
      g.segment(k, 15).noalias() += w * j.leftCols(15).transpose() * r;
      auto& pt = points[v.point];
      pt.B.middleRows(k, 15).noalias() += w * j.leftCols(15).transpose() * j.rightCols(3);
      pt.C.noalias() += w * j.rightCols(3).transpose() * j.rightCols(3);
      pt.g.noalias() += w * j.rightCols(3).transpose() * r;
    }
    return Cost(e, p.huber);
  }
  Step Solve(double lambda) override {
    const int n = H.rows();
    Matrix S = H;
    S.diagonal().array() += lambda * H.diagonal().array().max(1e-12);
    Vector rhs = -g;
    std::vector<Eigen::LLT<Eigen::Matrix3d>> factors(points.size());
    Step step;
    for (size_t k = 0; k < points.size(); ++k) {
      const auto& p = points[k];
      Eigen::Matrix3d c = p.C;
      c.diagonal().array() += lambda * p.C.diagonal().array().max(1e-12);
      factors[k].compute(c);
      if (factors[k].info() != Eigen::Success) {
        return step;
      }
      S.noalias() -= p.B * factors[k].solve(p.B.transpose());
      rhs.noalias() += p.B * factors[k].solve(p.g);
    }
    Eigen::LLT<Matrix> f(S);
    if (f.info() != Eigen::Success) {
      return step;
    }
    Vector dx = f.solve(rhs);
    if (!dx.allFinite()) {
      return step;
    }
    step.dx.resize(n + 3 * points.size());
    Eigen::Map<Vector>(step.dx.data(), n) = dx;
    Vector hdx = H * dx;
    double quadratic = dx.dot(hdx), linear = g.dot(dx), err2 = 0, rhs2 = g.squaredNorm();
    Vector residual = hdx + g + lambda * H.diagonal().array().max(1e-12).matrix().cwiseProduct(dx);
    for (size_t k = 0; k < points.size(); ++k) {
      const auto& p = points[k];
      Eigen::Vector3d dp = factors[k].solve(-p.g - p.B.transpose() * dx);
      if (!dp.allFinite()) {
        return Step{};
      }
      Eigen::Map<Eigen::Vector3d>(step.dx.data() + n + 3 * k) = dp;
      quadratic += dp.dot(p.C * dp + 2 * p.B.transpose() * dx);
      linear += p.g.dot(dp);
      residual.noalias() += p.B * dp;
      Eigen::Vector3d rr =
          p.C * dp + p.B.transpose() * dx + p.g + lambda * p.C.diagonal().array().max(1e-12).matrix().cwiseProduct(dp);
      err2 += rr.squaredNorm();
      rhs2 += p.g.squaredNorm();
    }
    step.predicted = -linear - .5 * quadratic;
    step.relative_residual = std::sqrt((err2 + residual.squaredNorm()) / std::max(rhs2, 1e-30));
    step.valid =
        std::isfinite(step.predicted) && std::isfinite(step.relative_residual) && step.relative_residual < 1e-6;
    return step;
  }
};
}  // namespace

Evaluation EvaluateHost(const Problem& p, const std::vector<double>& x, bool jacobian) {
  const int n = 15 * p.count;
  Evaluation e;
  e.prior.resize(n);
  e.edge.resize(15 * p.edges.size());
  e.visual.resize(2 * p.visuals.size());
  if (jacobian) {
    e.prior_j.resize(n * n);
    e.edge_j.resize(450 * p.edges.size());
    e.visual_j.resize(36 * p.visuals.size());
  }
  Matrix dj = Matrix::Zero(n, n);
  Vector delta(n);
  for (int i = 0; i < p.count; ++i) {
    double j[225];
    Linearize<15, 15>(
        x.data() + 15 * i, [&](auto* a, auto* b) { difference(p.nodes[i], p.anchors[i], a, b); }, delta.data() + 15 * i,
        jacobian ? j : nullptr);
    if (jacobian) {
      dj.block(15 * i, 15 * i, 15, 15) = Eigen::Map<Matrix>(j, 15, 15);
    }
  }
  Eigen::Map<const Matrix> W(p.W.data(), Dim, Dim);
  Eigen::Map<Vector>(e.prior.data(), n) = W.topLeftCorner(n, n) * delta + Eigen::Map<const Vector>(p.offset.data(), n);
  if (jacobian) {
    Eigen::Map<Matrix>(e.prior_j.data(), n, n) = W.topLeftCorner(n, n) * dj;
  }
  for (size_t i = 0; i < p.edges.size(); ++i) {
    const auto& v = p.edges[i];
    double a[30];
    std::copy_n(x.data() + 15 * v.left, 15, a);
    std::copy_n(x.data() + 15 * v.right, 15, a + 15);
    Linearize<15, 30>(
        a, [&](auto* a, auto* b) { edge_res(v, p.nodes[v.left], p.nodes[v.right], a, b); }, e.edge.data() + 15 * i,
        jacobian ? e.edge_j.data() + 450 * i : nullptr);
  }
  for (size_t i = 0; i < p.visuals.size(); ++i) {
    const auto& v = p.visuals[i];
    double a[18];
    std::copy_n(x.data() + 15 * v.node, 15, a);
    std::copy_n(x.data() + n + 3 * v.point, 3, a + 15);
    Linearize<2, 18>(
        a, [&](auto* a, auto* b) { visual_res(v, p.nodes[v.node], p.points[v.point].data(), a, b); },
        e.visual.data() + 2 * i, jacobian ? e.visual_j.data() + 36 * i : nullptr);
  }
  return e;
}
std::unique_ptr<Backend> MakeCpu(const Problem& p) { return std::make_unique<Cpu>(p); }
Report Minimize(const Problem& p, Backend& backend) {
  Report r;
  r.x.resize(15 * p.count + 3 * p.points.size());
  double current = backend.Evaluate(r.x, true), lambda = 1e-3;
  r.before = current;
  int consecutive = 0;
  for (int i = 0; i < p.iterations && std::isfinite(current) && current >= 1e-7; ++i) {
    ++r.iterations;
    auto step = backend.Solve(lambda);
    if (!step.valid || !(step.predicted > 0)) {
      ++r.failed;
      if (++consecutive >= 5) {
        break;
      }
      lambda = std::min(1e6, lambda * 10);
      continue;
    }
    r.max_linear_residual = std::max(r.max_linear_residual, step.relative_residual);
    auto trial = r.x;
    double norm2 = 0;
    for (size_t j = 0; j < trial.size(); ++j) {
      trial[j] += step.dx[j];
      norm2 += step.dx[j] * step.dx[j];
    }
    double updated = backend.Evaluate(trial, false);
    double rho = (current - updated) / step.predicted;
    const bool converged = norm2 < 1e-10 || step.predicted / std::max(current, 1e-30) < 1e-6 || updated < 1e-7;
    // Both implementations use exactly this controller, including finite checks.
    if (std::isfinite(updated) && updated <= current && (converged || rho >= .25)) {
      r.x.swap(trial);
      current = updated;
      consecutive = 0;
      if (rho > .75) {
        lambda = std::max(1e-6, .3 * lambda);
      }
      if (converged) {
        break;
      }
      current = backend.Evaluate(r.x, true);
    } else {
      ++r.rejected;
      if (++consecutive >= 5) {
        break;
      }
      lambda = std::min(1e6, lambda * 10);
    }
  }
  r.after = current;
  return r;
}
Result Optimize(const Problem& p, bool gpu) {
  Result invalid;
  invalid.before = invalid.after = std::numeric_limits<float>::quiet_NaN();
  if (const char* reason = invalid_problem(p)) {
    TraceMessage("WINDOW_NUMERIC_REJECT stage=input reason=%s", reason);
    return invalid;
  }
  auto backend = gpu ? MakeGpu(p) : MakeCpu(p);
  auto r = Minimize(p, *backend);
  if (!float_representable(r.before) || !float_representable(r.after) ||
      !std::all_of(r.x.begin(), r.x.end(), float_representable)) {
    TraceMessage("WINDOW_NUMERIC_REJECT stage=output reason=float_range");
    return invalid;
  }
  Result out;
  std::copy_n(r.x.data(), 15 * p.count, out.x.data());
  out.points.assign(r.x.begin() + 15 * p.count, r.x.end());
  out.before = r.before;
  out.after = r.after;
  out.iterations = r.iterations;
  TraceMessage("FP64_SOLVE gpu=%d before=%.12g after=%.12g iterations=%d rejected=%d failed=%d linear_residual=%.6g",
               int(gpu), r.before, r.after, r.iterations, r.rejected, r.failed, r.max_linear_residual);
  return out;
}
// Versioned, same-build diagnostic snapshots. No pointers or process state are serialized.
namespace {
struct Header {
  unsigned long long magic = 0x4650363457494e31ULL;
  int count, iterations;
  double huber;
};
template <class T>
void Write(std::ostream& s, const T& v) {
  s.write(reinterpret_cast<const char*>(&v), sizeof(v));
}
template <class T>
void Read(std::istream& s, T& v) {
  s.read(reinterpret_cast<char*>(&v), sizeof(v));
}
template <class T>
void WriteVec(std::ostream& s, const std::vector<T>& v) {
  size_t n = v.size();
  Write(s, n);
  s.write(reinterpret_cast<const char*>(v.data()), n * sizeof(T));
}
template <class T>
void ReadVec(std::istream& s, std::vector<T>& v) {
  size_t n;
  Read(s, n);
  if (!s || n > 1000000) {
    throw std::runtime_error("bad snapshot size");
  }
  v.resize(n);
  s.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
}
}  // namespace
void SaveProblem(const Problem& p, const char* path) {
  std::ofstream s(path, std::ios::binary);
  Header h;
  h.count = p.count;
  h.iterations = p.iterations;
  h.huber = p.huber;
  Write(s, h);
  Write(s, p.nodes);
  Write(s, p.anchors);
  Write(s, p.W);
  Write(s, p.offset);
  WriteVec(s, p.points);
  WriteVec(s, p.edges);
  WriteVec(s, p.visuals);
  if (!s) {
    throw std::runtime_error("cannot save window snapshot");
  }
}
Problem LoadProblem(const char* path) {
  std::ifstream s(path, std::ios::binary);
  Header h;
  Read(s, h);
  if (!s || h.magic != 0x4650363457494e31ULL || h.count < 2 || h.count > 6) {
    throw std::runtime_error("invalid snapshot");
  }
  Problem p;
  p.count = h.count;
  p.iterations = h.iterations;
  p.huber = h.huber;
  Read(s, p.nodes);
  Read(s, p.anchors);
  Read(s, p.W);
  Read(s, p.offset);
  ReadVec(s, p.points);
  ReadVec(s, p.edges);
  ReadVec(s, p.visuals);
  if (!s) {
    throw std::runtime_error("truncated snapshot");
  }
  return p;
}
}  // namespace gpu_window::fp64
