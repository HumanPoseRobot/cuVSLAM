#pragma once
#include <dlfcn.h>
#include <Eigen/Dense>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
#include "joint_gpu_dense.h"
#include "pipelines/gpu_window/window_linear.h"

namespace cuvslam::pnp::joint_geometry::dense {
// Optional laboratory acceleration. The reference model and retained variables
// do not change. No system BLAS or released runtime is modified.
struct Blas {
  using Gemm = void (*)(const char*, const char*, const int*, const int*, const int*, const double*, const double*,
                        const int*, const double*, const int*, const double*, double*, const int*);
  using Geqp3 = void (*)(const int*, const int*, double*, const int*, int*, double*, double*, const int*, int*);
  using Geqrf = void (*)(const int*, const int*, double*, const int*, double*, double*, const int*, int*);
  using Ormqr = void (*)(const char*, const char*, const int*, const int*, const int*, const double*, const int*,
                         const double*, double*, const int*, double*, const int*, int*);
  using Potrf = void (*)(const char*, const int*, double*, const int*, int*);
  using Potrs = void (*)(const char*, const int*, const int*, const double*, const int*, double*, const int*, int*);
  void* handle = nullptr;
  Gemm gemm = nullptr;
  Geqp3 geqp3 = nullptr;
  Geqrf geqrf = nullptr;
  Ormqr ormqr = nullptr;
  Potrf potrf = nullptr;
  Potrs potrs = nullptr;
  template <class T>
  T symbol(const char* name) {
    auto value = reinterpret_cast<T>(dlsym(handle, name));
    if (!value) {
      throw std::runtime_error("missing laboratory LP64 BLAS/LAPACK symbol");
    }
    return value;
  }
  Blas() {
    const char* library = std::getenv("VIO_JOINT_BLAS_LIBRARY");
    if (!library || !*library) {
      return;
    }
    handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      throw std::runtime_error("cannot load laboratory LP64 BLAS/LAPACK");
    }
    gemm = symbol<Gemm>("dgemm_");
    geqp3 = symbol<Geqp3>("dgeqp3_");
    geqrf = symbol<Geqrf>("dgeqrf_");
    ormqr = symbol<Ormqr>("dormqr_");
    potrf = symbol<Potrf>("dpotrf_");
    potrs = symbol<Potrs>("dpotrs_");
    const char* count = std::getenv("VIO_JOINT_BLAS_THREADS");
    const int threads = count ? std::atoi(count) : 1;
    if (threads < 1 || threads > 64) {
      throw std::invalid_argument("laboratory BLAS thread count");
    }
    symbol<void (*)(int)>("openblas_set_num_threads")(threads);
  }
};
inline Blas& blas() {
  static Blas value;
  return value;
}
inline int size(Eigen::Index n) {
  if (n < 0 || n > std::numeric_limits<int>::max()) {
    throw std::overflow_error("LP64 matrix dimension overflow");
  }
  return static_cast<int>(n);
}
inline int workspace(double queried, int minimum) {
  if (!std::isfinite(queried) || queried > std::numeric_limits<int>::max()) {
    throw std::overflow_error("LAPACK workspace overflow");
  }
  return std::max(minimum, static_cast<int>(queried));
}
inline Eigen::MatrixXd multiply(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  if (a.cols() != b.rows()) {
    throw std::invalid_argument("joint product shape");
  }
  if (!blas().handle || !a.size() || !b.size()) {
    return a * b;
  }
  const int m = size(a.rows()), n = size(b.cols()), k = size(a.cols());
  Eigen::MatrixXd result(m, n);
  const char normal = 'N';
  const double one = 1., zero = 0.;
  blas().gemm(&normal, &normal, &m, &n, &k, &one, a.data(), &m, b.data(), &k, &zero, result.data(), &m);
  return result;
}
inline Eigen::MatrixXd gram(const Eigen::MatrixXd& a) {
  if (std::getenv("VIO_JOINT_GPU_GRAM") && a.size()) {
    Eigen::MatrixXd result;
    if (GpuGram(a, result)) {
      return result;
    }
  }

  if (!blas().handle || !a.size()) {
    return a.transpose() * a;
  }
  const int n = size(a.cols()), k = size(a.rows());
  Eigen::MatrixXd result(n, n);
  const char transpose = 'T', normal = 'N';
  const double one = 1., zero = 0.;
  blas().gemm(&transpose, &normal, &n, &n, &k, &one, a.data(), &k, a.data(), &k, &zero, result.data(), &n);
  return result;
}
inline void apply_qt(const Eigen::MatrixXd& qr, const Eigen::VectorXd& tau, Eigen::MatrixXd& rhs) {
  const int m = size(rhs.rows()), n = size(rhs.cols()), k = size(tau.size());
  if (!m || !n || !k) {
    return;
  }
  const char left = 'L', transpose = 'T';
  int length = -1, status = 0;
  double query = 0.;
  blas().ormqr(&left, &transpose, &m, &n, &k, qr.data(), &m, tau.data(), rhs.data(), &m, &query, &length, &status);
  if (status) {
    throw std::runtime_error("LAPACK Q query failed");
  }
  length = workspace(query, n);
  std::vector<double> work(length);
  blas().ormqr(&left, &transpose, &m, &n, &k, qr.data(), &m, tau.data(), rhs.data(), &m, work.data(), &length, &status);
  if (status || !rhs.allFinite()) {
    throw std::runtime_error("LAPACK orthogonal transform failed");
  }
}
inline bool CertifyFullRank(const Eigen::MatrixXd& qr, int original_rows = 0) {
  const int m = size(qr.rows()), n = size(qr.cols());
  if (!n || n > m || !qr.allFinite()) {
    return false;
  }
  using Trcon = void (*)(const char*, const char*, const char*, const int*, const double*, const int*, double*, double*,
                         int*, int*);
  const auto condition = reinterpret_cast<Trcon>(dlsym(blas().handle, "dtrcon_"));
  if (!condition) {
    return false;
  }
  const char norm = '1', upper = 'U', nonunit = 'N';
  int status = 0;
  double reciprocal = 0.;
  std::vector<double> scratch(3 * n);
  std::vector<int> integer(n);
  condition(&norm, &upper, &nonunit, &n, qr.data(), &m, &reciprocal, scratch.data(), integer.data(), &status);
  const auto diagonal = qr.topRows(n).diagonal().cwiseAbs().eval();
  const double roundoff = 256. * std::numeric_limits<double>::epsilon() * std::max(std::max(m, n), original_rows);
  // Match the original rank cutoff AND require an independent reciprocal-
  // condition estimate above the matrix-size-dependent roundoff budget.
  if (status || !std::isfinite(reciprocal) || reciprocal <= roundoff ||
      diagonal.minCoeff() <= 1e-10 * diagonal.maxCoeff()) {
    return false;
  }
  return true;
}
// The nuisance columns may include exactly-zero gauge columns. For a
// certified full-rank active column space, unpivoted blocked QR spans the
// same space as pivoted QR. Rank/conditioning uncertainty falls back to the
// original rank-revealing implementation; no observation or prior is dropped.
inline bool project_full_rank(const Eigen::MatrixXd& normalized, const Eigen::MatrixXd& kept, const Eigen::VectorXd& r,
                              Eigen::MatrixXd& rows, Eigen::VectorXd& residual) {
  std::vector<int> active;
  for (int i = 0; i < normalized.cols(); ++i) {
    if (normalized.col(i).squaredNorm() > 0.) {
      active.push_back(i);
    }
  }
  if (active.empty()) {
    rows = kept;
    residual = r;
    return true;
  }
  if (active.size() > size_t(normalized.rows())) {
    return false;
  }
  const int m = size(normalized.rows()), n = size(active.size());
  Eigen::MatrixXd qr(m, n);
  for (int i = 0; i < n; ++i) {
    qr.col(i) = normalized.col(active[i]);
  }
  if (std::getenv("VIO_JOINT_GPU_QR")) {
    return GpuProjectFullRank(qr, kept, r, rows, residual);
  }
  Eigen::VectorXd tau(n);
  int length = -1, status = 0;
  double query = 0.;
  blas().geqrf(&m, &n, qr.data(), &m, tau.data(), &query, &length, &status);
  if (status) {
    return false;
  }
  length = workspace(query, n);
  std::vector<double> work(length);
  blas().geqrf(&m, &n, qr.data(), &m, tau.data(), work.data(), &length, &status);
  if (status || !qr.allFinite()) {
    return false;
  }
  if (!CertifyFullRank(qr)) {
    return false;
  }
  Eigen::MatrixXd transformed(m, kept.cols() + 1);
  transformed.leftCols(kept.cols()) = kept;
  transformed.rightCols(1) = r;
  apply_qt(qr, tau, transformed);
  rows = transformed.bottomRows(m - n).leftCols(kept.cols());
  residual = transformed.bottomRows(m - n).rightCols(1);
  return true;
}
inline void project(const Eigen::MatrixXd& nuisance, const Eigen::MatrixXd& kept, const Eigen::VectorXd& r,
                    Eigen::MatrixXd& rows, Eigen::VectorXd& residual) {
  if (!blas().handle || !nuisance.size()) {
    gpu_window::project_out(nuisance, kept, r, rows, residual);
    return;
  }
  Eigen::MatrixXd qr = nuisance;
  for (int col = 0; col < qr.cols(); ++col) {
    const double norm = qr.col(col).norm();
    if (norm > 1e-30) {
      qr.col(col) /= norm;
    }
  }
  if (std::getenv("VIO_JOINT_FAST_QR") && project_full_rank(qr, kept, r, rows, residual)) {
    return;
  }
  const int m = size(qr.rows()), n = size(qr.cols()), count = std::min(m, n);
  std::vector<int> permutation(n, 0);
  Eigen::VectorXd tau(count);
  int length = -1, status = 0;
  double query = 0.;
  blas().geqp3(&m, &n, qr.data(), &m, permutation.data(), tau.data(), &query, &length, &status);
  if (status) {
    throw std::runtime_error("LAPACK pivot QR query failed");
  }
  length = workspace(query, 3 * n + 1);
  std::vector<double> work(length);
  blas().geqp3(&m, &n, qr.data(), &m, permutation.data(), tau.data(), work.data(), &length, &status);
  if (status || !qr.allFinite()) {
    throw std::runtime_error("LAPACK pivot QR failed");
  }
  const auto diagonal = qr.topLeftCorner(count, count).diagonal().cwiseAbs().eval();
  const int rank = (diagonal.array() > 1e-10 * diagonal.maxCoeff()).count();
  Eigen::MatrixXd transformed(m, kept.cols() + 1);
  transformed.leftCols(kept.cols()) = kept;
  transformed.rightCols(1) = r;
  apply_qt(qr, tau, transformed);
  rows = transformed.bottomRows(m - rank).leftCols(kept.cols());
  residual = transformed.bottomRows(m - rank).rightCols(1);
}
inline void compress(const Eigen::MatrixXd& rows, const Eigen::VectorXd& r, Eigen::MatrixXd& w,
                     Eigen::VectorXd& residual) {
  if (std::getenv("VIO_JOINT_GPU_QR") && rows.size() && GpuCompress(rows, r, w, residual)) {
    return;
  }

  if (!blas().handle || !rows.size()) {
    gpu_window::compress_prior(rows, r, w, residual);
    return;
  }
  Eigen::MatrixXd qr = rows;
  const int m = size(qr.rows()), n = size(qr.cols()), count = std::min(m, n);
  Eigen::VectorXd tau(count);
  int length = -1, status = 0;
  double query = 0.;
  blas().geqrf(&m, &n, qr.data(), &m, tau.data(), &query, &length, &status);
  if (status) {
    throw std::runtime_error("LAPACK QR query failed");
  }
  length = workspace(query, n);
  std::vector<double> work(length);
  blas().geqrf(&m, &n, qr.data(), &m, tau.data(), work.data(), &length, &status);
  if (status || !qr.allFinite()) {
    throw std::runtime_error("LAPACK QR failed");
  }
  Eigen::MatrixXd transformed = r;
  apply_qt(qr, tau, transformed);
  w = Eigen::MatrixXd::Zero(n, n);
  w.topRows(count) = qr.topRows(count).triangularView<Eigen::Upper>();
  residual = Eigen::VectorXd::Zero(n);
  residual.head(count) = transformed.col(0).head(count);
}
inline bool solve_spd(const Eigen::MatrixXd& a, const Eigen::VectorXd& rhs, Eigen::VectorXd& result) {
  if (std::getenv("VIO_JOINT_GPU_SPD") && GpuSolveSpd(a, rhs, result)) {
    return true;
  }

  if (!blas().handle) {
    Eigen::LDLT<Eigen::MatrixXd> factor(a);
    if (factor.info() != Eigen::Success || !factor.isPositive()) {
      return false;
    }
    result = factor.solve(rhs);
    return result.allFinite();
  }
  Eigen::MatrixXd factor = a;
  const int n = size(a.rows()), one = 1;
  const char lower = 'L';
  int status = 0;
  blas().potrf(&lower, &n, factor.data(), &n, &status);
  if (status) {
    return false;
  }
  result = rhs;
  blas().potrs(&lower, &n, &one, factor.data(), &n, result.data(), &n, &status);
  return status == 0 && result.allFinite();
}
}  // namespace cuvslam::pnp::joint_geometry::dense
