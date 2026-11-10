#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <limits>
#include <set>
#include <stdexcept>
#include "window_types.h"
namespace gpu_window {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
inline bool position_consistent(const Eigen::Matrix3d& covariance, const Eigen::Vector3d& delta, double& score) {
  score = std::numeric_limits<double>::infinity();
  if (!covariance.allFinite() || !delta.allFinite()) {
    return false;
  }
  Eigen::LDLT<Eigen::Matrix3d> raw((covariance + covariance.transpose()) * .5);
  if (raw.info() != Eigen::Success || !raw.isPositive() || raw.vectorD().minCoeff() < 0) {
    return false;
  }
  const Eigen::Matrix3d inflated = 4. * covariance + .000025 * Eigen::Matrix3d::Identity();
  Eigen::LDLT<Eigen::Matrix3d> factor(inflated);
  score = delta.dot(factor.solve(delta));
  return factor.info() == Eigen::Success && factor.isPositive() && std::isfinite(score) && score <= 16.27;
}

inline Matrix pinverse(const Matrix& in) {
  Matrix h = (in + in.transpose()) * .5;
  Vector scale = h.diagonal().cwiseMax(1e-12).cwiseSqrt().cwiseInverse();
  Matrix a = scale.asDiagonal() * h * scale.asDiagonal();
  Eigen::SelfAdjointEigenSolver<Matrix> es(a);
  if (es.info() != Eigen::Success) {
    throw std::runtime_error("window eigen failure");
  }
  Vector d = es.eigenvalues();
  double tol = std::max(1., d.maxCoeff()) * 1e-9;
  for (int i = 0; i < d.size(); ++i) {
    d[i] = d[i] > tol ? 1 / d[i] : 0;
  }
  return scale.asDiagonal() * es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose() * scale.asDiagonal();
}
// Preserve both H and the nonzero linear term, without injecting damping into the prior.
inline void square_root(const Matrix& h, const Vector& g, Matrix& W, Vector& c) {
  Vector scale = h.diagonal().cwiseMax(1e-12).cwiseSqrt().cwiseInverse();
  Matrix a = scale.asDiagonal() * ((h + h.transpose()) * .5) * scale.asDiagonal();
  Eigen::SelfAdjointEigenSolver<Matrix> es(a);
  if (es.info() != Eigen::Success) {
    throw std::runtime_error("window prior eigen failure");
  }
  Vector ev = es.eigenvalues();
  double tol = std::max(1., ev.maxCoeff()) * 1e-9;
  if (ev.minCoeff() < -1e-5 * std::max(1., ev.maxCoeff())) {
    throw std::runtime_error("indefinite window prior");
  }
  Vector root = ev, inv = ev;
  for (int i = 0; i < ev.size(); ++i) {
    root[i] = ev[i] > tol ? std::sqrt(ev[i]) : 0;
    inv[i] = root[i] > 0 ? 1 / root[i] : 0;
  }
  W = root.asDiagonal() * es.eigenvectors().transpose() * scale.cwiseInverse().asDiagonal();
  c = inv.asDiagonal() * es.eigenvectors().transpose() * scale.asDiagonal() * g;
}

// Rank-revealing orthogonal projection eliminates nuisance variables without
// subtracting nearly equal Hessian blocks. All subsequent information matrices
// are formed as A.transpose()*A in double precision.
inline void project_out(const Matrix& nuisance, const Matrix& kept, const Vector& residual, Matrix& projected,
                        Vector& projected_residual) {
  if (nuisance.cols() == 0 || nuisance.rows() == 0) {
    projected = kept;
    projected_residual = residual;
    return;
  }
  Matrix scaled = nuisance;
  for (int k = 0; k < scaled.cols(); ++k) {
    double norm = scaled.col(k).norm();
    if (norm > 1e-30) {
      scaled.col(k) /= norm;
    }
  }
  Eigen::ColPivHouseholderQR<Matrix> qr(scaled);
  qr.setThreshold(1e-10);
  const int rank = qr.rank();
  Matrix transformed = qr.householderQ().adjoint() * kept;
  Vector transformed_r = qr.householderQ().adjoint() * residual;
  projected = transformed.bottomRows(kept.rows() - rank);
  projected_residual = transformed_r.tail(kept.rows() - rank);
}
inline void compress_prior(const Matrix& rows, const Vector& residual, Matrix& W, Vector& c) {
  const int n = rows.cols();
  W = Matrix::Zero(n, n);
  c = Vector::Zero(n);
  if (rows.rows() == 0) {
    return;
  }
  Eigen::HouseholderQR<Matrix> qr(rows);
  const int retained = std::min<int>(rows.rows(), n);
  W.topRows(retained) = qr.matrixQR().topRows(retained).template triangularView<Eigen::Upper>();
  Vector transformed = qr.householderQ().adjoint() * residual;
  c.head(retained) = transformed.head(retained);
}
struct Reduced {
  Matrix H = Matrix::Zero(Dim, Dim);
  Vector g = Vector::Zero(Dim);
  Matrix rows;
  Vector residual;
};
inline Reduced assemble(const Problem& p, const Evaluation& e, bool marginal) {
  std::vector<Matrix> blocks;
  std::vector<Vector> residuals;
  Eigen::Map<const Eigen::Matrix<float, Dim, Dim, Eigen::RowMajor>> pj(e.prior_j.data());
  Eigen::Map<const Eigen::VectorXf> pr(e.prior.data(), Dim);
  blocks.push_back(pj.cast<double>());
  residuals.push_back(pr.cast<double>());
  for (size_t z = 0; z < p.edges.size(); ++z) {
    const auto& f = p.edges[z];
    if (marginal && f.left != 0 && f.right != 0) {
      continue;
    }
    Eigen::Map<const Eigen::Matrix<float, 15, 30, Eigen::RowMajor>> J(e.edge_j.data() + z * 450);
    Eigen::Map<const Eigen::Matrix<float, 15, 1>> r(e.edge.data() + z * 15);
    Matrix block = Matrix::Zero(15, Dim);
    block.block(0, f.left * 15, 15, 15) = J.leftCols(15).cast<double>();
    block.block(0, f.right * 15, 15, 15) = J.rightCols(15).cast<double>();
    blocks.push_back(block);
    residuals.push_back(r.cast<double>());
  }
  std::set<int> touched;
  for (auto& v : p.visuals) {
    if (v.node == 0) {
      touched.insert(v.point);
    }
  }
  std::vector<std::vector<int>> observations(p.points.size());
  for (size_t i = 0; i < p.visuals.size(); ++i) {
    auto& v = p.visuals[i];
    if (!marginal || touched.count(v.point)) {
      observations[v.point].push_back(i);
    }
  }
  for (auto& indices : observations) {
    if (indices.empty()) {
      continue;
    }
    Matrix point = Matrix::Zero(2 * indices.size(), 3);
    Matrix pose = Matrix::Zero(2 * indices.size(), Dim);
    Vector residual(2 * indices.size());
    for (size_t k = 0; k < indices.size(); ++k) {
      int i = indices[k];
      const auto& v = p.visuals[i];
      Eigen::Map<const Eigen::Matrix<float, 2, 18, Eigen::RowMajor>> J(e.visual_j.data() + i * 36);
      Eigen::Map<const Eigen::Vector2f> r(e.visual.data() + i * 2);
      double rn = r.cast<double>().norm();
      double weight = rn > p.huber ? std::sqrt(p.huber / rn) : 1.;
      pose.block(2 * k, 15 * v.node, 2, 15) = weight * J.leftCols(15).cast<double>();
      point.block(2 * k, 0, 2, 3) = weight * J.rightCols(3).cast<double>();
      residual.segment(2 * k, 2) = weight * r.cast<double>();
    }
    Matrix reduced;
    Vector rr;
    project_out(point, pose, residual, reduced, rr);
    if (reduced.rows()) {
      blocks.push_back(reduced);
      residuals.push_back(rr);
    }
  }
  Reduced out;
  int total = 0;
  for (auto& b : blocks) {
    total += b.rows();
  }
  out.rows.resize(total, Dim);
  out.residual.resize(total);
  int offset = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    out.rows.middleRows(offset, blocks[i].rows()) = blocks[i];
    out.residual.segment(offset, blocks[i].rows()) = residuals[i];
    offset += blocks[i].rows();
  }
  out.H = out.rows.transpose() * out.rows;
  out.g = out.rows.transpose() * out.residual;
  return out;
}
inline Reduced eliminate_first(const Reduced& r, int count) {
  int n = 15 * (count - 1);
  Reduced out;
  Matrix projected;
  Vector rr;
  project_out(r.rows.leftCols(15), r.rows.middleCols(15, n), r.residual, projected, rr);
  out.rows = Matrix::Zero(projected.rows(), Dim);
  out.rows.leftCols(n) = projected;
  out.residual = rr;
  out.H = out.rows.transpose() * out.rows;
  out.g = out.rows.transpose() * rr;
  return out;
}
}  // namespace gpu_window
