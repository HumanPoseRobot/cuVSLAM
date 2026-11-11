#include <cassert>
#include <iostream>
#include "window_fp64.h"
#include "window_linear.h"
#include "window_math.h"
#include "window_validation.h"
using namespace gpu_window;
int main() {
  {
    assert(!float_representable(1e40));
    assert(!float_representable(std::numeric_limits<double>::infinity()));
    assert(float_representable(std::numeric_limits<float>::max()));
    gpu_window::Evaluation bad;
    bad.prior_j.push_back(std::numeric_limits<float>::quiet_NaN());
    assert(!valid_evaluation(bad));
    Problem p;
    p.count = 2;
    assert(invalid_problem(p) == nullptr);
    p.nodes[0].R[0] = -1.;
    assert(invalid_problem(p) != nullptr);
    for (bool gpu : {false, true}) {
      auto r = fp64::Optimize(p, gpu);
      assert(!valid_result(r, 0));
    }
    p.nodes[0] = State{};
    p.offset[0] = 1e30;
    for (bool gpu : {false, true}) {
      auto r = fp64::Optimize(p, gpu);
      assert(!valid_result(r, 0));
    }
    p.offset[0] = 0;
    p.count = Slots + 1;
    assert(invalid_problem(p) != nullptr);
    p.count = 2;
    Visual v;
    v.node = -1;
    p.visuals.push_back(v);
    assert(invalid_problem(p) != nullptr);
    for (double angle : {0., 1e-8, 3.14017, 3.14018, 3.141592653589793}) {
      double x[3] = {angle, 0, 0}, R[9], back[3], reconstructed[9];
      gpu_vi::exp3(x, R);
      assert(valid_rotation(R));
      gpu_vi::log3(R, back);
      gpu_vi::exp3(back, reconstructed);
      for (int k = 0; k < 9; ++k) {
        assert(std::abs(R[k] - reconstructed[k]) < 2e-6);
      }
    }
    std::cout << "Numeric range, invalid rotation/index and rotation boundary tests passed\n";
  }

  // Log(Exp(x)) must have identity derivative inside the principal branch,
  // including both sides of the near-pi implementation switch.
  for (double angle : {0., 1., 3.14017, 3.14018, 3.141592643589793}) {
    for (int col = 0; col < 3; ++col) {
      gpu_vi::Jet x[3], R[9], back[3];
      for (int k = 0; k < 3; ++k) {
        x[k] = {angle * (k + 1) / std::sqrt(14.), k == col ? 1. : 0.};
      }
      gpu_vi::exp3(x, R);
      gpu_vi::log3(R, back);
      for (int k = 0; k < 3; ++k) {
        assert(std::abs(back[k].v - x[k].v) < 1e-10);
        assert(std::abs(back[k].d - (k == col ? 1. : 0.)) < 1e-6);
      }
    }
  }
  std::cout << "Near-pi rotation value and Jacobian continuity passed\n";
  // A purely quadratic velocity prior has an exact model reduction. This
  // catches a missing diagonal weight or factor of 1/2 in the LM gain ratio.
  {
    Problem q;
    q.count = 2;
    q.iterations = 20;
    for (int k = 0; k < 30; ++k) {
      q.W[k * Dim + k] = (k == 6 ? 100. : 1.);
    }
    q.offset[6] = 25.;
    for (bool gpu : {false, true}) {
      auto b = gpu ? fp64::MakeGpu(q) : fp64::MakeCpu(q);
      std::vector<double> x(30);
      double initial = b->Evaluate(x, true);
      auto step = b->Solve(1e-3);
      assert(step.valid);
      assert(std::abs(step.dx[6] + .25 / 1.001) < 1e-12);
      double trial = b->Evaluate(step.dx, false);
      assert(std::abs(step.predicted - (initial - trial)) < 1e-10);
    }
    std::cout << "FP64 quadratic predicted reduction and zero-landmark solve passed\n";
  }
  Problem p;
  p.count = 6;
  p.iterations = 20;
  p.huber = 2.;
  for (int i = 0; i < Slots; ++i) {
    p.nodes[i].p[0] = .03 * i;
    p.anchors[i] = p.nodes[i];
  }
  for (int k = 0; k < 15; ++k) {
    p.W[k * Dim + k] = 10;
  }
  for (int i = 0; i < 5; ++i) {
    Edge e;
    e.left = i;
    e.right = i + 1;
    e.data.dt = .1;
    e.data.gravity[2] = -9.81;
    for (int k = 0; k < 9; ++k) {
      e.data.R[0][k] = p.nodes[i].R[k];
      e.data.R[1][k] = p.nodes[i + 1].R[k];
      e.data.dR[k] = k % 4 == 0 ? 1 : 0;
      e.data.imu_sqrt[k * 9 + k] = 10;
    }
    for (int k = 0; k < 3; ++k) {
      e.data.p[0][k] = p.nodes[i].p[k];
      e.data.p[1][k] = p.nodes[i + 1].p[k];
    }
    e.data.dP[0] = .03;
    e.data.dV[2] = .981;
    e.data.dP[2] = .04905;
    for (int k = 0; k < 6; ++k) {
      e.rw[6 * k + k] = 100;
    }
    p.edges.push_back(e);
  }
  for (int j = 0; j < 20; ++j) {
    p.points.push_back({.15 * (j % 5 - 2), .12 * (j / 5 - 2), 2. + .1 * j});
    for (int i = 0; i < 6; ++i) {
      for (int c = 0; c < 2; ++c) {
        Visual v;
        v.node = i;
        v.point = j;
        v.obs.Rci[0] = v.obs.Rci[4] = v.obs.Rci[8] = 1;
        v.obs.tci[0] = -.1 * c;
        v.obs.xy[0] = (p.points[j][0] - p.nodes[i].p[0] - .1 * c) / p.points[j][2];
        v.obs.xy[1] = p.points[j][1] / p.points[j][2];
        v.obs.sqrt_info[0] = v.obs.sqrt_info[3] = 100;
        v.obs.min_depth = .05;
        p.visuals.push_back(v);
      }
    }
  }
  gpu_window::Result z;
  z.points.resize(3 * p.points.size());
  auto e = Evaluate(p, z);
  for (auto v : e.edge) {
    assert(std::abs(v) < 1e-6);
  }
  for (auto v : e.visual) {
    assert(std::abs(v) < 1e-6);
  }
  for (auto v : e.prior) {
    assert(std::abs(v) < 1e-6);
  }
  for (auto& edge : p.edges) {
    for (int k = 0; k < 3; ++k) {
      edge.data.JRg[k * 4] = .1;
      edge.data.JVg[k * 4] = .05;
      edge.data.JVa[k * 4] = .1;
      edge.data.JPg[k * 4] = .003;
      edge.data.JPa[k * 4] = .005;
    }
  }
  // Independent central differences at a nonzero tangent, including both endpoint biases.
  for (int k = 0; k < Dim; ++k) {
    z.x[k] = ((k % 5) - 2) * .0002;
  }
  for (int k = 0; k < int(z.points.size()); ++k) {
    z.points[k] = ((k % 3) - 1) * .0003;
  }
  e = Evaluate(p, z);
  double worst = 0;
  for (int c = 0; c < 30; ++c) {
    gpu_window::Result a = z, b = z;
    float h = 1e-4f;
    a.x[c] += h;
    b.x[c] -= h;
    auto ea = Evaluate(p, a), eb = Evaluate(p, b);
    for (int r = 0; r < 15; ++r) {
      double num = (ea.edge[r] - eb.edge[r]) / (2 * h), ad = e.edge_j[r * 30 + c];
      worst = std::max(worst, std::abs(num - ad) / std::max(1., std::abs(ad)));
    }
  }
  for (int c = 0; c < 18; ++c) {
    gpu_window::Result a = z, b = z;
    float h = 1e-4f;
    if (c < 15) {
      a.x[c] += h;
      b.x[c] -= h;
    } else {
      a.points[c - 15] += h;
      b.points[c - 15] -= h;
    }
    auto ea = Evaluate(p, a), eb = Evaluate(p, b);
    for (int r = 0; r < 2; ++r) {
      double num = (ea.visual[r] - eb.visual[r]) / (2 * h), ad = e.visual_j[r * 18 + c];
      worst = std::max(worst, std::abs(num - ad) / std::max(1., std::abs(ad)));
    }
  }
  {
    Problem q = p;
    for (int s = 0; s < Slots; ++s) {
      double angle[3] = {.08, -.04, .12};
      gpu_vi::exp3(angle, q.anchors[s].R);
      q.anchors[s].p[1] += .2;
    }
    for (int i = 0; i < Dim; ++i) {
      for (int j = 0; j < Dim; ++j) {
        q.W[i * Dim + j] = (i == j ? 2. : .001 * ((i + j) % 5 - 2));
      }
    }
    auto qe = Evaluate(q, z);
    for (int c : {0, 2, 3, 8, 9, 14, 30, 33, 74, 89}) {
      auto a = z, b = z;
      float h = 1e-4;
      a.x[c] += h;
      b.x[c] -= h;
      auto ea = Evaluate(q, a), eb = Evaluate(q, b);
      for (int r = 0; r < Dim; ++r) {
        double num = (ea.prior[r] - eb.prior[r]) / (2 * h), ad = qe.prior_j[r * Dim + c];
        worst = std::max(worst, std::abs(num - ad) / std::max(1., std::abs(ad)));
      }
    }
  }
  assert(worst < 2e-3);
  std::cout << "independent finite-difference Jacobian worst=" << worst << "\n";
  p.nodes[3].p[1] += .015;
  for (auto& ed : p.edges) {
    for (int k = 0; k < 3; ++k) {
      ed.data.p[0][k] = p.nodes[ed.left].p[k];
      ed.data.p[1][k] = p.nodes[ed.right].p[k];
    }
  }
  auto sol = Optimize(p);
  auto cpu64_backend = fp64::MakeCpu(p);
  auto gpu64_backend = fp64::MakeGpu(p);
  std::vector<double> initial64(15 * p.count + 3 * p.points.size());
  const double cpu_cost = cpu64_backend->Evaluate(initial64, true);
  const double gpu_cost = gpu64_backend->Evaluate(initial64, true);
  assert(std::abs(cpu_cost - gpu_cost) < 1e-9 * std::max(1., cpu_cost));
  auto cpu_step = cpu64_backend->Solve(1e-3);
  auto gpu_step = gpu64_backend->Solve(1e-3);
  assert(cpu_step.valid && gpu_step.valid);
  for (size_t i = 0; i < cpu_step.dx.size(); ++i) {
    assert(std::abs(cpu_step.dx[i] - gpu_step.dx[i]) < 1e-7);
  }
  auto cpu64 = fp64::Minimize(p, *cpu64_backend);
  auto gpu64 = fp64::Minimize(p, *gpu64_backend);
  assert(std::abs(cpu64.after - gpu64.after) < 1e-8 * std::max(1., cpu64.after));
  for (size_t i = 0; i < cpu64.x.size(); ++i) {
    assert(std::abs(cpu64.x[i] - gpu64.x[i]) < 1e-6);
  }
  std::cout << "FP64 CPU/GPU normal solve and shared LM agree, residual=" << gpu_step.relative_residual << "\n";
  auto schur = OptimizeSchur(p);
  auto cpu = OptimizeCpu(p);
  assert(std::abs(schur.before - cpu.before) <= 1e-5 * std::max(1.f, cpu.before));
  assert(std::abs(schur.after - cpu.after) <= 1e-3 * std::max(1.f, cpu.after));
  for (int i = 0; i < 15 * p.count; ++i) {
    assert(std::abs(schur.x[i] - cpu.x[i]) < 1e-2);
  }
  assert(std::isfinite(sol.after) && sol.after < sol.before * .01);
  std::cout << "six-state GPU solve: " << sol.before << " -> " << sol.after << "\n";
  double score;
  Eigen::Matrix3d cov = Eigen::Matrix3d::Identity() * 1e-6;
  assert(position_consistent(cov, Eigen::Vector3d(.005, 0, 0), score));
  assert(!position_consistent(cov, Eigen::Vector3d(.4, 0, 0), score));
  cov(0, 0) = -1;
  assert(!position_consistent(cov, Eigen::Vector3d::Zero(), score));
  cov(0, 0) = std::numeric_limits<double>::quiet_NaN();
  assert(!position_consistent(cov, Eigen::Vector3d::Zero(), score));
  std::cout
      << "backend consistency gate accepts small updates and rejects invalid covariance / inconsistent corrections\n";
  // Marginalization must preserve H AND its linear term, not merely covariance.
  Matrix A = Matrix::Random(70, 30);
  Matrix H = A.transpose() * A + Matrix::Identity(30, 30);
  Vector g = Vector::Random(30);
  Matrix inv = pinverse(H.topLeftCorner(15, 15));
  Matrix B = H.topRightCorner(15, 15);
  Matrix h = H.bottomRightCorner(15, 15) - B.transpose() * inv * B;
  Vector grad = g.tail(15) - B.transpose() * inv * g.head(15);
  Matrix W;
  Vector off;
  square_root(h, grad, W, off);
  assert((W.transpose() * W - h).norm() / h.norm() < 1e-9);
  assert((W.transpose() * off - grad).norm() < 1e-8);
  Vector full = -H.ldlt().solve(g);
  Vector kept = -h.ldlt().solve(grad);
  assert((full.tail(15) - kept).norm() < 1e-9);
  // Rank-deficient PSD blocks are eliminated without artificial damping.
  Matrix rank = Matrix::Zero(3, 3);
  rank(0, 0) = 2;
  rank(1, 1) = 3;
  assert((rank * pinverse(rank) * rank - rank).norm() < 1e-10);
  std::cout << "Schur linear term, square-root prior, and gauge rank tests passed\n";
  // Cross-check sparse point elimination against an independent dense normal matrix.
  auto ev = Evaluate(p, z);
  int width = Dim + 3 * p.points.size();
  Matrix dense = Matrix::Zero(width, width);
  Vector rhs = Vector::Zero(width);
  auto add = [&](const Vector& r, const Matrix& j, double weight) {
    dense += weight * j.transpose() * j;
    rhs += weight * j.transpose() * r;
  };
  {
    Matrix j = Matrix::Zero(Dim, width);
    for (int i = 0; i < Dim; ++i) {
      for (int k = 0; k < Dim; ++k) {
        j(i, k) = ev.prior_j[i * Dim + k];
      }
    }
    Vector r(Dim);
    for (int i = 0; i < Dim; ++i) {
      r[i] = ev.prior[i];
    }
    add(r, j, 1);
  }
  for (size_t i = 0; i < p.edges.size(); ++i) {
    Matrix j = Matrix::Zero(15, width);
    Vector r(15);
    for (int k = 0; k < 15; ++k) {
      r[k] = ev.edge[i * 15 + k];
      for (int c = 0; c < 30; ++c) {
        int col = (c < 15 ? p.edges[i].left : p.edges[i].right) * 15 + c % 15;
        j(k, col) = ev.edge_j[i * 450 + k * 30 + c];
      }
    }
    add(r, j, 1);
  }
  for (size_t i = 0; i < p.visuals.size(); ++i) {
    Matrix j = Matrix::Zero(2, width);
    Vector r(2);
    auto& v = p.visuals[i];
    for (int k = 0; k < 2; ++k) {
      r[k] = ev.visual[i * 2 + k];
      for (int c = 0; c < 18; ++c) {
        int col = c < 15 ? v.node * 15 + c : Dim + v.point * 3 + c - 15;
        j(k, col) = ev.visual_j[i * 36 + k * 18 + c];
      }
    }
    add(r, j, r.norm() > p.huber ? p.huber / r.norm() : 1.);
  }
  auto reduced = assemble(p, ev, false);
  Matrix inverse = pinverse(dense.bottomRightCorner(width - Dim, width - Dim));
  Matrix cross = dense.topRightCorner(Dim, width - Dim);
  Matrix expected = dense.topLeftCorner(Dim, Dim) - cross * inverse * cross.transpose();
  Vector expected_g = rhs.head(Dim) - cross * inverse * rhs.tail(width - Dim);
  assert((expected - reduced.H).norm() / expected.norm() < 1e-8);
  assert((expected_g - reduced.g).norm() < 1e-7);
  std::cout << "GPU Jacobian sparse assembly matches independent dense Schur\n";
  // Square-root compression preserves both curvature and the nonzero linear term.
  Matrix root;
  Vector offset;
  compress_prior(reduced.rows, reduced.residual, root, offset);
  assert((root.transpose() * root - reduced.H).norm() / reduced.H.norm() < 1e-10);
  assert((root.transpose() * offset - reduced.g).norm() / std::max(1., reduced.g.norm()) < 1e-10);
  // Nearly dependent nuisance columns across different units must not make the
  // marginalized information indefinite through cancellation.
  Matrix weak = Matrix::Random(90, 20);
  weak.col(1) = weak.col(0) + 1e-11 * weak.col(1);
  weak.col(3) *= 1e9;
  weak.col(4) *= 1e-9;
  Matrix keep = Matrix::Random(90, 15);
  Vector residual = Vector::Random(90);
  Matrix projected;
  Vector rr;
  project_out(weak, keep, residual, projected, rr);
  Matrix safe = projected.transpose() * projected;
  Eigen::SelfAdjointEigenSolver<Matrix> eig(safe);
  assert(eig.eigenvalues().minCoeff() > -1e-10);
  compress_prior(projected, rr, root, offset);
  assert((root.transpose() * root - safe).norm() / safe.norm() < 1e-10);
  std::cout << "square-root projection/compression and ill-conditioned marginalization passed\n";
}
