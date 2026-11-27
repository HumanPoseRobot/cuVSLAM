#include "history_archive.h"
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

namespace p = gpu_window::posterior;
namespace g = cuvslam::pnp::joint_geometry;
using M = Eigen::MatrixXd;
using V = Eigen::VectorXd;
void Require(bool condition, const char* text) {
  if (!condition) {
    throw std::runtime_error(text);
  }
}
p::Graph FromPrior(const p::Prior& prior) {
  p::Graph x;
  x.prior = prior;
  x.times = prior.times;
  x.nodes = prior.nodes;
  x.ids = prior.ids;
  x.points = prior.points;
  x.gravity = prior.gravity;
  x.fixed_pose_time = prior.fixed_pose_time;
  return x;
}
p::Prior Blueprint(const p::Graph& x) {
  p::Prior prior;
  prior.times = x.times;
  prior.nodes = x.nodes;
  prior.ids = x.ids;
  prior.points = x.points;
  prior.gravity = x.gravity;
  prior.fixed_pose_time = x.fixed_pose_time;
  return prior;
}
g::Factor Rows(const p::Graph& x) { return p::Assemble(x, p::Linearize(x, false)); }
struct Error {
  double information, gradient, energy;
};
Error Reference(const p::Graph& current, p::Graph reference) {
  for (size_t i = 0; i < current.times.size(); ++i) {
    const auto it = std::find(reference.times.begin(), reference.times.end(), current.times[i]);
    Require(it != reference.times.end(), "reference state missing");
    reference.nodes[it - reference.times.begin()] = current.nodes[i];
  }
  for (size_t i = 0; i < current.ids.size(); ++i) {
    const auto it = std::find(reference.ids.begin(), reference.ids.end(), current.ids[i]);
    Require(it != reference.ids.end(), "reference point missing");
    reference.points[it - reference.ids.begin()] = current.points[i];
  }
  reference.gravity = current.gravity;
  const auto all = Rows(reference), candidate = Rows(current);
  std::vector<int> kept;
  for (auto t : current.times) {
    const int node = std::find(reference.times.begin(), reference.times.end(), t) - reference.times.begin();
    for (int k = 0; k < 15; ++k) {
      kept.push_back(15 * node + k);
    }
  }
  for (auto id : current.ids) {
    const int point = std::find(reference.ids.begin(), reference.ids.end(), id) - reference.ids.begin();
    for (int k = 0; k < 3; ++k) {
      kept.push_back(15 * reference.nodes.size() + 3 * point + k);
    }
  }
  kept.push_back(all.rows.cols() - 2);
  kept.push_back(all.rows.cols() - 1);
  std::vector<bool> selected(all.rows.cols());
  for (int k : kept) {
    selected[k] = true;
  }
  M A(all.rows.rows(), kept.size()), B(all.rows.rows(), all.rows.cols() - kept.size());
  for (size_t i = 0; i < kept.size(); ++i) {
    A.col(i) = all.rows.col(kept[i]);
  }
  int at = 0;
  for (int i = 0; i < all.rows.cols(); ++i) {
    if (!selected[i]) {
      B.col(at) = all.rows.col(i);
      const double n = B.col(at).norm();
      if (n > 0.) {
        B.col(at) /= n;
      }
      ++at;
    }
  }
  const V unprojected_scale = A.colwise().norm();
  V r = all.residual;
  if (B.cols()) {
    Eigen::JacobiSVD<M> svd(B, Eigen::ComputeThinU);
    const double tolerance = std::numeric_limits<double>::epsilon() * std::max(B.rows(), B.cols()) *
                             (svd.singularValues().size() ? svd.singularValues()[0] : 0.);
    int rank = (svd.singularValues().array() > tolerance).count();
    const M U = svd.matrixU().leftCols(rank);
    A -= U * (U.transpose() * A).eval();
    r -= U * (U.transpose() * r).eval();
  }
  V scale = unprojected_scale;
  for (int k = 0; k < scale.size(); ++k) {
    scale[k] = scale[k] > 0. ? 1. / scale[k] : 1.;
  }
  M X = A * scale.asDiagonal(), Y = candidate.rows * scale.asDiagonal();
  M H = X.transpose() * X;
  return {(Y.transpose() * Y - H).norm() / std::max(1., H.norm()),
          (Y.transpose() * candidate.residual - X.transpose() * r).norm() / std::max(1., X.norm() * r.norm()),
          std::abs(candidate.residual.squaredNorm() - r.squaredNorm()) / std::max(1., r.squaredNorm())};
}
void Check(const p::Graph& current, const p::Graph& full, const char* label) {
  const auto e = Reference(current, full);
  std::cout << label << " H=" << e.information << " g=" << e.gradient << " E=" << e.energy << '\n';
  Require(e.information < 1e-9 && e.gradient < 1e-9 && e.energy < 1e-9, "full-factor SVD mismatch");
}
p::Graph Fixture(int deficient) {
  p::Graph x;
  for (int i = 0; i < 4; ++i) {
    x.times.push_back((i + 1) * 1000000000LL);
    gpu_window::State s{};
    s.R[0] = s.R[4] = s.R[8] = 1.;
    s.p[0] = .1 * i;
    x.nodes.push_back(s);
  }
  for (int i = 0; i < 6; ++i) {
    x.ids.push_back(72057594037937397LL + 2 * i);  // Intentionally beyond float64's exact integer range.
    x.points.emplace_back(1. + i, .2, 4.);
  }
  int n = 15 * x.nodes.size() + 3 * x.points.size() + 2;
  std::mt19937 generator(81399);
  std::normal_distribution<double> random;
  g::Factor f;
  f.rows.resize(n + 31, n);
  f.residual.resize(n + 31);
  for (int i = 0; i < f.rows.rows(); ++i) {
    f.residual[i] = .1 * random(generator);
    for (int j = 0; j < n; ++j) {
      f.rows(i, j) = random(generator) / std::sqrt(n);
    }
  }
  if (deficient == 1) {
    f.rows.col(15 * x.nodes.size() + 2).setZero();
  }
  if (deficient >= 2) {
    f.rows.middleCols(15 * x.nodes.size(), 3).setZero();
  }
  if (deficient == 3) {
    f.rows.middleCols(15 * x.nodes.size() + 3, 3).setZero();
    const int old = f.rows.rows();
    f.rows.conservativeResize(old + 3, n);
    f.rows.bottomRows(3).setZero();
    f.rows.block<3, 3>(old, 15 * x.nodes.size()) = M::Identity(3, 3);
    f.rows.block<3, 3>(old, 15 * x.nodes.size() + 3) = -M::Identity(3, 3);
    f.residual.conservativeResize(old + 3);
    f.residual.tail(3).setZero();
  }
  x.prior = p::StoreLinearizedPrior(Blueprint(x), std::move(f));
  return x;
}
void Retire(p::Graph& current, p::history::Archive& archive, const std::vector<int64_t>& times,
            const std::vector<int64_t>& ids) {
  const auto l = p::Linearize(current, false);
  archive.Capture(current, l, times, ids);
  current = FromPrior(p::Marginalize(current, l, times, ids, -1, -1));
}
void AddPoint(p::Graph& graph, int64_t id) {
  graph.ids.push_back(id);
  graph.points.emplace_back(1000., -500., 200.);  // Wrong legacy seed must not become fixed 3D evidence.
}
void Observe(p::Graph& current, p::Graph& full, int64_t id) {
  p::Graph observation;
  observation.times = {full.times.back()};
  observation.nodes = {full.nodes.back()};
  observation.ids = {id};
  const auto it = std::find(full.ids.begin(), full.ids.end(), id);
  observation.points = {full.points[it - full.ids.begin()]};
  observation.gravity = full.gravity;
  observation.fixed_pose_time = full.fixed_pose_time;
  g::Factor f;
  f.rows = M::Zero(3, 20);
  f.rows.block<3, 3>(0, 3) = -M::Identity(3, 3);
  f.rows.block<3, 3>(0, 15) = M::Identity(3, 3);
  f.residual = V(3);
  f.residual << .15, -.1, .05;
  const auto prior = p::StoreLinearizedPrior(Blueprint(observation), std::move(f));
  for (auto* target : {&current, &full}) {
    auto obs = *target;
    obs.prior = prior;
    const auto sum = g::combine({Rows(*target), Rows(obs)}, 15 * target->nodes.size() + 3 * target->points.size() + 2);
    target->prior = p::StoreLinearizedPrior(Blueprint(*target), sum);
  }
}
void Run(int deficient) {
  auto full = Fixture(deficient), current = full;
  const auto ids = full.ids;
  p::history::Archive archive;
  Retire(current, archive, {full.times[2], full.times[3]}, {ids[1], ids[2], ids[3], ids[4], ids[5]});
  Check(current, full, "first_retirement");
  Retire(current, archive, {full.times[3]}, {ids[4], ids[5]});
  Check(current, full, "second_retirement");
  auto no_history = current;
  auto negative_reference = full;
  AddPoint(no_history, ids[2]);
  Observe(no_history, negative_reference, ids[2]);
  const auto negative = Reference(no_history, negative_reference);
  Require(negative.information > 1e-5, "negative control did not detect omitted history");
  AddPoint(current, ids[0]);
  archive.Restore(current);
  Check(current, full, "restore_old_point");
  Observe(current, full, ids[0]);
  Check(current, full, "new_observation");
  Retire(current, archive, {full.times[3]}, {ids[4], ids[5]});
  Check(current, full, "retire_restored_point");
  AddPoint(current, ids[2]);
  archive.Restore(current);
  Check(current, full, "restore_other_point");
  AddPoint(current, ids[0]);
  archive.Restore(current);
  Check(current, full, "restore_with_new_history");
  const auto prior = current.prior;
  const auto before = archive.GetStats();
  archive.Restore(current);
  Require((current.prior.factor().rows - prior.factor().rows).norm() == 0., "repeat restore changed prior");
  Require(before.queries == archive.GetStats().queries, "repeat restore consumed history");
  auto trial = current;
  auto trial_archive = archive;
  AddPoint(trial, ids[1]);
  trial_archive.Restore(trial);
  Check(trial, full, "rejected_trial_reference");
  Require(archive.OwnsPoint(ids[1]), "rejected trial consumed committed history");
  Require(before.live_records == archive.GetStats().live_records, "rejected trial mutated records");
  Require((current.prior.factor().rows - prior.factor().rows).norm() == 0., "rejected trial wrote back prior");
  std::cout << "FIXTURE_PASS deficient=" << deficient << " records=" << archive.LiveRecords() << '\n';
}
int main() {
  try {
    Run(0);
    Run(1);
    Run(2);
    Run(3);
    std::cout << "ALL_HISTORY_ARCHIVE_TESTS_PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
