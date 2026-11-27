#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include "history_archive.h"
#include "joint_online.h"

namespace p = gpu_window::posterior;
namespace g = cuvslam::pnp::joint_geometry;
using M = Eigen::MatrixXd;
using V = Eigen::VectorXd;
M Pull(const p::Graph& full, const p::Prior& prior, const M& probe) {
  M out(15 * prior.times.size() + 3 * prior.ids.size() + 2, probe.cols());
  int row = 0;
  for (auto t : prior.times) {
    const auto it = std::find(full.times.begin(), full.times.end(), t);
    if (it == full.times.end()) {
      throw std::runtime_error("probe source state missing");
    }
    out.middleRows(row, 15) = probe.middleRows(15 * (it - full.times.begin()), 15);
    row += 15;
  }
  for (auto id : prior.ids) {
    const auto it = std::find(full.ids.begin(), full.ids.end(), id);
    if (it == full.ids.end()) {
      throw std::runtime_error("probe source point missing");
    }
    out.middleRows(row, 3) = probe.middleRows(15 * full.nodes.size() + 3 * (it - full.ids.begin()), 3);
    row += 3;
  }
  out.bottomRows(2) = probe.bottomRows(2);
  return out;
}
g::Factor AtAnchor(const p::Prior& prior) {
  p::Graph x;
  x.prior = prior;
  x.times = prior.times;
  x.nodes = prior.nodes;
  x.ids = prior.ids;
  x.points = prior.points;
  x.gravity = prior.gravity;
  x.fixed_pose_time = prior.fixed_pose_time;
  return p::Assemble(x, p::Linearize(x, false));
}
int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      return 2;
    }
    const auto full = p::LoadGraph(argv[1]);
    auto live = p::LoadGraph(argv[2]);
    if (!full.prior.empty()) {
      throw std::runtime_error("initial full factor graph required");
    }
    const auto start = std::chrono::steady_clock::now();
    const auto l = p::Linearize(full, false);
    p::history::Archive archive;
    archive.Capture(full, l, live.times, live.ids);
    const auto capture_end = std::chrono::steady_clock::now();
    const int n = 15 * full.nodes.size(), d = n + 3 * full.points.size() + 2;
    V diagonal = V::Zero(d);
    for (const auto& b : l.blocks) {
      V raw = b.state.colwise().squaredNorm().transpose();
      diagonal.head(n) += raw.head(n);
      diagonal.tail(2) += raw.tail(2);
      if (b.point_index >= 0) {
        diagonal.segment<3>(n + 3 * b.point_index) += b.point.colwise().squaredNorm().transpose();
      }
    }
    std::mt19937 random(27019);
    std::normal_distribution<double> normal;
    M probe(d, 12);
    for (int i = 0; i < d; ++i) {
      const double scale = diagonal[i] > 0. ? 1. / std::sqrt(diagonal[i]) : 1.;
      for (int j = 0; j < probe.cols(); ++j) {
        probe(i, j) = normal(random) * scale;
      }
    }
    M state(n + 2, probe.cols());
    state.topRows(n) = probe.topRows(n);
    state.bottomRows(2) = probe.bottomRows(2);
    V original_q = V::Zero(probe.cols()), original_g = V::Zero(probe.cols());
    double original_energy = 2. * l.constant;
    for (const auto& b : l.blocks) {
      M z = b.state * state;
      if (b.point_index >= 0) {
        z += b.point * probe.middleRows(n + 3 * b.point_index, 3);
      }
      original_q += z.colwise().squaredNorm().transpose();
      original_g += z.transpose() * b.residual;
      original_energy += b.residual.squaredNorm();
    }
    V restored_q = V::Zero(probe.cols()), restored_g = V::Zero(probe.cols());
    double restored_energy = 0.;
    auto add = [&](const p::Prior& prior) {
      const auto f = AtAnchor(prior);
      const M z = f.rows * Pull(full, prior, probe);
      restored_q += z.colwise().squaredNorm().transpose();
      restored_g += z.transpose() * f.residual;
      restored_energy += f.residual.squaredNorm();
    };
    for (const auto& [id, record] : archive.RecordsForTest()) {
      add(record->conditional);
    }
    add(live.prior);
    const double qerror = (restored_q - original_q).norm() / std::max(1., original_q.norm());
    const double gerror = (restored_g - original_g).norm() / std::max(1., original_g.norm());
    const double eerror = std::abs(restored_energy - original_energy) / std::max(1., original_energy);
    if (!(qerror < 1e-7 && gerror < 1e-7 && eerror < 1e-7)) {
      std::cerr << "INITIAL_COMPLEMENT_FAILURE " << qerror << ' ' << gerror << ' ' << eerror << '\n';
      return 1;
    }
    const auto original = AtAnchor(live.prior);
    std::set<int64_t> live_ids(live.ids.begin(), live.ids.end());
    int64_t requested = -1;
    for (auto id : full.ids) {
      if (!live_ids.count(id) && archive.OwnsPoint(id)) {
        requested = id;
        break;
      }
    }
    if (requested < 0) {
      throw std::runtime_error("no archived point to test");
    }
    live.ids.push_back(requested);
    live.points.emplace_back(1000., -500., 200.);
    archive.Restore(live);
    const auto all = p::Assemble(live, p::Linearize(live, false));
    std::vector<int> keep;
    // Restore changes prior ordering; assemble maps back to live graph order.
    for (int k = 0; k < 15 * int(live.nodes.size()); ++k) {
      keep.push_back(k);
    }
    for (int k = 0; k < 3 * int(live.points.size() - 1); ++k) {
      keep.push_back(15 * live.nodes.size() + k);
    }
    keep.push_back(all.rows.cols() - 2);
    keep.push_back(all.rows.cols() - 1);
    const auto marginal = g::eliminate(all, keep);
    V scale = original.rows.colwise().norm();
    for (int i = 0; i < scale.size(); ++i) {
      scale[i] = scale[i] > 0. ? 1. / scale[i] : 1.;
    }
    M A = original.rows * scale.asDiagonal(), B = marginal.rows * scale.asDiagonal();
    const double he = (B.transpose() * B - A.transpose() * A).norm() / std::max(1., (A.transpose() * A).norm());
    const double ge = (B.transpose() * marginal.residual - A.transpose() * original.residual).norm() /
                      std::max(1., A.norm() * original.residual.norm());
    const double ee = std::abs(marginal.residual.squaredNorm() - original.residual.squaredNorm()) /
                      std::max(1., original.residual.squaredNorm());
    if (!(he < 1e-7 && ge < 1e-7 && ee < 1e-7)) {
      std::cerr << "RESTORE_MARGINAL_FAILURE " << he << ' ' << ge << ' ' << ee << '\n';
      return 1;
    }
    const auto finish = std::chrono::steady_clock::now();
    const auto stats = archive.GetStats();
    std::ofstream out(argv[3]);
    if (!out) {
      throw std::runtime_error("cannot write actual math result");
    }
    out << std::setprecision(17) << "{\"passed\":true,\"states\":" << full.nodes.size()
        << ",\"points\":" << full.points.size() << ",\"quadratic_probe_error\":" << qerror
        << ",\"gradient_probe_error\":" << gerror << ",\"energy_error\":" << eerror
        << ",\"restore_current_information_error\":" << he << ",\"restore_current_gradient_error\":" << ge
        << ",\"restore_current_energy_error\":" << ee << ",\"requested_point\":" << requested
        << ",\"records_after_query\":" << stats.live_records << ",\"coefficient_bytes\":" << stats.coefficient_bytes
        << ",\"capture_seconds\":" << std::chrono::duration<double>(capture_end - start).count()
        << ",\"total_seconds\":" << std::chrono::duration<double>(finish - start).count() << "}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
