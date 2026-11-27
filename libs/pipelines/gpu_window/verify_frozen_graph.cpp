#include <Eigen/SVD>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include "joint_initial_graph.h"

using namespace gpu_window::posterior;
namespace gaussian = cuvslam::pnp::joint_geometry;
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Independent reference: point nuisance ranges are computed by SVD rather
// than the production QR. The column order is [all navigation, gravity, probe
// points], deliberately different from the retained production prior.
gaussian::Factor ReferenceRows(const Graph& graph, const Linearization& linear, const std::vector<int64_t>& probe) {
  const int state = 15 * graph.nodes.size() + 2, dimension = state + 3 * probe.size();
  std::vector<Matrix> rows;
  std::vector<Vector> residuals;
  int total = linear.constant > 0. ? 1 : 0;
  for (const auto& block : linear.blocks) {
    Matrix kept = block.state;
    Vector residual = block.residual;
    auto wanted = probe.end();
    if (block.point_index >= 0) {
      wanted = std::find(probe.begin(), probe.end(), graph.ids[block.point_index]);
    }
    if (block.point_index >= 0 && wanted == probe.end()) {
      Matrix point = block.point;
      for (int k = 0; k < 3; ++k) {
        const double norm = point.col(k).norm();
        if (norm > 0.) {
          point.col(k) /= norm;
        }
      }
      Eigen::JacobiSVD<Matrix> svd(point, Eigen::ComputeFullU);
      const auto singular = svd.singularValues();
      const double tolerance =
          256. * std::numeric_limits<double>::epsilon() * std::max(point.rows(), point.cols()) * singular[0];
      const int rank = (singular.array() > tolerance).count();
      const Matrix null = svd.matrixU().rightCols(point.rows() - rank);
      kept = null.transpose() * block.state;
      residual = null.transpose() * block.residual;
    }
    Matrix row = Matrix::Zero(kept.rows(), dimension);
    row.leftCols(state) = kept;
    if (wanted != probe.end()) {
      row.middleCols(state + 3 * (wanted - probe.begin()), 3) = block.point;
    }
    total += row.rows();
    rows.push_back(std::move(row));
    residuals.push_back(std::move(residual));
  }
  gaussian::Factor factor;
  factor.rows = Matrix::Zero(total, dimension);
  factor.residual = Vector::Zero(total);
  int at = 0;
  for (size_t k = 0; k < rows.size(); ++k) {
    factor.rows.middleRows(at, rows[k].rows()) = rows[k];
    factor.residual.segment(at, residuals[k].size()) = residuals[k];
    at += rows[k].rows();
  }
  if (linear.constant > 0.) {
    factor.residual[at] = std::sqrt(2. * linear.constant);
  }
  return factor;
}

Graph Stored(const Prior& prior) {
  Graph graph;
  graph.times = prior.times;
  graph.ids = prior.ids;
  graph.nodes = prior.nodes;
  graph.points = prior.points;
  graph.gravity = prior.gravity;
  graph.fixed_pose_time = prior.fixed_pose_time;
  graph.prior = prior;
  return graph;
}

void Array(std::ostream& out, const Vector& value) {
  out << '[';
  for (int k = 0; k < value.size(); ++k) {
    if (k) {
      out << ',';
    }
    out << value[k];
  }
  out << ']';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument("usage: verify_frozen_graph input.bin output.json [12|all]");
    }
    const auto graph = LoadGraph(argv[1]);
    if (!graph.prior.empty() || graph.nodes.size() < 2 || graph.fixed_pose_time != graph.times.back()) {
      throw std::invalid_argument("expected frozen initialization graph with current-frame gauge");
    }
    const bool all = std::string(argv[3]) == "all";
    std::map<int64_t, std::set<int>> stereo;
    for (size_t i = 0; i < graph.visuals.size(); ++i) {
      const auto& visual = graph.visuals[i];
      if (visual.node == int(graph.nodes.size()) - 1) {
        stereo[graph.ids[visual.point]].insert(graph.visual_cameras[i]);
      }
    }
    std::vector<int64_t> retained;
    for (const auto& [id, cameras] : stereo) {
      if (all || cameras.size() >= 2) {
        retained.push_back(id);
        if (!all && retained.size() == 12) {
          break;
        }
      }
    }
    if (retained.empty()) {
      throw std::runtime_error("no stereo landmarks at current initialization frame");
    }
    const auto linear = Linearize(graph, false);
    const auto prior = Marginalize(graph, linear, {graph.times.back()}, retained, graph.times[graph.times.size() - 2],
                                   graph.times[graph.times.size() - 2]);
    const auto stored_graph = Stored(prior);
    auto production = Assemble(stored_graph, Linearize(stored_graph, false));
    std::vector<int> production_kept;
    for (int k = 6; k < 15; ++k) {
      production_kept.push_back(k);
    }
    if (!all) {
      for (int k = 15; k < production.rows.cols() - 2; ++k) {
        production_kept.push_back(k);
      }
    }
    production_kept.push_back(production.rows.cols() - 2);
    production_kept.push_back(production.rows.cols() - 1);
    production = gaussian::eliminate(production, production_kept);

    const std::vector<int64_t> probe = all ? std::vector<int64_t>{} : retained;
    const auto reference = ReferenceRows(graph, linear, probe);
    const int navigation = 15 * graph.nodes.size();
    std::vector<int> reference_kept;
    for (int k = navigation - 9; k < navigation; ++k) {
      reference_kept.push_back(k);
    }
    for (int k = navigation + 2; k < reference.rows.cols(); ++k) {
      reference_kept.push_back(k);
    }
    reference_kept.push_back(navigation);
    reference_kept.push_back(navigation + 1);
    std::vector<bool> keep(reference.rows.cols(), false);
    for (int k : reference_kept) {
      keep[k] = true;
    }
    Matrix nuisance(reference.rows.rows(), reference.rows.cols() - reference_kept.size());
    Matrix kept(reference.rows.rows(), reference_kept.size());
    Vector scale(kept.cols());
    int at = 0;
    for (int k = 0; k < reference.rows.cols(); ++k) {
      if (!keep[k]) {
        nuisance.col(at) = reference.rows.col(k);
        const double norm = nuisance.col(at).norm();
        if (norm > 0.) {
          nuisance.col(at) /= norm;
        }
        ++at;
      }
    }
    for (size_t k = 0; k < reference_kept.size(); ++k) {
      kept.col(k) = reference.rows.col(reference_kept[k]);
      scale[k] = kept.col(k).norm();
      if (!(scale[k] > 0.)) {
        throw std::runtime_error("retained variable has no input Jacobian");
      }
      kept.col(k) /= scale[k];
    }
    std::cerr << "SVD nuisance rows=" << nuisance.rows() << " cols=" << nuisance.cols() << '\n';
    Eigen::JacobiSVD<Matrix> svd(nuisance, Eigen::ComputeThinU);
    const auto singular = svd.singularValues();
    const double rank_tolerance =
        256. * std::numeric_limits<double>::epsilon() * std::max(nuisance.rows(), nuisance.cols()) * singular[0];
    const int rank = (singular.array() > rank_tolerance).count();
    const Matrix q = svd.matrixU().leftCols(rank);
    const Matrix reference_rows = kept - q * (q.transpose() * kept);
    const Vector reference_r = reference.residual - q * (q.transpose() * reference.residual);
    const Matrix actual_rows = production.rows * scale.cwiseInverse().asDiagonal();
    const Matrix expected_h = reference_rows.transpose() * reference_rows;
    const Vector expected_g = reference_rows.transpose() * reference_r;
    const Matrix actual_h = actual_rows.transpose() * actual_rows;
    const Vector actual_g = actual_rows.transpose() * production.residual;
    const double h_error = (actual_h - expected_h).norm() / std::max(1e-300, expected_h.norm());
    const double g_error =
        (actual_g - expected_g).norm() / std::max(1e-300, reference_rows.norm() * reference_r.norm());
    Eigen::JacobiSVD<Matrix> a(reference_rows, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::JacobiSVD<Matrix> b(actual_rows, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const double tolerance =
        256. * std::numeric_limits<double>::epsilon() * std::max(reference.rows.rows(), reference.rows.cols());
    const bool observable = a.singularValues().minCoeff() > tolerance * a.singularValues()[0] &&
                            b.singularValues().minCoeff() > tolerance * b.singularValues()[0];
    const double common_rank_bound = tolerance * a.singularValues()[0];
    const int rank_a = (a.singularValues().array() > common_rank_bound).count();
    const int rank_b = (b.singularValues().array() > common_rank_bound).count();
    const Vector minimum_a =
        reference_r - a.matrixU().leftCols(rank_a) * (a.matrixU().leftCols(rank_a).transpose() * reference_r);
    const Vector minimum_b = production.residual - b.matrixU().leftCols(rank_b) *
                                                       (b.matrixU().leftCols(rank_b).transpose() * production.residual);
    const double constant_error =
        std::abs(minimum_a.squaredNorm() - minimum_b.squaredNorm()) /
        std::max({1e-300, reference.residual.squaredNorm(), production.residual.squaredNorm()});
    const double numeric_budget =
        4096. * std::numeric_limits<double>::epsilon() * std::max(reference.rows.rows(), reference.rows.cols());
    bool passed =
        h_error <= numeric_budget && g_error <= numeric_budget && constant_error <= numeric_budget && rank_a == rank_b;
    std::ofstream out(argv[2]);
    out << std::setprecision(17) << "{\"input\":" << std::quoted(argv[1])
        << ",\"scope\":\"same frozen visual/IMU factor graph; no truth; independent SVD reference\""
        << ",\"frames\":" << graph.nodes.size() << ",\"landmarks\":" << graph.points.size()
        << ",\"visual_factors\":" << graph.visuals.size() << ",\"imu_factors\":" << graph.edges.size()
        << ",\"retained_landmarks\":" << retained.size() << ",\"covariance_probe_landmarks\":" << probe.size()
        << ",\"retained_rank\":" << rank_a << ",\"retained_dof\":" << kept.cols()
        << ",\"constant_energy_operator_error\":" << constant_error << ",\"nuisance_rank\":" << rank
        << ",\"retained_observable\":" << (observable ? "true" : "false")
        << ",\"scaled_information_relative_error\":" << h_error << ",\"scaled_rhs_operator_error\":" << g_error
        << ",\"machine_precision_budget\":" << numeric_budget
        << ",\"nonzero_retained_gradient_norm\":" << expected_g.norm()
        << ",\"reference_min_singular\":" << a.singularValues().minCoeff()
        << ",\"reference_condition\":" << a.singularValues()[0] / std::max(1e-300, a.singularValues().minCoeff())
        << ",\"update_order\":\"velocity3,gyro_bias3,acc_bias3,probe_points3,gravity2\"";
    if (observable) {
      const Vector mu_a = -a.solve(reference_r), mu_b = -b.solve(production.residual);
      const Matrix covariance_a =
          a.matrixV() * a.singularValues().array().square().inverse().matrix().asDiagonal() * a.matrixV().transpose();
      const Matrix covariance_b =
          b.matrixV() * b.singularValues().array().square().inverse().matrix().asDiagonal() * b.matrixV().transpose();
      const double mean_backward = (expected_h * mu_b + expected_g).norm() /
                                   std::max(1e-300, expected_h.norm() * mu_b.norm() + expected_g.norm());
      const double covariance_backward =
          (expected_h * covariance_b - Matrix::Identity(kept.cols(), kept.cols())).norm() /
          std::max(1e-300, expected_h.norm() * covariance_b.norm() + std::sqrt(double(kept.cols())));
      passed = passed && mean_backward <= numeric_budget && covariance_backward <= numeric_budget;
      out << ",\"mean_backward_error\":" << mean_backward << ",\"covariance_backward_error\":" << covariance_backward
          << ",\"mean_forward_difference_norm\":" << ((mu_a - mu_b).array() / scale.array()).matrix().norm()
          << ",\"covariance_forward_relative_difference\":"
          << (covariance_a - covariance_b).norm() / covariance_a.norm() << ",\"reference_mean_update\":";
      Array(out, mu_a.cwiseQuotient(scale));
      out << ",\"transferred_mean_update\":";
      Array(out, mu_b.cwiseQuotient(scale));
      out << ",\"weakest_scaled_direction\":";
      Array(out, a.matrixV().rightCols(1));
    }
    out << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
    out.close();
    std::cout << "passed=" << passed << " observable=" << observable << " H=" << h_error << " g=" << g_error << '\n';
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
