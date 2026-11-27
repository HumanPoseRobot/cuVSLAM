#include "history_archive.h"
#include "joint_online.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace p = gpu_window::posterior;
using M = Eigen::MatrixXd;
using V = Eigen::VectorXd;
void Require(bool value, const char *message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}
template <class Change>
void Rejected(const p::Graph &base, Change change,
              const std::string &expected) {
  auto graph = base;
  change(graph);
  bool caught = false;
  try {
    p::Linearize(graph, false);
  } catch (const std::invalid_argument &error) {
    caught = std::string(error.what()).find(expected) != std::string::npos;
  }
  Require(caught, "invalid factor graph was not rejected for expected reason");
}
int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      return 2;
    }
    const auto base = p::LoadGraph(argv[1]);
    p::Linearize(base, false);
    Require(!base.visuals.empty() && !base.edges.empty() && !base.prior.empty(),
            "complete fixture required");
    Rejected(
        base,
        [](auto &g) {
          g.ids.push_back(g.ids.front());
          g.points.push_back(g.points.front());
        },
        "metadata");
    Rejected(
        base, [](auto &g) { g.times.back() = g.times.front(); }, "metadata");
    Rejected(
        base,
        [](auto &g) {
          g.visuals.push_back(g.visuals.front());
          g.visual_cameras.push_back(g.visual_cameras.front());
        },
        "duplicate");
    Rejected(
        base, [](auto &g) { g.edges.push_back(g.edges.front()); }, "duplicate");
    Rejected(
        base,
        [](auto &g) {
          const auto &v = g.visuals.front();
          g.prior.consumed_visual.push_back(
              {g.times[v.node], g.ids[v.point], g.visual_cameras.front()});
          std::sort(g.prior.consumed_visual.begin(),
                    g.prior.consumed_visual.end());
        },
        "owned");
    Rejected(
        base, [](auto &g) { g.prior.visual_through = g.times.back(); },
        "owned");
    Rejected(
        base,
        [](auto &g) {
          g.prior.imu_left_through = g.times[g.edges.front().left];
        },
        "owned");

    p::Graph graph;
    graph.times = {1};
    graph.nodes = {base.nodes.front()};
    for (double &value : graph.nodes[0].R) {
      value = double(float(value));
    }
    graph.gravity = base.gravity;
    graph.fixed_pose_time = -1;
    p::Prior prior;
    prior.times = graph.times;
    prior.nodes = graph.nodes;
    prior.gravity = graph.gravity;
    prior.fixed_pose_time = -1;
    p::Factor source;
    source.rows = M::Identity(17, 17);
    source.residual.resize(17);
    for (int i = 0; i < 17; ++i) {
      source.residual[i] = .1 * std::cos(i);
      for (int j = 0; j < 17; ++j) {
        source.rows(i, j) += .03 * std::sin(3 * i + 7 * j);
      }
    }
    auto unconverted = graph;
    unconverted.prior = prior;
    unconverted.prior.SetFactor(source);
    const auto wrong =
        p::Assemble(unconverted, p::Linearize(unconverted, false));
    const double negative_chart_error = (wrong.rows - source.rows).norm();
    Require(negative_chart_error > 1e-10,
            "chart negative control failed to detect omitted conversion");
    graph.prior = p::StoreLinearizedPrior(prior, source);
    const auto actual = p::Assemble(graph, p::Linearize(graph, false));
    const double jacobian_error = (source.rows - actual.rows).norm();
    const double residual_error = (source.residual - actual.residual).norm();
    Require(jacobian_error < 1e-12 && residual_error < 1e-12,
            "anchored chart changed local affine factor");
    double finite_difference_error = 0.;
    for (int k = 0; k < 17; ++k) {
      V step = V::Zero(17);
      step[k] = 1e-6;
      auto plus = p::Retract(graph, step), minus = p::Retract(graph, -step);
      const V derivative =
          (p::Assemble(plus, p::Linearize(plus, false)).residual -
           p::Assemble(minus, p::Linearize(minus, false)).residual) /
          (2e-6);
      const double column_error = (derivative - source.rows.col(k)).norm();
      std::cout << "FD_COLUMN " << k << " " << column_error << "\n";
      finite_difference_error = std::max(finite_difference_error, column_error);
    }
    Require(finite_difference_error < 1e-7,
            "stored chart derivative finite difference mismatch");
    auto ordinary = graph, shifted = graph;
    auto with_constant = graph.prior.factor();
    const int old_rows = with_constant.rows.rows();
    with_constant.rows.conservativeResize(old_rows + 1, 17);
    with_constant.rows.row(old_rows).setZero();
    with_constant.residual.conservativeResize(old_rows + 1);
    with_constant.residual[old_rows] = 1e5;
    shifted.prior.SetFactor(std::move(with_constant));
    const auto normal_report = p::Optimize(ordinary);
    const auto shifted_report = p::Optimize(shifted);
    const double constant_position_error =
        (Eigen::Map<const Eigen::Vector3d>(ordinary.nodes[0].p) -
         Eigen::Map<const Eigen::Vector3d>(shifted.nodes[0].p))
            .norm();
    Require(normal_report.success && shifted_report.success &&
                normal_report.iterations == shifted_report.iterations &&
                constant_position_error < 1e-10,
            "zero-information residual changed LM acceptance or stopping");
    std::ofstream result(argv[2]);
    result.precision(17);
    result
        << "{\"passed\":true,\"fault_injections\":7,\"chart_jacobian_error\":"
        << jacobian_error << ",\"chart_residual_error\":" << residual_error
        << ",\"finite_difference_error\":" << finite_difference_error
        << ",\"negative_chart_error\":" << negative_chart_error
        << ",\"constant_position_error\":" << constant_position_error << "}\n";
    std::cout << "CORRECTNESS_REGRESSION_PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
