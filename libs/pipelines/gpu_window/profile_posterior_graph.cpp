#include <sys/resource.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "joint_online.h"
#include "pnp/joint_gpu_dense.h"
using namespace gpu_window::posterior;
int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }
  try {
    auto graph = LoadGraph(argv[1]);
    const auto warming = std::chrono::steady_clock::now();
    if (std::getenv("VIO_JOINT_GPU_QR") || std::getenv("VIO_JOINT_GPU_SPD") || std::getenv("VIO_JOINT_GPU_GRAM")) {
      if (!cuvslam::pnp::joint_geometry::dense::WarmGpu()) {
        return 3;
      }
    }
    const double warmup = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warming).count();
    const auto counters_before = cuvslam::pnp::joint_geometry::dense::GetGpuMetrics();
    rusage memory_before{};
    getrusage(RUSAGE_SELF, &memory_before);
    const auto start = std::chrono::steady_clock::now();
    const bool reuse = std::getenv("VIO_POSTERIOR_REUSE_LINEARIZATION");
    Linearization final_linearization;
    auto report = Optimize(graph, reuse ? &final_linearization : nullptr);
    const auto optimized = std::chrono::steady_clock::now();
    Eigen::MatrixXd covariance;
    Eigen::VectorXd gradient;
    const bool observable =
        CurrentMarginal(graph, graph.times.back(), covariance, gradient, reuse ? &final_linearization : nullptr);
    const auto marginal = std::chrono::steady_clock::now();
    auto retained = RetainForProfile(graph, 3);
    const auto end = std::chrono::steady_clock::now();
    rusage memory_after{};
    getrusage(RUSAGE_SELF, &memory_after);
    auto milliseconds = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::ofstream f(argv[2]);
    f << std::setprecision(17);
    const auto counters = cuvslam::pnp::joint_geometry::dense::GetGpuMetrics();
    f << "{\"minor_page_faults\":" << memory_after.ru_minflt - memory_before.ru_minflt << ",\"warmup_ms\":" << warmup
      << ",\"gpu_gram\":" << counters.gram - counters_before.gram
      << ",\"gpu_solve\":" << counters.solve - counters_before.solve
      << ",\"gpu_project\":" << counters.project - counters_before.project
      << ",\"gpu_compress\":" << counters.compress - counters_before.compress
      << ",\"gpu_failures\":" << counters.failures - counters_before.failures
      << ",\"gpu_rank_fallbacks\":" << counters.rank_fallbacks - counters_before.rank_fallbacks
      << ",\"gpu_allocations\":" << counters.allocations - counters_before.allocations
      << ",\"success\":" << report.success << ",\"observable\":" << observable << ",\"frames\":" << graph.nodes.size()
      << ",\"points\":" << graph.points.size() << ",\"visuals\":" << graph.visuals.size()
      << ",\"iterations\":" << report.iterations << ",\"before\":" << report.before << ",\"after\":" << report.after
      << ",\"optimize_ms\":" << milliseconds(start, optimized) << ",\"linearize_ms\":" << report.linearize_ms
      << ",\"assemble_ms\":" << report.assemble_ms << ",\"linear_solve_ms\":" << report.linear_solve_ms
      << ",\"retract_ms\":" << report.retract_ms << ",\"covariance_ms\":" << milliseconds(optimized, marginal)
      << ",\"marginalize_ms\":" << milliseconds(marginal, end) << ",\"total_ms\":" << milliseconds(start, end)
      << ",\"state\":[";
    bool first = true;
    for (const auto& node : graph.nodes) {
      for (const auto* p : {node.R, node.p, node.v, node.bg, node.ba}) {
        const int n = p == node.R ? 9 : 3;
        for (int i = 0; i < n; ++i) {
          if (!first) {
            f << ',';
          }
          first = false;
          f << p[i];
        }
      }
    }
    f << "],\"covariance\":[";
    for (int row = 0; row < covariance.rows(); ++row) {
      for (int col = 0; col < covariance.cols(); ++col) {
        if (row || col) {
          f << ',';
        }
        f << covariance(row, col);
      }
    }
    f << "],\"gravity\":[" << graph.gravity.x() << ',' << graph.gravity.y() << ',' << graph.gravity.z() << "]}";
    return report.success && observable ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
