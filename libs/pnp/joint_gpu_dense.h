#pragma once
#include <Eigen/Dense>
#include <cstdint>
namespace cuvslam::pnp::joint_geometry::dense {
struct GpuMetrics {
  uint64_t gram = 0, solve = 0, project = 0, compress = 0, failures = 0, rank_fallbacks = 0, allocations = 0;
};
GpuMetrics GetGpuMetrics();
bool WarmGpu();
bool GpuGram(const Eigen::MatrixXd&, Eigen::MatrixXd&);
bool GpuSolveSpd(const Eigen::MatrixXd&, const Eigen::VectorXd&, Eigen::VectorXd&);
bool GpuProjectFullRank(const Eigen::MatrixXd&, const Eigen::MatrixXd&, const Eigen::VectorXd&, Eigen::MatrixXd&,
                        Eigen::VectorXd&);
bool GpuCompress(const Eigen::MatrixXd&, const Eigen::VectorXd&, Eigen::MatrixXd&, Eigen::VectorXd&);
}  // namespace cuvslam::pnp::joint_geometry::dense
