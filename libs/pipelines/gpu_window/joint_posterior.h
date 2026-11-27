#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>
#include "pnp/joint_sqrt_prior.h"
#include "window_math.h"

namespace gpu_window::posterior {
using cuvslam::pnp::joint_geometry::Factor;
struct VisualKey {
  int64_t time = 0, point = 0;
  int camera = 0;
  bool operator<(const VisualKey& other) const {
    return std::tie(time, point, camera) < std::tie(other.time, other.point, other.camera);
  }
  bool operator==(const VisualKey& other) const {
    return time == other.time && point == other.point && camera == other.camera;
  }
};

// The variable order is navigation (15 per frame), Cartesian landmarks, then
// the two-dimensional gravity chart. Gravity is retained across the boundary;
// marginalizing it and subsequently treating its mean as exact is not equivalent.
struct Prior {
  // Copies share only immutable coefficients. A new marginal prior is sealed
  // from a separate mutable builder, so trials cannot write through aliases.
  const Factor& factor() const { return *factor_; }
  void SetFactor(Factor value) { factor_ = std::make_shared<const Factor>(std::move(value)); }
  std::vector<int64_t> times, ids;
  std::vector<State> nodes;
  std::vector<Eigen::Vector3d> points;
  Eigen::Vector3d gravity = Eigen::Vector3d(0., 0., -9.81);
  int64_t fixed_pose_time = -1;
  // visual_through retires departed frames. Evidence already absorbed at a
  // STILL RETAINED frame needs exact keys: new landmark birth observations at
  // that same frame must not be discarded by a blanket timestamp cutoff.
  int64_t visual_through = -1, imu_left_through = -1;
  std::vector<VisualKey> consumed_visual;
  bool empty() const { return factor().rows.cols() == 0; }

private:
  std::shared_ptr<const Factor> factor_ = std::make_shared<const Factor>();
};

struct Graph {
  std::vector<int64_t> times, ids;
  std::vector<State> nodes;
  std::vector<Eigen::Vector3d> points;
  std::vector<Edge> edges;
  std::vector<Visual> visuals;
  std::vector<int> visual_cameras;
  Eigen::Vector3d gravity = Eigen::Vector3d(0., 0., -9.81);
  Prior prior;
  int64_t fixed_pose_time = -1;
  double huber = 1.5;
  int iterations = 20;
};

struct Block {
  Eigen::MatrixXd state, point;
  Eigen::VectorXd residual;
  int point_index = -1;
};
struct Linearization {
  std::vector<Block> blocks;
  double cost = 0.;
  double constant = 0.;
  std::vector<VisualKey> visual_evidence;
};
struct Report {
  bool success = false;
  double before = 0., after = 0., max_linear_residual = 0.;
  int iterations = 0;
  double linearize_ms = 0., assemble_ms = 0., linear_solve_ms = 0., retract_ms = 0.;
};

Eigen::Matrix<double, 3, 2> GravityBasis(const Eigen::Vector3d& gravity);
Eigen::Vector3d RetractGravity(const Eigen::Vector3d& anchor, const Eigen::Vector2d& delta);
Eigen::Vector2d GravityDifference(const Eigen::Vector3d& value, const Eigen::Vector3d& anchor);
Eigen::Matrix2d GravityDifferenceJacobian(const Eigen::Vector3d& value, const Eigen::Vector3d& anchor);

Linearization Linearize(const Graph& graph, bool boundary);
Factor Assemble(const Graph& graph, const Linearization& linearization);
Graph Retract(const Graph& graph, const Eigen::VectorXd& increment);
Report Optimize(Graph& graph, Linearization* final_linearization = nullptr);

// Keeps precisely the named states and points, and always retains gravity.
// The caller passes ONLY the evidence to be absorbed at this boundary. Other
// raw factors remain owned by the live graph and must not be counted twice.
Prior Marginalize(const Graph& graph, const Linearization& consumed, const std::vector<int64_t>& keep_times,
                  const std::vector<int64_t>& keep_ids, int64_t visual_through, int64_t imu_left_through);
bool CurrentMarginal(const Graph& graph, int64_t time, Eigen::MatrixXd& covariance, Eigen::VectorXd& gradient,
                     const Linearization* final_linearization = nullptr);
}  // namespace gpu_window::posterior

namespace gpu_window::posterior {
void NormalEquations(const Graph&, const Linearization&, Eigen::MatrixXd&, Eigen::VectorXd&);
}
