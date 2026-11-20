#pragma once
#include <Eigen/Dense>
#include <map>
#include "camera/rig.h"
#include "map/map.h"
#include "pipelines/gpu_window/window_math.h"
#include "pnp/visual_support.h"

namespace cuvslam::pnp {
namespace geometry_profile_math {
struct Rows {
  Eigen::MatrixXd pose, point;
  Eigen::VectorXd residual;
};
struct Step {
  Eigen::Matrix<double, 6, 1> pose = Eigen::Matrix<double, 6, 1>::Zero();
  std::vector<Eigen::Vector3d> points;
  double predicted = 0.;
  bool valid = false;
};
Step Solve(const std::vector<Rows>& blocks, double lambda);
}  // namespace geometry_profile_math

// Conditional geometric likelihood: historical keyframe poses are fixed.
// One latent point is shared across all current cameras, preserving their
// correlated map uncertainty. Latent corrections never modify the public map.
class GeometryProfile {
public:
  void reset();
  bool refine(const map::UnifiedMap&, const camera::Rig&, const std::unordered_map<TrackId, Vector3T>&,
              const std::vector<camera::Observation>&, size_t required, bool source_ray, int64_t time_ns,
              Isometry3T& rig_from_world, Matrix6T* information = nullptr);

private:
  struct Prior {
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Matrix3d rows = Eigen::Matrix3d::Zero();
    Eigen::Vector3d residual = Eigen::Vector3d::Zero();
    gpu_window::SourceRay chart;
  };
  int64_t map_time_ = -1;
  std::map<TrackId, Prior> priors_;
  void update(const map::UnifiedMap&, const camera::Rig&);
};
}  // namespace cuvslam::pnp
