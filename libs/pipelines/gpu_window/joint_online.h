#pragma once

#include <memory>
#include <optional>
#include "joint_initial_graph.h"
#include "pipelines/inertial_pnp.h"

namespace gpu_window::posterior {
cuvslam::sba_imu::Pose StatePose(const State& state);
struct Initialization {
  Graph complete;
  Graph live;
  Eigen::MatrixXd root_covariance;
  Report solve;
};

class Online {
public:
  Online();
  ~Online();
  bool ready() const;
  void reset();
  void discard();
  bool initialize(Graph graph, const CurrentFrame& current, const cuvslam::camera::Rig& rig,
                  const cuvslam::imu::ImuCalibration& calibration, Initialization& result);
  void adopt(Initialization&& result, const CurrentFrame& current);

  bool solve(cuvslam::map::UnifiedMap& map, const cuvslam::camera::Rig& rig,
             const cuvslam::imu::ImuCalibration& calibration, const cuvslam::imu::ImuMeasurementStorage& storage,
             const std::unordered_map<cuvslam::TrackId, cuvslam::Vector3T>& points,
             const std::vector<cuvslam::camera::Observation>& observations,
             const cuvslam::pipelines::InertialPnPSettings& settings, int64_t time, bool is_keyframe,
             cuvslam::sba_imu::Pose& candidate, cuvslam::sba_imu::Pose& prediction, bool& prediction_valid);
  bool commit(cuvslam::map::UnifiedMap& map, const cuvslam::imu::ImuCalibration& calibration, int64_t time);
  bool pose_at(int64_t time, cuvslam::sba_imu::Pose& pose) const;
  bool prediction_at(int64_t time, cuvslam::sba_imu::Pose& pose) const;
  std::optional<cuvslam::Matrix3T> position_covariance(int64_t time) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gpu_window::posterior

namespace gpu_window::posterior {
Graph RetainForProfile(const Graph&, int);
}
