#pragma once

#include <string>
#include <unordered_map>
#include "camera/rig.h"
#include "imu/imu_sba_problem.h"
#include "joint_posterior.h"
#include "map/map.h"

namespace gpu_window::posterior {
struct CurrentFrame {
  int64_t time_ns;
  cuvslam::sba_imu::Pose pose;
  std::unordered_map<cuvslam::TrackId, cuvslam::Vector3T> points;
  std::vector<cuvslam::camera::Observation> observations;
};
Graph BuildInitializationGraph(const cuvslam::map::UnifiedMap::SubMap& history,
                               const std::vector<cuvslam::sba_imu::Pose>& initialized, const CurrentFrame& current,
                               const cuvslam::camera::Rig& rig, const cuvslam::imu::ImuCalibration& calibration,
                               const cuvslam::imu::ImuMeasurementStorage& storage, const Eigen::Vector3d& gravity);
void SaveGraph(const Graph& graph, const std::string& path);
Graph LoadGraph(const std::string& path);
}  // namespace gpu_window::posterior
