#pragma once
#include <memory>
#include <map>
#include <functional>
#include "camera/observation.h"
#include "map/map.h"
#include "imu/imu_sba_problem.h"
#include "pipelines/inertial_pnp.h"
namespace cuvslam::sba_imu {
class WindowBackend {
public:
 WindowBackend(); ~WindowBackend();
 struct FrameInput {
   map::UnifiedMap& map;
   const imu::ImuCalibration& calibration;
   const camera::Rig& rig;
   const imu::ImuMeasurementStorage& measurements;
   const std::unordered_map<TrackId,Vector3T>& landmarks;
   const std::vector<camera::Observation>& observations;
   const Vector3T& gravity;
   const pipelines::InertialPnPSettings& settings;
   const Pose& previous;
   const Pose& tracked;
   int64_t previous_time_ns;
   int64_t time_ns;
   std::function<bool(const Pose&,const std::map<TrackId,Vector3T>&)> accept_candidate;
 };
 bool solve(const FrameInput&,Pose&);
 bool commit(map::UnifiedMap&,const imu::ImuCalibration&,int64_t);
 void reset();
 void discard_pending();
private:
 struct Impl;std::unique_ptr<Impl> impl_;
};
}
