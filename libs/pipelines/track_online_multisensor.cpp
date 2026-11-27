#include <fstream>
#include <iomanip>
#include <unordered_set>
#include <set>
#include "pipelines/gpu_window/frontend_guard_adapter.h"
#include "imu_runtime_options.h"

/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include "pipelines/track_online_multisensor.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>

#include "camera/observation.h"
#include "camera/rig.h"
#include "common/frame_id.h"
#include "common/include_eigen.h"
#include "common/isometry.h"
#include "common/rerun.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"
#include "pipelines/visualizer.h"

#include "pipelines/service_sba.h"
#ifdef USE_CUDA
#include "pipelines/service_sba_gpu.h"
#endif

namespace cuvslam::pipelines {

namespace {

void flatten_observations(const MulticamObservations& multicam_obs, std::vector<camera::Observation>& obs_vector) {
  obs_vector.clear();
  size_t num_observations = 0;
  for (const auto& cam_observations : multicam_obs) {
    num_observations += cam_observations.size();
  }
  obs_vector.reserve(num_observations);
  for (const auto& observations : multicam_obs) {
    std::copy(observations.begin(), observations.end(), std::back_inserter(obs_vector));
  }
}

void warn_if_non_pinhole(const camera::Rig& rig) {
  for (int32_t i = 0; i < rig.num_cameras; ++i) {
    if (dynamic_cast<const camera::PinholeCameraModel*>(rig.intrinsics[i]) == nullptr) {
      TraceWarning("SolverSfMMultisensor supports only pinhole cameras. Camera %d uses a different model.", i);
    }
  }
}

}  // namespace

SolverSfMMultisensor::SolverSfMMultisensor(map::UnifiedMap& map, const camera::Rig& rig,
                                           const sba::Settings& sba_settings)
    : rig_(rig), map_(map), pose_estimator_(rig), depth_maps_(rig_), with_imu_(false) {
  warn_if_non_pinhole(rig_);

  const auto& sba_mode = sba_settings.mode;
  if (sba_mode != sba::OriginalCPU && sba_mode != sba::OriginalGPU && sba_mode != sba::Disabled) {
    TraceError("Original VO cant run with inertial SBA");
  }

  switch (sba_mode) {
    case sba::OriginalCPU:
      sba_service_ = std::make_unique<CpuSbaService>(sba_settings, rig, map_);
      break;
#ifdef USE_CUDA
    case sba::OriginalGPU:
      sba_service_ = std::make_unique<GpuSbaService>(sba_settings, rig, map_);
      break;
#endif
    default:
      sba_service_ = nullptr;
      break;
  }
}

SolverSfMMultisensor::SolverSfMMultisensor(map::UnifiedMap& map, const camera::Rig& rig,
                                           const sba::Settings& sba_settings, const cuvslam::imu::ImuCalibration& calib)
    : rig_(rig), map_(map), pose_estimator_(rig), depth_maps_(rig_), with_imu_(true) {
  warn_if_non_pinhole(rig_);

  const auto& sba_mode = sba_settings.mode;
  switch (sba_mode) {
    case sba::InertialCPU:
      sba_service_ = std::make_unique<ImuSbaCPUService>(sba_settings, rig, calib, map_);
      break;
#ifdef USE_CUDA
    case sba::InertialGPU:
      sba_service_ = std::make_unique<ImuSbaGPUService>(sba_settings, rig, calib, map_);
      break;
#endif
    default:
      sba_service_ = nullptr;
      break;
  }

  imu_ = std::make_unique<ImuFusionContext>(map_, rig_, calib, pose_estimator_);
  imu_->register_gravity_callback();
  const char* mode = std::getenv("CUVSLAM_CAM01_TEMPORAL");
  if (mode && rig_.num_cameras==4 && !sba_settings.async) {
    const std::string_view name(mode);
    temporal_audit_ = name=="audit" || name=="insert" || name=="strict";
    temporal_insert_ = name=="insert" || name=="strict";
    temporal_strict_ = name=="strict";

  }
  metric_init_enabled_ = imu_runtime::MetricEnabled(rig_.num_cameras, sba_settings.async, std::getenv("CUVSLAM_METRIC_INIT"));
  TraceMessage("VIO_OPTIONS cameras=%d metric=%d temporal=%d", rig_.num_cameras, static_cast<int>(metric_init_enabled_), static_cast<int>(temporal_insert_));
}

SolverSfMMultisensor::~SolverSfMMultisensor() = default;

const camera::Rig& SolverSfMMultisensor::getRig() const { return rig_; }

void SolverSfMMultisensor::reset() {
  bootstrap_refresh_requested_ = false;
  bootstrap_stereo_landmarks_.clear();
  bootstrap_retaining_=false;
  temporal_.clear();
  temporal_ready_ = false;
  depth_maps_.reset();
  if (sba_service_) {
    sba_service_->restart();
  }
  if (imu_) {
    imu_->reset();
  }
  prev_rig_from_world_ = Isometry3T::Identity();
  prev_static_info_exp_.setZero();
}

bool SolverSfMMultisensor::solveNextFrame(int64_t time_ns, const sof::FrameState& frameState,
                                          const MulticamObservations& observations, const pnp::RGBDInfos& depth_infos,
                                          Isometry3T& world_from_rig, Matrix6T& static_info_exp,
                                          const SolverPerFrameSettings& solver_settings, std::vector<Track2D>* tracks2d,
                                          Tracks3DMap* tracks3d) {
  bootstrap_retaining_=false;
  if (with_imu_) {
    return solveNextFrameInertial(time_ns, frameState, observations, depth_infos, world_from_rig, static_info_exp,
                                  solver_settings, tracks2d, tracks3d);
  }
  return solveNextFrameVisualOnly(time_ns, frameState, observations, depth_infos, world_from_rig, static_info_exp,
                                  solver_settings, tracks2d, tracks3d);
}

bool SolverSfMMultisensor::solveNextFrameVisualOnly(int64_t time_ns, const sof::FrameState& frameState,
                                                    const MulticamObservations& observations,
                                                    const pnp::RGBDInfos& depth_infos, Isometry3T& world_from_rig,
                                                    Matrix6T& static_info_exp,
                                                    const SolverPerFrameSettings& solver_settings,
                                                    std::vector<Track2D>* tracks2d, Tracks3DMap* tracks3d) {
  TRACE_EVENT ev = profiler_domain_.trace_event("SolverSfMMultisensor::solveNextFrame()", profiler_color_);

  flatten_observations(observations, obs_vector_);
  bool result = true;

  if (map_.empty()) {
    static_info_exp.setZero();
  } else {
    map_.get_recent_landmarks(recent_landmarks_);
    if (std::getenv("VIO_OBSERVATION_MAP_LOOKUP")) {
      map::Map<TrackId,Vector3T> observed;
      map_.get_observed_landmarks(obs_vector_,observed);
      size_t additional=0;
      for(const auto& [id,point]:observed){additional+=recent_landmarks_.try_emplace(id,point).second;}
      TraceMessage("MAP_OBSERVATION_LOOKUP t=%lld obs=%zu added=%zu",(long long)time_ns,obs_vector_.size(),additional);
    }


    Isometry3T rig_from_world = prev_rig_from_world_;
    bool res;
    {
      TRACE_EVENT pnp_ev = profiler_domain_.trace_event("pose_estimator.solve");
      res = pose_estimator_.solve(rig_from_world, static_info_exp, obs_vector_, recent_landmarks_, depth_infos,
                                  depth_maps_.planes(), depth_maps_.depth_points());
    }

    RERUN(logTrajectory, rig_from_world, "world/trajectories/vo_trajectory", Color(0, 255, 0), TrajectoryType::VO);

    if (res) {
      prev_rig_from_world_ = rig_from_world;
      prev_static_info_exp_ = static_info_exp;
    } else {
      static_info_exp = prev_static_info_exp_;
      result = false;
    }

    {
      TRACE_EVENT dpm_ev = profiler_domain_.trace_event("depth_point_map.update");
      if (depth_maps_.update_post_solve(depth_infos, prev_rig_from_world_.inverse())) {
        RERUN(logPoints3D, depth_maps_.depth_points(), "world/depth_points", Color(0, 200, 255), 0.008f);
      }
    }
  }

  world_from_rig = prev_rig_from_world_.inverse();
  RERUN(logCameraFrustums, rig_, world_from_rig, "world/frustums");

  if (frameState == sof::FrameState::Key) {
    map::State state{prev_rig_from_world_};
    process_keyframe(time_ns, world_from_rig, obs_vector_, depth_infos, state, sba_imu::IMUPreintegration{},
                     solver_settings.sba);
  }

  if (tracks2d && tracks3d) {
    TRACE_EVENT export_ev = profiler_domain_.trace_event("exportTracks");
    exportTracks(obs_vector_, *tracks2d, *tracks3d, prev_rig_from_world_);
  }

  return result;
}

bool SolverSfMMultisensor::solveNextFrameInertial(int64_t time_ns, const sof::FrameState& frameState,
                                                  const MulticamObservations& observations,
                                                  const pnp::RGBDInfos& depth_infos, Isometry3T& world_from_rig,
                                                  Matrix6T& static_info_exp,
                                                  const SolverPerFrameSettings& solver_settings,
                                                  std::vector<Track2D>* tracks2d, Tracks3DMap* tracks3d) {
  TRACE_EVENT ev = profiler_domain_.trace_event("SolverSfMMultisensor::solveNextFrame(inertial)", profiler_color_);

  flatten_observations(observations, obs_vector_);
  RERUN(logObservations, obs_vector_, rig_, "world/camera_0/images/observations", Color(255, 165, 0));

  auto prep = imu_->prepare_frame(time_ns);
  TraceMessage("VIO_FUSION_STATE t=%lld ready=%d gravity=%d healthy=%d", static_cast<long long>(time_ns), static_cast<int>(prep.imu_state == StateMachine::State::Ok), static_cast<int>(map_.get_gravity().has_value()), static_cast<int>(prep.no_drops));

  if(const char* dir=std::getenv("VIO_MAP_SNAPSHOT_DIR")) {
    const double seconds=time_ns*1e-9;const char* high=std::getenv("VIO_DIAG_MAX_S");
    if(!high || seconds<=std::atof(high)) {
      std::ofstream f(std::string(dir)+"/"+std::to_string(time_ns)+".history");f<<std::setprecision(17);
      const auto history=map_.get_recent_submap(map_.size(),false);
      for(size_t i=0;i<history.consecutive_keyframes.size();++i) {
        const auto& k=history.consecutive_keyframes[i].keyframe;f<<"K "<<k->time_ns();
        const auto pose=k->get_state().rig_from_w;
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)f<<' '<<pose.matrix()(r,c);f<<'\n';
        for(const auto& lo:history.landmark_and_obs[i]) {
          const auto point=lo.landmark->get_pose();if(!point)continue;
          for(const auto& o:lo.observations)f<<"O "<<k->time_ns()<<' '<<o.id<<' '<<int(o.cam_id)<<' '<<point->transpose()<<' '<<o.xy.transpose()<<' '<<o.xy_info(0,0)<<' '<<o.xy_info(0,1)<<' '<<o.xy_info(1,1)<<'\n';
        }
      }
      for(const auto& o:obs_vector_)f<<"C "<<o.id<<' '<<int(o.cam_id)<<' '<<o.xy.transpose()<<' '<<o.xy_info(0,0)<<' '<<o.xy_info(0,1)<<' '<<o.xy_info(1,1)<<'\n';
    }
  }
  bool pnp_result = false;
  auto& recovery = imu_->recovery_anchor;
  const bool trying_recovery = recovery && prep.no_drops && prep.imu_state == StateMachine::State::Ok;
  if (!map_.empty()) {
    map_.get_recent_landmarks(recent_landmarks_);
    if (std::getenv("VIO_OBSERVATION_MAP_LOOKUP")) {
      map::Map<TrackId,Vector3T> observed;
      map_.get_observed_landmarks(obs_vector_,observed);
      size_t additional=0;
      for(const auto& [id,point]:observed){additional+=recent_landmarks_.try_emplace(id,point).second;}
      TraceMessage("MAP_OBSERVATION_LOOKUP t=%lld obs=%zu added=%zu",(long long)time_ns,obs_vector_.size(),additional);
    }


    if (prep.imu_state != StateMachine::State::Ok && std::getenv("VIO_BOOTSTRAP_STEREO_REFRESH")) {
      for(const auto& observation:obs_vector_) {
        auto point=bootstrap_stereo_landmarks_.find(observation.id);
        if(point!=bootstrap_stereo_landmarks_.end()) recent_landmarks_.try_emplace(point->first,point->second);
      }
    }
    if (prep.imu_state == StateMachine::State::Ok) {
      if (trying_recovery) {
        for (const auto& landmark : recovery->landmarks) {
          recent_landmarks_.try_emplace(landmark.id, landmark.point_w);
        }
      }
      pnp_result = imu_->solve_inertial(recent_landmarks_, obs_vector_, depth_infos, depth_maps_.planes(),
                                        depth_maps_.depth_points(), time_ns, solver_settings.imu_pnp, prep.no_drops,
                                        frameState == sof::FrameState::Key && !trying_recovery);
    } else {
      pnp_result = imu_->solve_visual(recent_landmarks_, obs_vector_, depth_infos, depth_maps_.planes(),
                                      depth_maps_.depth_points(), time_ns);
    }
  }

  const bool bootstrap = metric_init_enabled_ && !map_.get_gravity() &&
                         std::none_of(depth_infos.begin(),depth_infos.end(),[](auto p){return p!=nullptr;});
  imu_->set_init_observation_requirement(size_t(std::max(0,solver_settings.imu_pnp.min_observations)));
  if(!pnp_result&&bootstrap&&!solver_settings.sba.async&&std::getenv("VIO_BOOTSTRAP_RETAIN_MAP")&&imu_->defer_bootstrap_failure(time_ns,solver_settings.sm)) {
    bootstrap_retaining_=true;bootstrap_refresh_requested_=true;
    world_from_rig=prev_rig_from_world_.inverse();static_info_exp.setZero();return false;
  }
  bootstrap_refresh_requested_=false;
  if(bootstrap&&pnp_result&&prep.no_drops&&!solver_settings.sba.async&&std::getenv("VIO_BOOTSTRAP_STEREO_REFRESH")) {
    const Isometry3T accepted=imu_->accepted_working_rig_pose();
    const Isometry3T rw=accepted.inverse();size_t support=0;
    for(const auto& observation:obs_vector_) {
      const auto point=recent_landmarks_.find(observation.id);
      if(point==recent_landmarks_.end()||observation.cam_id>=rig_.num_cameras){continue;}
      const Vector3T pc=rig_.camera_from_rig[observation.cam_id]*rw*point->second;
      if(!pc.allFinite()||pc.z()<=.05f){continue;}
      const Vector2T error=pc.hnormalized()-observation.xy;Vector2T pixel;
      if(!frontend_guard::actual_pixel_residual(*rig_.intrinsics[observation.cam_id],pc.hnormalized(),observation.xy,pixel)){continue;}
      if(frontend_guard::observation_consistent(9.*observation.xy_info.cast<double>(),error.cast<double>(),pixel.cast<double>())){++support;}
    }
    const size_t required=std::max(0,solver_settings.imu_pnp.min_observations);
    bootstrap_refresh_requested_=support<required;
    TraceMessage("BOOTSTRAP_SUPPORT t=%lld support=%zu required=%zu request=%d",(long long)time_ns,support,required,int(bootstrap_refresh_requested_));
  }
  if (std::getenv("VIO_BOOTSTRAP_STEREO_REFRESH") && !bootstrap && prep.no_drops) {
    size_t mapped=0;
    for(const auto& observation:obs_vector_) mapped+=recent_landmarks_.count(observation.id);
    bootstrap_refresh_requested_=!pnp_result || mapped<size_t(std::max(0,solver_settings.imu_pnp.min_observations));
  }
  if (std::getenv("VIO_BOOTSTRAP_STEREO_REFRESH")) {
    // Private bootstrap reservoir: no extra map keyframe, SBA or gravity-init
    // event. Creation uses an accepted pose; only a later image can use these
    // points in PnP and must pass the unchanged support/rank checks.
    if (!bootstrap || !pnp_result || !prep.no_drops || frameState==sof::FrameState::Key) {
      bootstrap_stereo_landmarks_.clear();
    } else {
      std::unordered_set<TrackId> visible;
      for(const auto& observation:obs_vector_) visible.insert(observation.id);
      for(auto it=bootstrap_stereo_landmarks_.begin();it!=bootstrap_stereo_landmarks_.end();) {
        if(!visible.count(it->first)) it=bootstrap_stereo_landmarks_.erase(it); else ++it;
      }
      if(bootstrap_refresh_requested_) {
        MulticamTriangulator triangulator(rig_);
        const auto points=triangulator.triangulate(imu_->accepted_working_rig_pose(),obs_vector_);
        size_t added=0;
        for(const auto& point:points) {
          if (point.point_w.allFinite() && !recent_landmarks_.count(point.id)) {
            const bool inserted = bootstrap_stereo_landmarks_.try_emplace(point.id, point.point_w).second;
            added += inserted;
            if (inserted) {
              if (const char* directory = std::getenv("VIO_TRIANGULATION_BIRTH_DIR")) {
                std::ofstream f(std::string(directory) + "/" + std::to_string(time_ns) + ".reservoir", std::ios::app);
                f << std::setprecision(17) << "P " << point.id << ' ' << point.point_w.transpose() << '\n';
                const auto rw = imu_->accepted_working_rig_pose().inverse().matrix();
                f << "T";
                for (int r = 0; r < 4; ++r) {
                  for (int c = 0; c < 4; ++c) { f << ' ' << rw(r,c); }
                }
                f << '\n';
                for (const auto& o : obs_vector_) {
                  if (o.id == point.id) {
                    f << "O " << int(o.cam_id) << ' ' << o.xy.transpose() << ' '
                      << o.xy_info(0,0) << ' ' << o.xy_info(0,1) << ' ' << o.xy_info(1,1) << '\n';
                  }
                }
              }
            }
          }
        }
        TraceMessage("BOOTSTRAP_STEREO_RESERVOIR t=%lld added=%zu total=%zu",(long long)time_ns,added,bootstrap_stereo_landmarks_.size());
      }
    }
  }
  auto frame = imu_->finalize_frame(pnp_result, prep.no_drops, time_ns, frameState, solver_settings.sm, bootstrap);
  const bool just_initialized = bootstrap && map_.get_gravity().has_value();
  const bool posterior_initial_key = just_initialized && std::getenv("VIO_POSTERIOR_TRANSFER");
  world_from_rig = frame.world_from_rig;
  static_info_exp = frame.static_info_exp;
  prev_rig_from_world_ = world_from_rig.inverse();

  if (trying_recovery && pnp_result && frame.valid_pose) {
    imu_->commit_recovery_anchor(time_ns);
  }
  if (!pnp_result && prep.no_drops && imu_->can_seed_recovery() && !solver_settings.sba.async &&
      (frameState == sof::FrameState::Key || std::getenv("VIO_BOOTSTRAP_STEREO_REFRESH"))) {
    const auto& predicted_pose = imu_->propagated_pose();
    // Triangulate in the propagated rig pose, not in the last published pose.
    // The hypothesis stays private until later images validate its reprojection.
    const Isometry3T seed_world_from_rig = imu_->recovery_rig_pose();
    MulticamTriangulator triangulator(rig_);
    auto candidates = triangulator.triangulate(seed_world_from_rig, obs_vector_);
    std::vector<Landmark> points;
    map::Map<TrackId, Vector3T> point_lookup;
    for (const auto& point : candidates) {
      if (point.point_w.allFinite() && recent_landmarks_.count(point.id) == 0) {
        points.push_back(point);
        point_lookup.emplace(point.id, point.point_w);
      }
    }
    if (imu_->validate_recovery_seed(point_lookup, obs_vector_, solver_settings.imu_pnp.min_observations, time_ns)) {
      std::vector<camera::Observation> supporting;
      for (const auto& observation : obs_vector_) {
        if (point_lookup.count(observation.id) != 0) {
          supporting.push_back(observation);
        }
      }
      recovery = ImuFusionContext::RecoveryAnchor{time_ns, predicted_pose, imu_->last_kf_preint(),
                                                  std::move(points), std::move(supporting)};
      TraceMessage("VIO_STEREO_SEED t=%lld landmarks=%zu", static_cast<long long>(time_ns),
                   recovery->landmarks.size());
    }
  }

  RERUN(logTrajectory, prev_rig_from_world_, "world/trajectories/vo_trajectory", Color(0, 255, 0), TrajectoryType::VO);

  // Visualise IMU-fusion intermediates anchored at the rig origin in world frame:
  //   - velocity (cyan) at true magnitude (m/s ↔ m on the world axes)
  //   - gravity (red) scaled by 0.1 so a 9.81 m/s² vector renders as a ~1 m arrow
  if (auto state = imu_->has_imu_state() ? std::optional<Vector3T>{imu_->velocity()} : std::nullopt; state) {
    [[maybe_unused]] const Vector3T origin = world_from_rig.translation();
    RERUN(logVector3D, *state, origin, "world/imu/velocity", Color(0, 200, 200), 1.0f);
    if (auto g = map_.get_gravity(); g) {
      RERUN(logVector3D, *g, origin, "world/imu/gravity", Color(220, 60, 60), 0.1f);
    }
  }

  const bool ready = temporal_audit_ && pnp_result && prep.no_drops &&
                     prep.imu_state == StateMachine::State::Ok &&
                     std::none_of(depth_infos.begin(),depth_infos.end(),[](auto p){ return p != nullptr; });
  const bool online_temporal = std::getenv("CUVSLAM_ONLINE_TEMPORAL") && !(std::getenv("VIO_POSTERIOR_TRANSFER") && map_.get_gravity());
  const bool recoverable_loss = frame.lost && imu_->can_reacquire_after_loss();
  if ((!online_temporal && (!ready || !temporal_ready_)) || (frame.lost && !recoverable_loss)) {
    temporal_.clear();
  }
  temporal_ready_ = ready;
  if (frame.lost) {
    return false;
  }

  // Refresh depth-point map under the latest pose so the next frame's ICP factor is anchored
  // correctly. No-op for depth-less rigs.
  {
    TRACE_EVENT dpm_ev = profiler_domain_.trace_event("depth_point_map.update");
    if (depth_maps_.update_post_solve(depth_infos, world_from_rig)) {
      RERUN(logPoints3D, depth_maps_.depth_points(), "world/depth_points", Color(0, 200, 255), 0.008f);
    }
  }

  RERUN(logCameraFrustums, rig_, world_from_rig, "world/frustums");

  if ((frameState == sof::FrameState::Key || posterior_initial_key) && imu_->allow_map_update()) {
    map::State state = imu_->build_keyframe_state(prev_rig_from_world_, frame.lost);
    process_keyframe(time_ns, world_from_rig, obs_vector_, depth_infos, state, imu_->last_kf_preint(),
                     solver_settings.sba, !just_initialized && !imu_->window_active());
    if (!solver_settings.sba.async && sba_service_ && imu_->read_committed_state(time_ns, world_from_rig, static_info_exp)) {
      prev_rig_from_world_ = world_from_rig.inverse();
    }
    if (!imu_->on_keyframe_committed(frame.lost)) {
      temporal_.clear();
      return false;
    }
    if (std::getenv("VIO_POSTERIOR_TRANSFER") && imu_->read_committed_state(time_ns, world_from_rig, static_info_exp)) {
      prev_rig_from_world_ = world_from_rig.inverse();
    }
    if (!online_temporal || just_initialized) temporal_.clear();
  }
  if (online_temporal && pnp_result && prep.no_drops && imu_->allow_map_update())
    temporal_.remember(time_ns,rig_,world_from_rig,obs_vector_,~0u);
  if (temporal_ready_ && !online_temporal && !(std::getenv("VIO_POSTERIOR_TRANSFER") && map_.get_gravity())) temporal_.remember(time_ns,rig_,world_from_rig,obs_vector_,12u);

  if (tracks2d && tracks3d) {
    TRACE_EVENT export_ev = profiler_domain_.trace_event("exportTracks");
    exportTracks(obs_vector_, *tracks2d, *tracks3d, prev_rig_from_world_);
  }

  return true;
}

void SolverSfMMultisensor::process_keyframe(int64_t time_ns, const Isometry3T& world_from_rig,
                                            const std::vector<camera::Observation>& obs,
                                            const pnp::RGBDInfos& depth_infos, const map::State& state,
                                            const sba_imu::IMUPreintegration& preint,
                                            const sba::Settings& sba_settings, bool run_sba) {
  TRACE_EVENT kf_ev = profiler_domain_.trace_event("keyframe_processing");

  std::vector<Landmark> landmarks;
  {
    TRACE_EVENT tri_ev = profiler_domain_.trace_event("triangulate_and_lift");
    landmarks = depth_maps_.build_keyframe_landmarks(world_from_rig, obs, depth_infos);
  }

  if (std::getenv("CUVSLAM_ONLINE_TEMPORAL") && !(std::getenv("VIO_POSTERIOR_TRANSFER") && map_.get_gravity())) {
    const size_t before = landmarks.size();
    temporal_.extend(time_ns, rig_, world_from_rig, obs, recent_landmarks_, ~0u,
                     true, 2.f, 1.75f, landmarks);
    TraceMessage("ONLINE_TEMPORAL t=%lld added=%zu", (long long)time_ns, landmarks.size()-before);
  } else if (temporal_ready_ && !(std::getenv("VIO_POSTERIOR_TRANSFER") && map_.get_gravity())) {
    // Slots 2,3 are physical cameras 0,1 in the experimental rig [2,3,0,1].
    temporal_.extend(time_ns, rig_, world_from_rig, obs, recent_landmarks_, 12u, temporal_insert_,
                     temporal_strict_ ? 4.f : 2.f, temporal_strict_ ? 1.f : 1.75f, landmarks);
  }
  if(const char* dir=std::getenv("VIO_MAP_SNAPSHOT_DIR")) {
    std::ofstream f(std::string(dir)+"/"+std::to_string(time_ns)+".created");f<<std::setprecision(17);
    for(const auto& point:landmarks) f<<point.id<<' '<<point.point_w.transpose()<<'\n';
  }
  map_.add_keyframe(time_ns, state, preint, obs, landmarks);

  {
    TRACE_EVENT pm_ev = profiler_domain_.trace_event("plane_map_update");
    depth_maps_.update_at_keyframe(depth_infos, world_from_rig);
  }
  RERUN(logPlanes, depth_maps_.planes(), "world/planes");

  if (sba_service_ && run_sba) {
    TRACE_EVENT sba_ev = profiler_domain_.trace_event("sba_notify");
    TraceMessage("VIO_FIX_SBA frames=%d fixed=%d iterations=%d robust=%.6f", sba_settings.num_inertial_sba_frames,
                 sba_settings.num_inertial_fixed_sba_frames, sba_settings.num_inertial_sba_iterations,
                 sba_settings.inertial_robustifier_scale);
    const auto cov = imu_ && !sba_settings.async ? imu_->joint_position_covariance() : std::nullopt;
    sba_service_->trigger(sba_settings, cov);
  }
}

void SolverSfMMultisensor::exportTracks(const std::vector<camera::Observation>& observations,
                                        std::vector<Track2D>& out_tracks2d, Tracks3DMap& out_tracks3d,
                                        const Isometry3T& rig_from_world) const {
  out_tracks2d.clear();
  out_tracks3d.clear();

  for (const camera::Observation& obs : observations) {
    const camera::ICameraModel& camera = *rig_.intrinsics[obs.cam_id];
    Vector2T uv;
    if (camera.denormalizePoint(obs.xy, uv)) {
      out_tracks2d.push_back({obs.cam_id, obs.id, uv});
    }
  }

  auto map_landmarks = map_.get_recent_landmarks();
  for (const camera::Observation& obs : observations) {
    if (map_landmarks.find(obs.id) != map_landmarks.end()) {
      const Vector3T& point_3d = map_landmarks.at(obs.id);
      out_tracks3d[obs.id] = rig_from_world * point_3d;
    }
  }
}

void SolverSfMMultisensor::add_imu_measurement(const cuvslam::imu::ImuMeasurement& m) {
  if (!imu_) {
    return;
  }
  imu_->add_measurement(m);
}

std::optional<Vector3T> SolverSfMMultisensor::get_gravity() const {
  if (!imu_) {
    return std::nullopt;
  }
  return imu_->get_gravity();
}

std::optional<SolverSfMMultisensor::ImuState> SolverSfMMultisensor::GetImuState() const {
  if (!imu_ || !imu_->has_imu_state()) {
    return std::nullopt;
  }
  return ImuState{imu_->velocity(), imu_->gyro_bias(), imu_->acc_bias()};
}

}  // namespace cuvslam::pipelines
