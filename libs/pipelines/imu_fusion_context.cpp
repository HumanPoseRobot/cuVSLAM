
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

#include "pipelines/imu_fusion_context.h"
#include "imu/world_velocity.h"
#include "imu/soft_inertial_pnp.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string_view>

#include "common/include_eigen.h"
#include "common/vector_2t.h"
#include "epipolar/camera_selection.h"

#include "pipelines/service_sba.h"

namespace cuvslam::pipelines {

using Vector15T = Eigen::Matrix<float, 15, 1>;

namespace {

// Reports whether the IMU stream covers the frame interval reliably. Counts samples in the
// preintegration vs the nominal expected count (calib.frequency * dT), absorbing a small jitter
// margin so normal sub-sample phase offsets don't flag as drops. The previous check truncated the
// expected count to int and divided by it, so on short frame intervals (e.g. 100 Hz IMU with
// 30 fps frames, expected count ~3) a single sample landing just outside [start_ts, end_ts] read
// as a 33% drop — even though the IMU stream was healthy.
bool check_imu_drops(sba_imu::IMUPreintegration& preint, const imu::ImuCalibration& calib, int64_t start_time_ns,
                     int64_t end_time_ns) {
  // First frame: prev_pose_ts_ns is sentinel (-1); no integration interval to validate.
  if (start_time_ns < 0 || end_time_ns <= start_time_ns) {
    return true;
  }
  const float freq = calib.frequency();
  if (freq <= 0.f) {
    return true;
  }
  const float dT_s = static_cast<float>((end_time_ns - start_time_ns) * 1e-9f);
  const float expected = freq * dT_s;
  const float actual = static_cast<float>(preint.size());
  // Allow up to ~2 samples of slack for phase/jitter (IMU timestamps are rarely periodic to within
  // a fraction of the nominal period — see e.g. multi_rgbd indoor where sample gaps span 5–20 ms
  // around a 10 ms nominal). Bound the slack by half the expected count so short intervals can't
  // hide a genuine dropout (otherwise e.g. expected=2 with actual=0 would still report healthy).
  constexpr float kJitterMarginSamples = 2.f;
  const float effective_jitter = std::min(kJitterMarginSamples, expected * 0.5f);
  const float missing = std::max(0.f, expected - actual - effective_jitter);
  const float drop_ratio = missing / expected;
  if (drop_ratio > 0.1f) {
    TraceWarning("Lost IMU msgs: %d, Frame time delta = %f [s], drop ratio = %f [%]", static_cast<int>(missing + 0.5f),
                 dT_s, drop_ratio * 100);
    return false;
  }
  return true;
}

// Prior in the solver's body rotation/translation, world velocity, and IMU bias coordinates.
// Re-seed after visual bootstrap or an external keyframe correction without reusing stale cross terms.
void seed_joint_prior(sba_imu::Pose& pose) {
  Vector15T information;
  information << 1e4f, 1e4f, 1e4f, 1e4f, 1e4f, 1e4f, 25.f, 25.f, 25.f,
                 1e4f, 1e4f, 1e4f, 25.f, 25.f, 25.f;
  pose.info = information.asDiagonal();
}

bool solve_joint(const imu::ImuCalibration& calib, const camera::Rig& rig,
                 const std::unordered_map<TrackId, Vector3T>& landmarks,
                 const std::vector<camera::Observation>& observations, const Vector3T& gravity,
                 const InertialPnPSettings& settings, sba_imu::Pose left, bool have_prior,
                 sba_imu::Pose& result, int64_t time_ns) {
  if (!have_prior) seed_joint_prior(left);
  sba_imu::Pose right;
  if (!left.predict_pose(gravity, left.preintegration, right)) return false;
  const auto prediction = right;
  sba_imu::StereoPnPInput input;
  input.rig = rig;
  input.gravity = gravity;
  input.robustifier_scale = settings.robustifier_scale;
  input.robustifier_scale_pose = -1.f;
  input.max_iterations = settings.max_iteration;
  input.prior_gyro = input.prior_acc = input.translation_constraint = 0.f;
  input.imu_penalty = 1e-3f;
  input.min_depth = 0.05f;
  input.shared_bias = true;
  std::array<int, 4> counts{};
  const Isometry3T rig_from_world = calib.rig_from_imu() * right.w_from_imu.inverse();
  for (const auto& obs : observations) {
    const auto it = landmarks.find(obs.id);
    if (it == landmarks.end() || obs.cam_id >= rig.num_cameras) continue;
    const Vector3T point = rig.camera_from_rig[obs.cam_id] * rig_from_world * it->second;
    if (!(point.z() > input.min_depth)) continue;
    ++counts.at(obs.cam_id);
    input.point_ids.push_back(static_cast<int>(input.points.size()));
    input.points.push_back(it->second);
    input.camera_ids.push_back(obs.cam_id);
    input.observation_xys.push_back(obs.xy);
    input.observation_infos.push_back(9.f * obs.xy_info);
  }
  const bool enough = settings.min_observations > 0 && settings.max_iteration > 0 &&
                      input.points.size() >= static_cast<size_t>(settings.min_observations);
  const bool ok = enough && SoftInertialPnP(calib, input, left, right, input.imu_penalty) &&
                  right.w_from_imu.matrix().allFinite() && right.velocity.allFinite() &&
                  right.gyro_bias.allFinite() && right.acc_bias.allFinite() && right.info.allFinite();
  TraceMessage("JOINT_FRAME t=%lld ok=%d cams=%d,%d,%d,%d dp=%.8f dv=%.8f dbg=%.8f dba=%.8f",
               static_cast<long long>(time_ns), static_cast<int>(ok), counts[0], counts[1], counts[2], counts[3],
               (right.w_from_imu.translation() - prediction.w_from_imu.translation()).norm(),
               (right.velocity - prediction.velocity).norm(), (right.gyro_bias - prediction.gyro_bias).norm(),
               (right.acc_bias - prediction.acc_bias).norm());
  if (ok) result = std::move(right);
  return ok;
}

// Run the pose-only cuNLS estimator with an IMU-predicted SE3 prior.
// Velocity and biases are constants in this factor; the caller updates velocity
// from world-frame position differences after solving the current pose.
bool run_rgbd_inertial_solve(const pnp::MultisensorPoseEstimator& estimator, const imu::ImuCalibration& calib,
                             const std::unordered_map<TrackId, Vector3T>& landmarks,
                             const std::vector<camera::Observation>& observations, const pnp::RGBDInfos& depth_infos,
                             const std::vector<map::Plane>& planes, const std::vector<Vector3T>& depth_points,
                             const Vector3T& gravity, sba_imu::Pose& prev_pose, sba_imu::Pose& curr_pose) {
  // Seed the LM with an IMU-integrated pose prediction for fast convergence.
  sba_imu::Pose predicted = curr_pose;
  if (!prev_pose.predict_pose(gravity, prev_pose.preintegration, predicted)) {
    return false;
  }
  const Isometry3T imu_from_rig = calib.rig_from_imu().inverse();
  Isometry3T rig_from_world = (predicted.w_from_imu * imu_from_rig).inverse();
  Matrix6T info;

  pnp::InertialPriorInput imu_in;
  imu_in.prev_rig_from_world = (prev_pose.w_from_imu * imu_from_rig).inverse();
  imu_in.prev_velocity = prev_pose.velocity;
  imu_in.prev_gyro_bias = prev_pose.gyro_bias;
  imu_in.prev_acc_bias = prev_pose.acc_bias;
  imu_in.preint = &prev_pose.preintegration;
  imu_in.gravity_world = gravity;
  imu_in.rig_from_imu = calib.rig_from_imu();

  pnp::InertialPosteriorOutput imu_out;
  const bool ok = estimator.solve(rig_from_world, info, observations, landmarks, depth_infos, planes, depth_points,
                                  imu_in, &imu_out);
  if (!ok) {
    return false;
  }

  curr_pose.w_from_imu = rig_from_world.inverse() * imu_from_rig.inverse();
  curr_pose.velocity = imu_out.velocity;
  curr_pose.gyro_bias = imu_out.gyro_bias;
  curr_pose.acc_bias = imu_out.acc_bias;
  curr_pose.info.setZero();
  curr_pose.info.block<6, 6>(0, 0) = info;
  return true;
}

}  // namespace

ImuFusionContext::ImuFusionContext(map::UnifiedMap& map, const camera::Rig& rig, const imu::ImuCalibration& calib,
                                   const pnp::MultisensorPoseEstimator& pose_estimator)
    : map_(map),
      rig_(rig),
      calib_(calib),
      pose_estimator_(pose_estimator),
      imu_storage_(1e5),
      optimizer_(10),
      init_bundler_(calib),
      sm_() {
  const char* option = std::getenv("CUVSLAM_JOINT_VI");
  joint_enabled_ = rig.num_cameras == 4 && option && std::string_view(option) == "1";
  prev_pose_.w_from_imu = calib_.rig_from_imu();
}

void ImuFusionContext::register_gravity_callback() {
  sm_.set_gravity_init_fn([this](size_t num_kfs) { return gravity_estimation_callback(num_kfs); });
}

void ImuFusionContext::reset() {
  imu_storage_.clear();
  sm_.reset();
  joint_active_ = has_joint_prior_ = false;
  is_first_run_ = true;
  integrated_ = false;
  prev_pose_ts_ns_ = -1;
  last_kf_ts_ns_ = -1;
  last_valid_ts_ns_ = -1;

  prev_pose_ = sba_imu::Pose{};
  prev_pose_.w_from_imu = calib_.rig_from_imu();
  curr_pose_ = sba_imu::Pose{};
  last_valid_pose_ = sba_imu::Pose{};
  integ_kf_ = sba_imu::Pose{};

  curr_pose_.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  prev_pose_.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  last_valid_pose_.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  integ_kf_.preintegration.Initialize(Vector3T::Zero(), Vector3T::Zero());
  last_kf_preint_.Initialize(Vector3T::Zero(), Vector3T::Zero());
  last_frame_preint_.Initialize(Vector3T::Zero(), Vector3T::Zero());
}

void ImuFusionContext::add_measurement(const imu::ImuMeasurement& m) {
  if (!imu_storage_.empty() && m.time_ns <= imu_storage_.back().time_ns) {
    TraceWarning("Ignoring non-increasing IMU timestamp");
    return;
  }
  imu_storage_.push_back(m);
}

ImuFusionContext::FramePrep ImuFusionContext::prepare_frame(int64_t time_ns) {
  if (is_first_run_) {
    Vector15T diag = Vector15T::Zero();
    diag.segment<6>(0).setConstant(1e6);
    prev_pose_.info = diag.asDiagonal();

    last_valid_pose_ = prev_pose_;
    integ_kf_ = prev_pose_;

    last_valid_pose_.preintegration = last_frame_preint_;
    integ_kf_.preintegration = last_frame_preint_;
    last_kf_preint_ = last_frame_preint_;
  }

  auto integrate_from = [&](int64_t start, const sba_imu::Pose& pose) {
    return sba_imu::IMUPreintegration(calib_, imu_storage_, pose.gyro_bias, pose.acc_bias, start < 0 ? time_ns : start,
                                      time_ns);
  };
  last_frame_preint_ = integrate_from(prev_pose_ts_ns_, prev_pose_);
  last_kf_preint_ = integrate_from(last_kf_ts_ns_, joint_active_ ? integ_kf_ : prev_pose_);
  last_valid_pose_.preintegration = integrate_from(last_valid_ts_ns_, last_valid_pose_);
  prev_pose_.preintegration = last_frame_preint_;
  curr_pose_ = prev_pose_;

  FramePrep prep;
  prep.imu_state = sm_.state();
  prep.predicted_world_from_rig = prev_pose_.w_from_imu * calib_.rig_from_imu().inverse();
  prep.no_drops = check_imu_drops(prev_pose_.preintegration, calib_, prev_pose_ts_ns_, time_ns);
  if (prev_pose_ts_ns_ >= 0 && time_ns > prev_pose_ts_ns_) {
    const float expected_dt = static_cast<float>((time_ns - prev_pose_ts_ns_) * 1e-9);
    prep.no_drops = prep.no_drops && std::abs(last_frame_preint_.dT_s - expected_dt) < 1e-6f;
    TraceMessage("VIO_FIX_INTERVAL start=%lld end=%lld camera_dt=%.9f integrated_dt=%.9f segments=%zu healthy=%d",
                 static_cast<long long>(prev_pose_ts_ns_), static_cast<long long>(time_ns), expected_dt,
                 last_frame_preint_.dT_s, last_frame_preint_.size(), static_cast<int>(prep.no_drops));
  }
  return prep;
}

bool ImuFusionContext::solve_inertial(const std::unordered_map<TrackId, Vector3T>& landmarks,
                                      const std::vector<camera::Observation>& observations,
                                      const pnp::RGBDInfos& depth_infos, const std::vector<map::Plane>& planes,
                                      const std::vector<Vector3T>& depth_points, int64_t time_ns,
                                      const InertialPnPSettings& settings, bool interval_healthy) {
  std::optional<Vector3T> maybe_gravity = map_.get_gravity();
  if (!maybe_gravity) {
    return false;
  }

  TRACE_EVENT ev = profiler_domain_.trace_event("solve_inertial");
  joint_active_ = joint_enabled_ && planes.empty() && depth_points.empty() &&
                  std::none_of(depth_infos.begin(), depth_infos.end(), [](auto p) { return p != nullptr; });
  if (joint_active_) {
    has_joint_prior_ = interval_healthy && solve_joint(calib_, rig_, landmarks, observations, *maybe_gravity,
                                                       settings, prev_pose_, has_joint_prior_, curr_pose_, time_ns);
    if (has_joint_prior_) return true;
    TraceMessage("JOINT_VISUAL_FALLBACK t=%lld", static_cast<long long>(time_ns));
    return solve_visual(landmarks, observations, depth_infos, planes, depth_points, time_ns);
  }
  has_joint_prior_ = false;
  // The cuNLS LM uses cunls::SE3BetweenFactorBatch with an IMU-derived Delta; velocity / biases
  // are baked in as constants in the Delta, so the solve does not refine them.  Pose is the only
  // state the LM moves.
  if (!run_rgbd_inertial_solve(pose_estimator_, calib_, landmarks, observations, depth_infos, planes, depth_points,
                               *maybe_gravity, prev_pose_, curr_pose_)) {
    return false;
  }
  if (!is_first_run_ && prev_pose_ts_ns_ >= 0) {
    float dt = static_cast<float>((time_ns - prev_pose_ts_ns_) * 1e-9);
    if (dt > 1e-6f) {
      curr_pose_.velocity = sba_imu::WorldVelocity(prev_pose_.w_from_imu, curr_pose_.w_from_imu, dt);
    }
  } else {
    curr_pose_.velocity.setZero();
  }
  return true;
}

bool ImuFusionContext::solve_visual(const std::unordered_map<TrackId, Vector3T>& landmarks,
                                    const std::vector<camera::Observation>& observations,
                                    const pnp::RGBDInfos& depth_infos, const std::vector<map::Plane>& planes,
                                    const std::vector<Vector3T>& depth_points, int64_t time_ns) {
  has_joint_prior_ = false;
  TRACE_EVENT ev = profiler_domain_.trace_event("solve_visual");
  const Isometry3T imu_from_rig = calib_.rig_from_imu().inverse();
  Isometry3T rig_from_world = (prev_pose_.w_from_imu * imu_from_rig).inverse();
  Matrix6T info;
  const bool ok =
      pose_estimator_.solve(rig_from_world, info, observations, landmarks, depth_infos, planes, depth_points);
  if (!ok) {
    return false;
  }

  curr_pose_.w_from_imu = rig_from_world.inverse() * imu_from_rig.inverse();
  curr_pose_.info.setZero();
  curr_pose_.info.block<6, 6>(0, 0) = info;

  curr_pose_.gyro_bias = prev_pose_.gyro_bias;
  curr_pose_.acc_bias = prev_pose_.acc_bias;
  if (!is_first_run_ && prev_pose_ts_ns_ >= 0) {
    float dt = static_cast<float>((time_ns - prev_pose_ts_ns_) * 1e-9);
    if (dt > 1e-6f) {
      curr_pose_.velocity = sba_imu::WorldVelocity(prev_pose_.w_from_imu, curr_pose_.w_from_imu, dt);
    }
  } else {
    curr_pose_.velocity.setZero();
  }
  return true;
}

ImuFusionContext::FrameResult ImuFusionContext::finalize_frame(bool pnp_result, bool no_drops, int64_t time_ns,
                                                               sof::FrameState fs,
                                                               const StateMachineSettings& sm_settings) {
  prev_pose_ts_ns_ = time_ns;

  StateMachine::State imu_state;
  {
    TRACE_EVENT ev = profiler_domain_.trace_event("update_frame_state");
    bool is_keyframe = fs == sof::FrameState::Key;
    StateMachine::Event event;
    if (!no_drops) {
      // IMU samples were lost across this frame — preintegration is unreliable.
      event = StateMachine::Event::FrameDropped;
    } else {
      event = pnp_result ? StateMachine::Event::FrameOk : StateMachine::Event::FrameFailed;
    }
    imu_state = sm_.on_event(event, time_ns, is_keyframe, sm_settings);
  }

  if (pnp_result && imu_state == StateMachine::State::Ok) {
    last_valid_ts_ns_ = time_ns;
    last_valid_pose_ = curr_pose_;
    last_valid_pose_.preintegration = sba_imu::IMUPreintegration(curr_pose_.gyro_bias, curr_pose_.acc_bias);
  }

  integrated_ = !pnp_result && imu_state == StateMachine::State::Ok;

  // Re-anchor integ_kf_ to the latest keyframe so IMU integration always starts from a known-good
  // visual anchor instead of accumulating frame-to-frame error.
  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  if (!map_.empty()) {
    auto recent_map = map_.get_recent_submap(1);
    if (!recent_map.consecutive_keyframes.empty()) {
      auto it = recent_map.consecutive_keyframes.rbegin();
      State s = it->keyframe->get_state();
      sba_imu::IMUPreintegration preint(calib_, imu_storage_, s.gyro_bias, s.acc_bias, it->keyframe->time_ns(),
                                        time_ns);
      integ_kf_ = {s.rig_from_w.inverse() * rig_from_imu, s.velocity, s.gyro_bias, s.acc_bias, preint};
    }
  }

  if (integrated_) {
    std::optional<Vector3T> maybe_gravity = map_.get_gravity();
    if (maybe_gravity) {
      integ_kf_.predict_pose(*maybe_gravity, integ_kf_.preintegration, curr_pose_);
    } else {
      integrated_ = false;
    }
  }

  FrameResult result;
  result.valid_pose = pnp_result || imu_state == StateMachine::State::Ok;
  if (result.valid_pose) {
    const Isometry3T imu_from_rig = rig_from_imu.inverse();
    result.world_from_rig = curr_pose_.w_from_imu * imu_from_rig;
    result.static_info_exp = curr_pose_.info.block<6, 6>(0, 0);
    std::swap(prev_pose_, curr_pose_);
  } else {
    result.world_from_rig = prev_pose_.w_from_imu * rig_from_imu.inverse();
  }

  // Pull SBA-refined biases back into prev_pose so the next frame's PnP starts from the latest
  // estimate. A joint posterior already contains its own bias estimate.
  if (!joint_active_ && !map_.empty()) {
    auto recent_map = map_.get_recent_submap(1);
    if (!recent_map.consecutive_keyframes.empty()) {
      State s = recent_map.consecutive_keyframes.rbegin()->keyframe->get_state();
      prev_pose_.gyro_bias = s.gyro_bias;
      prev_pose_.acc_bias = s.acc_bias;
    }
  }

  last_frame_preint_ = sba_imu::IMUPreintegration(prev_pose_.gyro_bias, prev_pose_.acc_bias);

  result.lost = !is_first_run_ && !pnp_result && !integrated_;
  if (!result.lost && is_first_run_) {
    is_first_run_ = false;
  }
  return result;
}

map::State ImuFusionContext::build_keyframe_state(const Isometry3T& rig_from_world, bool lost) const {
  const sba_imu::Pose& pose = lost ? last_valid_pose_ : prev_pose_;
  return map::State{rig_from_world, pose.velocity, pose.acc_bias, pose.gyro_bias};
}

void ImuFusionContext::on_keyframe_committed(bool lost) {
  last_kf_ts_ns_ = prev_pose_ts_ns_;
  if (joint_active_) integ_kf_ = prev_pose_;
  const sba_imu::Pose& pose = lost ? last_valid_pose_ : prev_pose_;
  last_kf_preint_ = sba_imu::IMUPreintegration(pose.gyro_bias, pose.acc_bias);
}

std::optional<Matrix3T> ImuFusionContext::joint_position_covariance() const {
  if (!joint_active_ || !has_joint_prior_) return std::nullopt;
  Eigen::MatrixXd info = prev_pose_.info.cast<double>();
  Eigen::LDLT<Eigen::MatrixXd> ldlt(info);
  if (ldlt.info() != Eigen::Success || ldlt.vectorD().minCoeff() <= 0.) return std::nullopt;
  Eigen::MatrixXd selector = Eigen::MatrixXd::Zero(15,3);
  selector.block<3,3>(3,0).setIdentity();
  Eigen::Matrix3d cov = ldlt.solve(selector).block<3,3>(3,0);
  const Eigen::Matrix3d rot = prev_pose_.w_from_imu.linear().cast<double>();
  cov = rot * cov * rot.transpose();
  if (!cov.allFinite()) return std::nullopt;
  return cov.cast<float>();
}

bool ImuFusionContext::read_committed_state(int64_t time_ns, Isometry3T& world_from_rig, Matrix6T& info) {
  if (!joint_active_ || prev_pose_ts_ns_ != time_ns) return false;
  const auto recent = map_.get_recent_submap(1);
  if (recent.consecutive_keyframes.empty()) return false;
  const auto& key = recent.consecutive_keyframes.back().keyframe;
  if (key->time_ns() != time_ns) return false;
  const auto state = key->get_state();
  const Isometry3T corrected = state.rig_from_w.inverse();
  const Isometry3T corrected_imu = corrected * calib_.rig_from_imu();
  if (!corrected.matrix().allFinite() || !state.velocity.allFinite() ||
      !state.gyro_bias.allFinite() || !state.acc_bias.allFinite()) return false;
  const float dp = (world_from_rig.translation() - corrected.translation()).norm();
  const bool changed = !prev_pose_.w_from_imu.matrix().isApprox(corrected_imu.matrix(), 1e-6f) ||
                       !prev_pose_.velocity.isApprox(state.velocity, 1e-6f) ||
                       !prev_pose_.gyro_bias.isApprox(state.gyro_bias, 1e-6f) ||
                       !prev_pose_.acc_bias.isApprox(state.acc_bias, 1e-6f);
  if (!changed) return false;
  prev_pose_.w_from_imu = corrected_imu;
  prev_pose_.velocity = state.velocity;
  prev_pose_.gyro_bias = state.gyro_bias;
  prev_pose_.acc_bias = state.acc_bias;
  seed_joint_prior(prev_pose_);
  has_joint_prior_ = false;
  if (last_valid_ts_ns_ == time_ns) last_valid_pose_ = prev_pose_;
  info = prev_pose_.info.topLeftCorner<6, 6>();
  world_from_rig = corrected;
  TraceMessage("JOINT_KEYFRAME t=%lld dp=%.8f", static_cast<long long>(time_ns), dp);
  return true;
}

std::optional<Vector3T> ImuFusionContext::get_gravity() const {
  auto gravity = map_.get_gravity();
  if (!gravity) {
    return std::nullopt;
  }
  Isometry3T rig_from_w = calib_.rig_from_imu() * prev_pose_.w_from_imu.inverse();
  return rig_from_w.linear() * (*gravity);
}

void ImuFusionContext::snap_to_keyframe_pose(const sba_imu::Pose& last_kf_pose) {
  auto apply = [&](sba_imu::Pose& pose) {
    pose.velocity = last_kf_pose.velocity;
    pose.gyro_bias = last_kf_pose.gyro_bias;
    pose.acc_bias = last_kf_pose.acc_bias;
    pose.preintegration.SetNewBias(last_kf_pose.gyro_bias, last_kf_pose.acc_bias);
    pose.preintegration.Reintegrate(calib_);
  };
  apply(curr_pose_);
  apply(prev_pose_);
  apply(integ_kf_);
  apply(last_valid_pose_);
}

bool ImuFusionContext::gravity_estimation_callback(size_t num_kfs) {
  TRACE_EVENT ev = profiler_domain_.trace_event("gravity_estimation_callback");
  auto recent_map = map_.get_recent_submap(map_.capacity());

  if (recent_map.consecutive_keyframes.size() < num_kfs) {
    return false;
  }

  const Isometry3T& rig_from_imu = calib_.rig_from_imu();
  Matrix3T Rgravity = Matrix3T::Identity();
  std::vector<sba_imu::Pose> gravity_cache;
  for (const auto& kf : recent_map.consecutive_keyframes) {
    State s = kf.keyframe->get_state();
    gravity_cache.push_back({s.rig_from_w.inverse() * rig_from_imu, s.velocity, s.gyro_bias, s.acc_bias,
                             kf.preintegration ? *kf.preintegration : sba_imu::IMUPreintegration()});
  }

  std::vector<std::vector<camera::Observation>> observations_per_kf;
  observations_per_kf.reserve(recent_map.consecutive_keyframes.size());
  for (const auto& lm_obs : recent_map.landmark_and_obs) {
    std::vector<camera::Observation> kf_obs;
    for (const auto& lo : lm_obs) {
      for (const auto& obs : lo.observations) {
        kf_obs.push_back(obs);
      }
    }
    observations_per_kf.push_back(std::move(kf_obs));
  }

  // optimize_inertial already does the staged init: SolveGyroBias (gyro-bias first), then
  // LinearAlignment (joint velocities + gravity + acc_bias via a linear system), then
  // RefineGravity (tangent-space gravity + acc_bias refinement), then LM polish. The
  // OptimizeInertialAdaptive variant is preferred for stereo+IMU because of its post-opt
  // rotation-from-IMU + epipolar-translation refinement, but that refinement consistently
  // regresses the multisensor path (verified empirically on EUROC V2_03), so we stay on
  // optimize_inertial here.
  if (!optimizer_.optimize_inertial(gravity_cache, calib_, Rgravity, 1e-3)) {
    return false;
  }

  TRACE_EVENT ev1 = profiler_domain_.trace_event("update map");
  const Vector3T gravity_for_sba = Rgravity * optimizer_.get_default_gravity();

  if (enable_imu_sba_init_) {
    run_imu_sba_init(recent_map, gravity_for_sba, gravity_cache, observations_per_kf);
  }

  // Write back per-keyframe biases / velocities / preintegrations from the refined cache.
  int id = 0;
  for (const auto& kf : recent_map.consecutive_keyframes) {
    const sba_imu::Pose& pose = gravity_cache[id];
    kf.keyframe->set_gyro_bias(pose.gyro_bias);
    kf.keyframe->set_acc_bias(pose.acc_bias);
    kf.keyframe->set_velocity(pose.velocity);
    kf.keyframe->set_pose((pose.w_from_imu * rig_from_imu.inverse()).inverse());
    if (kf.preintegration) {
      kf.preintegration->SetNewBias(pose.gyro_bias, pose.acc_bias);
      kf.preintegration->Reintegrate(calib_);
    }
    id++;
  }

  // Snap live tracking state to the last keyframe pose (closest temporal match for the next
  // tracking frame).
  snap_to_keyframe_pose(gravity_cache.back());

  map_.set_gravity(gravity_for_sba);
  return true;
}

void ImuFusionContext::run_imu_sba_init(const map::UnifiedMap::SubMap& recent_map, const Vector3T& gravity,
                                        std::vector<sba_imu::Pose>& gravity_cache,
                                        const std::vector<std::vector<camera::Observation>>& observations_per_kf) {
  const size_t n = recent_map.consecutive_keyframes.size();
  if (gravity_cache.size() != n || observations_per_kf.size() != n) {
    return;
  }

  sba_imu::ImuBAProblem problem;
  problem.rig = rig_;
  problem.gravity = gravity;
  problem.robustifier_scale = 1.5f;
  problem.robustifier_scale_pose = -1.0f;
  problem.prior_acc = 1e1;
  problem.prior_gyro = 0;
  problem.imu_penalty = 1e-2f;
  problem.boundary_imu_penalty = 1e-2f;
  problem.acc_rw_penalty = 0.1f;

  constexpr float visual_noise_px = 1.0f;
  constexpr float info_scale = (3.0f / visual_noise_px) * (3.0f / visual_noise_px);
  problem.reintegration_thresh = 1e-3f;

  problem.max_iterations = 10;
  problem.num_fixed_key_frames = CalcNumFixedKeyframes(n, 1);
  if (problem.num_fixed_key_frames < 1) {
    return;
  }

  std::vector<std::vector<Isometry3T>> world_from_cam(n, std::vector<Isometry3T>(rig_.num_cameras));
  for (size_t i = 0; i < n; i++) {
    Isometry3T world_from_rig_i = gravity_cache[i].w_from_imu * calib_.rig_from_imu().inverse();
    for (CameraId cam_id = 0; cam_id < static_cast<CameraId>(rig_.num_cameras); cam_id++) {
      world_from_cam[i][cam_id] = world_from_rig_i * rig_.camera_from_rig[cam_id].inverse();
    }
  }

  struct TrackObs {
    size_t kf_idx;
    CameraId cam_id;
    Vector2T xy;
    Matrix2T xy_info;
  };
  std::unordered_map<TrackId, std::vector<TrackObs>> track_obs_map;
  for (size_t i = 0; i < n; i++) {
    for (const auto& obs : observations_per_kf[i]) {
      track_obs_map[obs.id].push_back({i, obs.cam_id, obs.xy, obs.xy_info});
    }
  }

  std::unordered_map<TrackId, map::LandmarkPtr> track_to_landmark;
  for (size_t i = 0; i < n; i++) {
    for (const auto& lo : recent_map.landmark_and_obs[i]) {
      if (!lo.landmark->get_pose()) {
        continue;
      }
      for (const auto& obs : lo.observations) {
        track_to_landmark[obs.id] = lo.landmark;
      }
    }
  }

  std::unordered_map<TrackId, int> track_point_ids;
  std::unordered_map<map::LandmarkPtr, int> landmark_point_ids;

  for (auto& [track_id, obs_list] : track_obs_map) {
    auto it = track_to_landmark.find(track_id);
    if (it == track_to_landmark.end()) {
      continue;
    }

    bool triangulated = false;
    Vector3T new_point;

    for (size_t a = 0; a < obs_list.size() && !triangulated; a++) {
      for (size_t b = a + 1; b < obs_list.size() && !triangulated; b++) {
        if (obs_list[a].cam_id == obs_list[b].cam_id) {
          continue;
        }

        try {
          const auto& oa = obs_list[a];
          const auto& ob = obs_list[b];
          const Isometry3T& wfc_a = world_from_cam[oa.kf_idx][oa.cam_id];
          const Isometry3T& wfc_b = world_from_cam[ob.kf_idx][ob.cam_id];
          Isometry3T cam_a_from_cam_b = wfc_a.inverse() * wfc_b;

          Vector3T xyz_in_cam_a;
          float pm;
          epipolar::TriangulationState ts;
          if (epipolar::OptimalTriangulation(cam_a_from_cam_b, oa.xy, ob.xy, xyz_in_cam_a, pm, ts)) {
            new_point = wfc_a * xyz_in_cam_a;
            triangulated = true;
          }
        } catch (const std::exception&) {
          continue;
        }
      }
    }

    if (!triangulated) {
      size_t kf_idx = obs_list[0].kf_idx;
      const auto& kf = recent_map.consecutive_keyframes[kf_idx].keyframe;
      Isometry3T old_world_from_rig = kf->get_pose().inverse();
      Isometry3T new_world_from_rig = gravity_cache[kf_idx].w_from_imu * calib_.rig_from_imu().inverse();
      Isometry3T correction = new_world_from_rig * old_world_from_rig.inverse();
      new_point = correction * (*it->second->get_pose());
    }

    int pid = static_cast<int>(problem.points.size());
    track_point_ids[track_id] = pid;
    landmark_point_ids[it->second] = pid;
    problem.points.push_back(new_point);
  }
  if (problem.points.empty()) {
    return;
  }

  for (size_t i = 0; i < n; i++) {
    problem.rig_poses.push_back(gravity_cache[i]);
  }

  size_t total_observations = 0;
  for (const auto& obs_kf : observations_per_kf) {
    total_observations += obs_kf.size();
  }
  problem.observation_xys.reserve(total_observations);
  problem.observation_infos.reserve(total_observations);
  problem.point_ids.reserve(total_observations);
  problem.pose_ids.reserve(total_observations);
  problem.camera_ids.reserve(total_observations);

  for (size_t i = 0; i < n; i++) {
    for (const auto& obs : observations_per_kf[i]) {
      auto it = track_point_ids.find(obs.id);
      if (it == track_point_ids.end()) {
        continue;
      }
      problem.observation_xys.push_back(obs.xy);
      problem.observation_infos.push_back(obs.xy_info * info_scale);
      problem.point_ids.push_back(it->second);
      problem.pose_ids.push_back(static_cast<int32_t>(i));
      problem.camera_ids.push_back(obs.cam_id);
    }
  }

  if (!init_bundler_.solve(problem)) {
    return;
  }

  for (size_t i = 0; i < n; i++) {
    const auto& pose = problem.rig_poses[i];
    gravity_cache[i] = pose;

    const auto& [kf, preint] = recent_map.consecutive_keyframes[i];
    kf->set_state({calib_.rig_from_imu() * pose.w_from_imu.inverse(), pose.velocity, pose.acc_bias, pose.gyro_bias});
    if (preint) {
      *preint = pose.preintegration;
    }
  }
  for (auto& [l, pid] : landmark_point_ids) {
    l->set_pose(problem.points[pid]);
  }
}

}  // namespace cuvslam::pipelines
