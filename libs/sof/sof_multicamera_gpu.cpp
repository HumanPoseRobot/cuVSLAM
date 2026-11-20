
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

#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>

#include "common/isometry_utils.h"
#include "sof/internal/calibrated_patch_warp.h"
#include "sof/internal/calibrated_stereo_geometry.h"
#include "sof/internal/sof_multicamera_gpu.h"
#include "sof/internal/stereo_mask.h"
#include "sof/sof_create.h"

namespace cuvslam::sof {

std::unique_ptr<GPULKFeatureTracker> CreateGPUTracker(TrackerType type) {
  switch (type) {
    case TrackerType::LK:
      return std::make_unique<GPULKFeatureTracker>();
    case TrackerType::LKHorizontal:
      return std::make_unique<GPULKTrackerHorizontal>();
    default:
      TraceError("Unsupported GPU tracker type");
      return nullptr;
  }
}

MultiSOFGPU::MultiSOFGPU(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid,
                         FeaturePredictorPtr feature_predictor, const Settings& sof_settings,
                         const odom::KeyFrameSettings& keyframe_settings)
    : MultiSOFBase(rig, fid, sof_settings, keyframe_settings) {
  const auto& primary_cams = fid_.primary_cameras();

  auto tracking_cameras = primary_cams;
  if (std::getenv("CUVSLAM_ONLINE_TEMPORAL")) {
    for (CameraId cam = 0; cam < rig_.num_cameras; ++cam) {
      if (std::find(primary_cams.begin(), primary_cams.end(), cam) == primary_cams.end())
        tracking_cameras.push_back(cam);
    }
  }
  for (CameraId primary_cam_id : tracking_cameras) {
    const camera::ICameraModel& intrinsics = *rig_.intrinsics[primary_cam_id];
    auto selector = std::make_unique<SelectorStereo>(sof_settings.feature_selection_settings);
    mono_sof_.emplace_back(CreateMonoSOF(Implementation::kGPU, primary_cam_id, intrinsics, std::move(selector),
                                         feature_predictor, sof_settings));

    if (std::find(primary_cams.begin(), primary_cams.end(), primary_cam_id) == primary_cams.end()) continue;
    const auto& secondary_cams = fid_.secondary_cameras(primary_cam_id);

    auto& tracker_from_secondary_cam = secondary_from_primary_sof_[primary_cam_id];
    for (CameraId secondary_cam_id : secondary_cams) {
      auto tracker_ptr = CreateGPUTracker(sof_settings.lr_tracker);
      tracker_from_secondary_cam[secondary_cam_id].tracker = std::move(tracker_ptr);
    }
  }
}

void MultiSOFGPU::LaunchTrackingPrimaryToSecondary(CameraId primary_id, CameraId secondary_id,
                                                   const Sources& curr_sources, Images& curr_images, const Sources&,
                                                   const std::vector<camera::Observation>& primary_obs,
                                                   std::vector<camera::Observation>* /* secondary_obs */) {
  const ImageContextPtr primary_image = curr_images[primary_id];
  const ImageSource& secondary_source = curr_sources[secondary_id];
  const ImageContextPtr secondary_image = curr_images[secondary_id];

  const camera::ICameraModel& intrinsicsP = *rig_.intrinsics[primary_id];
  const camera::ICameraModel& intrinsicsS = *rig_.intrinsics[secondary_id];

  PrimaryToSecondaryGPUTracker& tracker = secondary_from_primary_sof_[primary_id][secondary_id];

  GPUArrayPinned<TrackData>& tracks_data = tracker.tracks_data;
  Stream& stream = tracker.stream;

  // Grow all peer buffers before writing host observations. copy_top_n checks
  // its range too late to protect these host writes in a release build.
  if (primary_obs.size() > size_t(std::numeric_limits<int>::max())) {
    throw std::length_error("stereo observation index exceeds int range");
  }
  if (primary_obs.size() > tracks_data.size()) {
    CUDA_CHECK(cudaStreamSynchronize(stream.get_stream()));
    tracks_data.resize(primary_obs.size());
    tracker.calibrated_tracks.resize(primary_obs.size());
    tracker.calibrated_indices.resize(primary_obs.size());
  }

  const Isometry3T secondary_from_primary =
      rig_.camera_from_rig[secondary_id] * rig_.camera_from_rig[primary_id].inverse();

  const float baseline = secondary_from_primary.translation().norm();
  const float avg_focal = 0.5f * (intrinsicsS.getFocal().x() + intrinsicsS.getFocal().y());
  const float cross_cam_search_radius = std::max(20.f, baseline * avg_focal * 2.f);

  // Choose the rotation-aware model from the calibrated CAMERA basis before
  // differentiating the fisheye projection. A noisy/undefined local inverse
  // near the image horizon is not evidence that parallel camera axes reversed.
  const Matrix2T camera_axes = secondary_from_primary.linear().topLeftCorner<2, 2>();
  const bool calibrated = std::getenv("VIO_STEREO_PATCH_WARP") &&
                          (!std::getenv("VIO_STEREO_LK_ORIENTATION_ONLY") || !TranslationPatchCompatible(camera_axes));
  tracker.calibrated_count = 0;
  tracker.calibrated_lookup.assign(primary_obs.size(), -1);
  for (size_t i = 0; i < primary_obs.size(); i++) {
    const camera::Observation& trackL = primary_obs[i];
    const Vector2T& xyL = trackL.xy;

    TrackData& data = tracks_data[i];
    data.track = {0.f, 0.f};
    data.offset = {0.f, 0.f};
    data.track_status = false;
    data.initial_guess_map = {1, 0, 0, 1};
    data.search_radius_px = -1.f;
    data.ncc_threshold = 0.8f;
    Vector2T uvL;
    if (!intrinsicsP.denormalizePoint(xyL, uvL) || !uvL.allFinite()) {
      continue;
    }

    const Vector3T ray_in_secondary = secondary_from_primary.linear() * xyL.homogeneous();
    const float z = ray_in_secondary.z();

    data.track = {uvL.x(), uvL.y()};
    if (z < 1e-8) {
      continue;
    }

    const Vector2T xyR_init(ray_in_secondary.x() / z, ray_in_secondary.y() / z);
    Vector2T uvR_init;
    if (!intrinsicsS.denormalizePoint(xyR_init, uvR_init) || !uvR_init.allFinite()) {
      continue;
    }

    const Vector2T offset = uvR_init - uvL;
    data.offset = {offset.x(), offset.y()};
    data.search_radius_px = cross_cam_search_radius;
    if (calibrated) {
      Matrix2T patch_map;
      if (!CalibratedPatchWarp(intrinsicsP, intrinsicsS, secondary_from_primary.linear(), uvL, patch_map)) {
        data.search_radius_px = -1.f;
        continue;
      }
      if (PatchWarpRequired(patch_map)) {
        const size_t index = tracker.calibrated_count++;
        tracker.calibrated_lookup[i] = static_cast<int>(index);
        tracker.calibrated_indices[index] = static_cast<int>(i);
        auto& warped = tracker.calibrated_tracks[index];
        warped = data;
        warped.initial_guess_map = {patch_map(0, 0), patch_map(0, 1), patch_map(1, 0), patch_map(1, 1)};
        if (!std::getenv("VIO_STEREO_PATCH_VALIDATE")) {
          data.search_radius_px = -1.f;  // Full-replacement diagnostic only.
        }
      }
    }
  }

  tracks_data.copy_top_n(ToGPU, primary_obs.size(), stream.get_stream());

  secondary_image->build_gpu_image_pyramid(secondary_source, box_prefilter_, stream.get_stream());
  secondary_image->build_gpu_gradient_pyramid(tracker.calibrated_count == 0, stream.get_stream());

  tracker.tracker->track_points(primary_image->gpu_gradient_pyramid(), secondary_image->gpu_gradient_pyramid(),
                                primary_image->gpu_image_pyramid(), secondary_image->gpu_image_pyramid(), tracks_data,
                                primary_obs.size(), stream.get_stream());
  tracks_data.copy_top_n(ToCPU, primary_obs.size(), stream.get_stream());
  if (tracker.calibrated_count > 0) {
    tracker.calibrated_tracks.copy_top_n(ToGPU, tracker.calibrated_count, stream.get_stream());
    if (std::getenv("VIO_STEREO_PATCH_VALIDATE")) {
      tracker.calibrated_indices.copy_top_n(ToGPU, tracker.calibrated_count, stream.get_stream());
      const auto& source = primary_image->gpu_image_pyramid()[0];
      const auto& target = secondary_image->gpu_image_pyramid()[0];
      CUDA_CHECK(calibrated_patch_check(source.get_texture_filter_linear(), target.get_texture_filter_linear(),
                                        {int(source.cols()), int(source.rows())},
                                        {int(target.cols()), int(target.rows())}, tracks_data.ptr(),
                                        tracker.calibrated_tracks.ptr(), tracker.calibrated_indices.ptr(),
                                        int(tracker.calibrated_count), stream.get_stream()));
    }
    tracker.calibrated_tracker.track_points(primary_image->gpu_gradient_pyramid(),
                                            secondary_image->gpu_gradient_pyramid(), primary_image->gpu_image_pyramid(),
                                            secondary_image->gpu_image_pyramid(), tracker.calibrated_tracks,
                                            tracker.calibrated_count, stream.get_stream());
    tracker.calibrated_tracks.copy_top_n(ToCPU, tracker.calibrated_count, stream.get_stream());
  }
  tracker.was_launched = true;
}

void MultiSOFGPU::GetTrackingResults(MulticamObservations& observations, const Sources& masks_sources,
                                     const Images& images) {
  std::vector<std::vector<camera::Observation>> secondary_observations(observations.size());
  const auto& primary_cams = fid_.primary_cameras();

  for (CameraId primary_id : primary_cams) {
    if (primary_id >= observations.size()) {
      continue;
    }
    const std::vector<camera::Observation>& primary_obs = observations[primary_id];
    const auto& secondary_cams = fid_.secondary_cameras(primary_id);

    for (CameraId secondary_id : secondary_cams) {
      PrimaryToSecondaryGPUTracker& tracker = secondary_from_primary_sof_[primary_id][secondary_id];
      if (!tracker.was_launched) {
        continue;
      }

      GPUArrayPinned<TrackData>& tracks_data = tracker.tracks_data;
      Stream& stream = tracker.stream;

      const camera::ICameraModel& intrinsicsS = *rig_.intrinsics[secondary_id];

      cudaStreamSynchronize(stream.get_stream());
      const StereoMask target(masks_sources[secondary_id], images[secondary_id]->get_image_meta());

      Matrix2T info;
      Vector2T xyR;
      Vector2T uvR;
      // secondary observations can be added, so only iterate over primary observations
      // FIXME: tracks_data.size() should return data size and not capacity, fix GPUArray[Pinned] and get rid of
      // obs_sizes
      for (size_t i = 0; i < primary_obs.size(); i++) {
        const int calibrated_index = i < tracker.calibrated_lookup.size() ? tracker.calibrated_lookup[i] : -1;
        const bool use_calibrated =
            calibrated_index >= 0 && tracker.calibrated_tracks[calibrated_index].cache_index != -2;
        auto& data = use_calibrated ? tracker.calibrated_tracks[calibrated_index] : tracks_data[i];
        // Diagnostic separation: retain a calibrated-NCC-verified ORIGINAL
        // observation, but never introduce a newly searched observation.
        if (use_calibrated && std::getenv("VIO_STEREO_VERIFY_ONLY")) {
          continue;
        }
        const TrackId& trackId = primary_obs[i].id;
        if (data.track_status) {
          uvR << data.track.x, data.track.y;
          info << data.info[0], data.info[1], data.info[2], data.info[3];
          Matrix2T info_xy;
          if (target.accepts(uvR) && camera::TryObservationInfoUVToXY(intrinsicsS, uvR, info, xyR, info_xy)) {
            if (use_calibrated && std::getenv("VIO_STEREO_RETRY_GEOMETRY")) {
              const Isometry3T secondary_from_primary =
                  rig_.camera_from_rig[secondary_id] * rig_.camera_from_rig[primary_id].inverse();
              if (!ConsistentCalibratedStereo(secondary_from_primary, primary_obs[i].xy, primary_obs[i].xy_info, xyR,
                                              info_xy)) {
                continue;
              }
            }
            secondary_observations[secondary_id].emplace_back(secondary_id, trackId, xyR, info_xy);
          }
        }
      }
    }
  }

  for (CameraId cam_id = 0; cam_id < secondary_observations.size(); ++cam_id) {
    auto& obs_vector = secondary_observations[cam_id];
    auto& x = observations[cam_id];
    std::move(obs_vector.begin(), obs_vector.end(), std::back_inserter(x));
  }
}

void MultiSOFGPU::StartKeyframe() {
  for (auto& [_, x] : secondary_from_primary_sof_) {
    for (auto& [cam_id, tracker] : x) {
      tracker.was_launched = false;
    }
  }
}

void MultiSOFGPU::reset() {
  for (const auto& sof : mono_sof_) {
    sof->reset();
  }

  reset_keyframe_selector();
}

}  // namespace cuvslam::sof
