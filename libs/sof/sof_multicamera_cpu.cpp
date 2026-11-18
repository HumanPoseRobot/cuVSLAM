
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

#include <cmath>
#include <memory>

#include "common/isometry_utils.h"
#include "sof/internal/sof_multicamera_cpu.h"
#include "sof/sof_create.h"
#include "sof/internal/stereo_mask.h"

namespace cuvslam::sof {

MultiSOFCPU::MultiSOFCPU(const camera::Rig& rig, const camera::FrustumIntersectionGraph& fid,
                         sof::FeaturePredictorPtr feature_predictor, const Settings& sof_settings,
                         const odom::KeyFrameSettings& keyframe_settings)
    : MultiSOFBase(rig, fid, sof_settings, keyframe_settings) {
  const auto& primary_cams = fid_.primary_cameras();

  for (CameraId primary_cam_id : primary_cams) {
    const camera::ICameraModel& intrinsics = *rig_.intrinsics[primary_cam_id];
    auto selector = std::make_unique<SelectorStereo>(sof_settings.feature_selection_settings);
    mono_sof_.emplace_back(CreateMonoSOF(Implementation::kCPU, primary_cam_id, intrinsics, std::move(selector),
                                         feature_predictor, sof_settings));

    const auto& secondary_cams = fid_.secondary_cameras(primary_cam_id);

    auto& tracker_from_secondary_cam = secondary_from_primary_sof_[primary_cam_id];
    for (CameraId secondary_cam_id : secondary_cams) {
      tracker_from_secondary_cam[secondary_cam_id] = CreateTracker(sof_settings.lr_tracker);
    }
  }
}

void MultiSOFCPU::LaunchTrackingPrimaryToSecondary(CameraId primary_id, CameraId secondary_id,
                                                   const Sources& curr_sources, sof::Images& curr_images,
                                                   const Sources& masks_sources, const std::vector<camera::Observation>& primary_obs,
                                                   std::vector<camera::Observation>* secondary_obs) {
  ImageContextPtr primary_image = curr_images[primary_id];
  const ImageSource& secondary_source = curr_sources[secondary_id];
  ImageContextPtr secondary_image = curr_images[secondary_id];

  const camera::ICameraModel& intrinsicsP = *rig_.intrinsics[primary_id];
  const camera::ICameraModel& intrinsicsS = *rig_.intrinsics[secondary_id];

  const std::unique_ptr<IFeatureTracker>& tracker = secondary_from_primary_sof_[primary_id][secondary_id];
  assert(tracker != nullptr);

  const Isometry3T secondary_from_primary =
      rig_.camera_from_rig[secondary_id] * rig_.camera_from_rig[primary_id].inverse();

  const float baseline = secondary_from_primary.translation().norm();
  const float avg_focal = 0.5f * (intrinsicsS.getFocal().x() + intrinsicsS.getFocal().y());
  const float cross_cam_search_radius = std::max(20.f, baseline * avg_focal * 2.f);

  secondary_image->build_cpu_image_pyramid(secondary_source, box_prefilter_);
  secondary_image->build_cpu_gradient_pyramid(tracker->isHorizontal());

  const StereoMask target(masks_sources[secondary_id], secondary_image->get_image_meta());
  for (const camera::Observation& trackL : primary_obs) {
    const TrackId& trackId = trackL.id;
    const Vector2T& xyL = trackL.xy;
    Vector2T uvL;
    if (!intrinsicsP.denormalizePoint(xyL, uvL)) {
      continue;
    }

    const Vector3T ray_in_secondary = secondary_from_primary.linear() * xyL.homogeneous();
    const float z = ray_in_secondary.z();

    if (z < 1e-8) {
      continue;
    }

    const Vector2T xyR_init(ray_in_secondary.x() / z, ray_in_secondary.y() / z);
    Vector2T uvR;
    if (!intrinsicsS.denormalizePoint(xyR_init, uvR) || !uvR.allFinite()) {
      continue;
    }

    Matrix2T info;

    if (tracker->trackPoint(primary_image->cpu_gradient_pyramid(), secondary_image->cpu_gradient_pyramid(),
                            primary_image->cpu_image_pyramid(), secondary_image->cpu_image_pyramid(), uvL, uvR, info,
                            cross_cam_search_radius)) {
      Vector2T xyR;
      Matrix2T info_xy;
      if (secondary_obs && target.accepts(uvR) &&
          camera::TryObservationInfoUVToXY(intrinsicsS, uvR, info, xyR, info_xy)) {
        secondary_obs->emplace_back(secondary_id, trackId, xyR, info_xy);
      }
    }
  }
}

void MultiSOFCPU::GetTrackingResults(MulticamObservations&, const Sources&, const Images&) { return; }

void MultiSOFCPU::StartKeyframe() {}

void MultiSOFCPU::reset() {
  for (auto& sof : mono_sof_) {
    sof->reset();
  }

  reset_keyframe_selector();
}

}  // namespace cuvslam::sof
