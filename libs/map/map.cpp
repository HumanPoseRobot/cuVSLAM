
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

#include "map/map.h"

#include "common/log.h"

namespace cuvslam::map {
using namespace cuvslam::camera;

UnifiedMap::UnifiedMap(size_t capacity) : capacity_(capacity) {
  if (capacity_ == 0) {
    throw std::runtime_error("UnifiedMap capacity cant be zero!");
  }
}

std::deque<UnifiedMap::KeyframeWithPreint> UnifiedMap::get_consecutive_keyframes() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return consecutive_keyframes_;
}

void UnifiedMap::add_keyframe(int64_t time_ns, const State& state, const IMUPreintegration& preint,
                              const std::vector<camera::Observation>& observations,
                              const std::vector<pipelines::Landmark>& triangulated_tracks) {
  TRACE_EVENT ev = profiler_domain_.trace_event("add_keyframe");
  // create new keyframe
  auto new_kf = std::make_shared<KeyFrame>(state, time_ns);

  FastMap<TrackId, LandmarkAndObserv> new_landmarks;

  std::lock_guard<std::mutex> lock(map_mutex_);

  for (const auto& [track_id, point] : triangulated_tracks) {
    auto it = keyframes_from_landmark_.find(track_id);
    if (it != keyframes_from_landmark_.end()) {
      const auto& kf_set = it->second;
      auto& visible_landmarks = landmarks_from_keyframe_[*kf_set.begin()];
      auto& [landmark, _] = visible_landmarks[track_id];
      if (!landmark->get_pose()) {
        landmark->set_pose(point);
      }
    } else {
      new_landmarks[track_id].landmark = std::make_shared<Landmark>(track_id, point);
    }
  }

  size_t fourth_observations = 0;
  FastMap<TrackId, LandmarkAndObserv> old_landmarks;
  for (const auto& obs : observations) {
    auto it = keyframes_from_landmark_.find(obs.id);
    if (it == keyframes_from_landmark_.end()) {
      // obs of new landmark
      auto& l_o = new_landmarks[obs.id];
      if (l_o.observations.size() == 3) ++fourth_observations;
      l_o.observations.try_push_back(obs);

      if (!l_o.landmark) {
        l_o.landmark = std::make_shared<Landmark>(obs.id);
      }

    } else {
      // obs of old landmark
      const auto& kf_set = it->second;
      auto& visible_landmarks = landmarks_from_keyframe_[*kf_set.begin()];
      auto& [landmark, _] = visible_landmarks[obs.id];

      auto& old_lm = old_landmarks[obs.id];
      old_lm.landmark = landmark;
      if (old_lm.observations.size() == 3) ++fourth_observations;
      old_lm.observations.try_push_back(obs);
    }
  }

  if (fourth_observations > 0) TraceMessage("JOINT_FOURTH t=%lld n=%zu", static_cast<long long>(time_ns), fourth_observations);
  auto& landmarks = landmarks_from_keyframe_[new_kf];
  landmarks = std::move(new_landmarks);
  std::move(old_landmarks.begin(), old_landmarks.end(), std::inserter(landmarks, landmarks.begin()));

  for (auto& [track_id, l_o] : landmarks) {
    keyframes_from_landmark_[track_id].insert(new_kf);
  }

  if (!consecutive_keyframes_.empty()) {
    auto& [kf, p] = consecutive_keyframes_.back();
    p = std::make_shared<IMUPreintegration>(preint);
  }

  consecutive_keyframes_.push_back({new_kf});

  while (consecutive_keyframes_.size() > capacity_) {
    remove_tail_keyframe_thread_unsafe();
  }
}

Map<TrackId, Vector3T> UnifiedMap::get_recent_landmarks() const {
  Map<TrackId, Vector3T> out;
  get_recent_landmarks(out);
  return out;
}

void UnifiedMap::get_recent_landmarks(Map<TrackId, Vector3T>& out) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("get_recent_landmarks");
  std::lock_guard<std::mutex> lock(map_mutex_);
  out.clear();
  if (consecutive_keyframes_.empty()) {
    return;
  }

  const auto& recent_landmarks = landmarks_from_keyframe_.at(consecutive_keyframes_.back().keyframe);
  out.reserve(recent_landmarks.size());
  for (const auto& [track_id, lm_with_obs] : recent_landmarks) {
    const std::optional<Vector3T>& point_3d = lm_with_obs.landmark->get_pose();
    if (point_3d) {
      out.insert({track_id, *point_3d});
    }
  }
}

TrackIdMap UnifiedMap::get_recent_landmarks(CameraId cam_id) const {
  TrackIdMap out;
  get_recent_landmarks(cam_id, out);
  return out;
}

void UnifiedMap::get_recent_landmarks(CameraId cam_id, TrackIdMap& out) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("get_recent_landmarks(cam_id)");
  std::lock_guard<std::mutex> lock(map_mutex_);
  out.clear();
  if (consecutive_keyframes_.empty()) {
    return;
  }

  const auto& recent_landmarks = landmarks_from_keyframe_.at(consecutive_keyframes_.back().keyframe);
  out.reserve(recent_landmarks.size());
  for (const auto& [track_id, lm_with_obs] : recent_landmarks) {
    const std::optional<Vector3T>& point_3d = lm_with_obs.landmark->get_pose();
    if (point_3d) {
      for (const auto& obs : lm_with_obs.observations) {
        if (obs.cam_id == cam_id) {
          out.insert({track_id, *point_3d});
          break;
        }
      }
    }
  }
}

void UnifiedMap::remove_tail_keyframe_thread_unsafe() {
  // thread unsafe means that mutex is locked in the outer scope

  KeyframePtr kf = consecutive_keyframes_.front().keyframe;

  for (const auto& [track_id, landmark] : landmarks_from_keyframe_.at(kf)) {
    auto& kf_set = keyframes_from_landmark_[track_id];
    kf_set.erase(kf);
    if (kf_set.empty()) {
      keyframes_from_landmark_.erase(track_id);
    }
  }
  assert(!landmarks_from_keyframe_.empty());
  landmarks_from_keyframe_.erase(kf);
  consecutive_keyframes_.pop_front();
}

UnifiedMap::SubMap UnifiedMap::get_recent_submap(size_t max_last_keyframes, bool filter_landmarks) const {
  TRACE_EVENT ev = profiler_domain_.trace_event("get_recent_submap");
  SubMap sub_map;
  sub_map.consecutive_keyframes.reserve(max_last_keyframes);

  std::lock_guard<std::mutex> lock(map_mutex_);
  {
    TRACE_EVENT ev1 = profiler_domain_.trace_event("add keyframes");
    // add recent keyframes
    auto it = consecutive_keyframes_.rbegin();
    for (size_t i = 0; i < max_last_keyframes; i++) {
      if (it == consecutive_keyframes_.rend()) {
        break;
      }
      sub_map.consecutive_keyframes.push_back(*it);
      it++;
    }
    std::reverse(sub_map.consecutive_keyframes.begin(), sub_map.consecutive_keyframes.end());
  }

  {
    sub_map.landmark_and_obs.resize(sub_map.consecutive_keyframes.size());
    TRACE_EVENT ev1 = profiler_domain_.trace_event("add landmarks");

    for (size_t i = 0; i < sub_map.consecutive_keyframes.size(); i++) {
      const auto& kf = sub_map.consecutive_keyframes[i].keyframe;
      auto& lms = sub_map.landmark_and_obs[i];
      for (const auto& [track_id, x] : landmarks_from_keyframe_.at(kf)) {
        if (lms.full()) {
          break;
        }
        if (!filter_landmarks) {
          lms.try_push_back(x);
          continue;
        }

        if (x.landmark->get_pose() && keyframes_from_landmark_.at(track_id).size() > 1) {
          lms.try_push_back(x);
        }
      }
    }
  }
  return sub_map;
}

size_t UnifiedMap::size() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return consecutive_keyframes_.size();
}

size_t UnifiedMap::num_landmarks() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return keyframes_from_landmark_.size();
}

bool UnifiedMap::empty() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return consecutive_keyframes_.empty();
}

void UnifiedMap::set_gravity(const Vector3T& gravity) {
  std::lock_guard<std::mutex> lock(gravity_mutex_);
  gravity_ = gravity;
}

void UnifiedMap::reset_gravity() {
  std::lock_guard<std::mutex> lock(gravity_mutex_);
  gravity_ = std::nullopt;
}

std::optional<Vector3T> UnifiedMap::get_gravity() const {
  std::lock_guard<std::mutex> lock(gravity_mutex_);
  return gravity_;
}

size_t UnifiedMap::capacity() const { return capacity_; }

void UnifiedMap::clear() {
  std::scoped_lock lock(map_mutex_, gravity_mutex_);

  gravity_ = std::nullopt;
  landmarks_from_keyframe_.clear();
  consecutive_keyframes_.clear();
  keyframes_from_landmark_.clear();
}

}  // namespace cuvslam::map
