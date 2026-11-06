#pragma once

#include <array>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "camera/observation.h"
#include "camera/rig.h"
#include "common/log.h"
#include "pipelines/track.h"

namespace cuvslam::pipelines {

// A short history within one map epoch. Keyframe map changes invalidate all cached poses.
class TemporalLandmarks {
public:
  struct Ray {
    Isometry3T world_from_cam;
    Vector2T xy;
    Vector2T focal;
  };
  enum class Result { FewViews, Parallax, Depth, Reprojection, Accepted };

  static Result fit(const std::vector<Ray>& views, Vector3T& point, float angle_deg, float pixel_limit) {
    if (views.size() < 3) return Result::FewViews;
    const auto& last = views.back();
    const Vector3T current_ray = (last.world_from_cam.linear() * last.xy.homogeneous()).normalized();
    const float min_sin = std::sin(angle_deg * 0.01745329252f);
    bool conditioned = false;
    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for (const Ray& view : views) {
      const Vector3T ray = (view.world_from_cam.linear() * view.xy.homogeneous()).normalized();
      if ((view.world_from_cam.translation() - last.world_from_cam.translation()).norm() >= .025f &&
          ray.dot(current_ray) > 0.f && ray.cross(current_ray).norm() >= min_sin)
        conditioned = true;
      const Eigen::Vector3d u = ray.cast<double>();
      const Eigen::Matrix3d projection = Eigen::Matrix3d::Identity() - u * u.transpose();
      normal += projection;
      rhs += projection * view.world_from_cam.translation().cast<double>();
    }
    if (!conditioned) return Result::Parallax;
    const auto ldlt = normal.ldlt();
    if (!ldlt.isPositive()) return Result::Parallax;
    point = ldlt.solve(rhs).cast<float>();
    if (!point.allFinite()) return Result::Depth;
    for (const Ray& view : views) {
      const Vector3T pc = view.world_from_cam.inverse() * point;
      if (pc.z() <= .15f || pc.norm() > 12.f) return Result::Depth;
      const auto error = (pc.hnormalized() - view.xy).cwiseProduct(view.focal);
      if (error.norm() > pixel_limit) return Result::Reprojection;
    }
    return Result::Accepted;
  }

  void clear() { history_.clear(); }

  void remember(int64_t time, const camera::Rig& rig, const Isometry3T& world_from_rig,
                const std::vector<camera::Observation>& obs, unsigned camera_mask) {
    if (!history_.empty() && time - history_.back().time < 70000000) return;
    Snapshot s;
    s.time = time;
    for (const auto& o : obs) {
      if (!(camera_mask & (1u << o.cam_id))) continue;
      s.rays[o.cam_id].emplace(o.id, make_ray(rig, world_from_rig, o));
    }
    history_.push_back(std::move(s));
    if (history_.size() > 12) history_.pop_front();
  }

  void extend(int64_t time, const camera::Rig& rig, const Isometry3T& pose, const std::vector<camera::Observation>& obs,
              const std::unordered_map<TrackId, Vector3T>& mapped, unsigned camera_mask, bool add_temporal,
              float angle_deg, float pixel_limit, std::vector<Landmark>& out) const {
    std::unordered_map<TrackId, unsigned> observed_cams;
    for (const auto& o : obs) observed_cams[o.id] |= 1u << o.cam_id;
    std::array<int, 6> pair{};
    for (const auto& track : observed_cams) {
      size_t k = 0;
      for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 4; ++b, ++k)
          if ((track.second & (1u << a)) && (track.second & (1u << b))) ++pair[k];
    }
    TraceMessage("CAM01_PAIRS t=%lld pairs=%d,%d,%d,%d,%d,%d", static_cast<long long>(time), pair[0], pair[1], pair[2],
                 pair[3], pair[4], pair[5]);
    std::unordered_set<TrackId> stereo, added;
    for (const auto& l : out) stereo.insert(l.id);
    struct Stats {
      int seen = 0, shared = 0, mapped = 0, stereo = 0;
      std::array<int, 5> outcome{};
    };
    std::array<Stats, 4> stats{};
    for (const auto& o : obs) {
      auto& a = stats[o.cam_id];
      ++a.seen;
      const unsigned cams = observed_cams.at(o.id);
      a.shared += (cams & (cams - 1)) != 0;
      if (mapped.count(o.id)) {
        ++a.mapped;
        continue;
      }
      if (stereo.count(o.id)) {
        ++a.stereo;
        continue;
      }
      if (!(camera_mask & (1u << o.cam_id)) || added.count(o.id)) continue;
      std::vector<Ray> rays;
      for (const auto& s : history_) {
        if (time <= s.time || time - s.time > 1100000000) continue;
        const auto it = s.rays[o.cam_id].find(o.id);
        if (it != s.rays[o.cam_id].end()) rays.push_back(it->second);
      }
      rays.push_back(make_ray(rig, pose, o));
      Vector3T point;
      const Result result = fit(rays, point, angle_deg, pixel_limit);
      ++a.outcome[static_cast<int>(result)];
      if (add_temporal && result == Result::Accepted) {
        out.push_back({o.id, point});
        added.insert(o.id);
      }
    }
    for (int i = 0; i < 4; ++i) {
      const auto& a = stats[i];
      TraceMessage(
          "CAM01_TRI t=%lld cam=%d seen=%d shared=%d mapped=%d stereo=%d few=%d parallax=%d depth=%d reproj=%d "
          "accepted=%d active=%d",
          static_cast<long long>(time), i, a.seen, a.shared, a.mapped, a.stereo, a.outcome[0], a.outcome[1],
          a.outcome[2], a.outcome[3], a.outcome[4], static_cast<int>(add_temporal));
    }
  }

private:
  struct Snapshot {
    int64_t time;
    std::array<std::unordered_map<TrackId, Ray>, 4> rays;
  };
  std::deque<Snapshot> history_;
  static Ray make_ray(const camera::Rig& rig, const Isometry3T& pose, const camera::Observation& o) {
    return {pose * rig.camera_from_rig[o.cam_id].inverse(), o.xy, rig.intrinsics[o.cam_id]->getFocal()};
  }
};
}  // namespace cuvslam::pipelines
