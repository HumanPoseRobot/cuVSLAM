#include <chrono>
#include <map>
#include "common/log.h"
#include "pnp/cauchy_parameters.h"
#include "pnp/pixel_support.h"
#include "window_joint_reference.h"

namespace gpu_window::joint_reference {
namespace {
Eigen::Matrix3d Skew(const Eigen::Vector3d& p) {
  Eigen::Matrix3d s;
  s << 0, -p.z(), p.y(), p.z(), 0, -p.x(), -p.y(), p.x(), 0;
  return s;
}
Eigen::Matrix<double, 2, 3> Projection(const Eigen::Vector3d& p) {
  Eigen::Matrix<double, 2, 3> j;
  j << 1. / p.z(), 0, -p.x() / (p.z() * p.z()), 0, 1. / p.z(), -p.y() / (p.z() * p.z());
  return j;
}
struct Observation {
  size_t node;
  cuvslam::camera::Observation value;
};
struct Group {
  Eigen::Vector3d point;
  std::vector<Observation> observations;
};
}  // namespace
bool SeedHistoricalGeometry(const cuvslam::map::UnifiedMap& map, const cuvslam::camera::Rig& rig,
                            const cuvslam::imu::ImuCalibration& cal, int64_t root_time, const State& root,
                            const std::map<cuvslam::TrackId, cuvslam::Vector3T>& points, Prior& output) {
  using namespace cuvslam;
  const auto started = std::chrono::steady_clock::now();
  const auto sub = map.get_recent_submap(map.capacity());
  if (sub.consecutive_keyframes.size() < 2 || points.empty()) {
    return false;
  }
  std::vector<Eigen::Isometry3d> world_from_imu;
  std::vector<int64_t> times;
  size_t root_node = sub.consecutive_keyframes.size();
  for (size_t i = 0; i < sub.consecutive_keyframes.size(); ++i) {
    const auto& keyframe = sub.consecutive_keyframes[i].keyframe;
    times.push_back(keyframe->time_ns());
    world_from_imu.push_back((keyframe->get_pose().inverse() * cal.rig_from_imu()).cast<double>());
    if (times.back() == root_time) {
      root_node = i;
    }
  }
  if (root_node == 0 || root_node == world_from_imu.size()) {
    return false;
  }
  std::map<TrackId, Group> groups;
  for (size_t i = 0; i < sub.landmark_and_obs.size(); ++i) {
    for (const auto& entry : sub.landmark_and_obs[i]) {
      const auto point = entry.landmark->get_pose();
      if (!point) {
        continue;
      }
      auto [it, inserted] = groups.try_emplace(entry.landmark->id());
      auto& group = it->second;
      if (inserted) {
        const auto wanted = points.find(entry.landmark->id());
        group.point = (wanted == points.end() ? *point : wanted->second).cast<double>();
      }
      for (const auto& obs : entry.observations) {
        if (obs.cam_id >= rig.num_cameras) {
          continue;
        }
        const Eigen::Isometry3d ci = (rig.camera_from_rig[obs.cam_id] * cal.rig_from_imu()).cast<double>();
        const Eigen::Vector3d pc = ci * world_from_imu[i].inverse() * group.point;
        if (!pc.allFinite() || !pnp::PixelObservationSupported(rig, obs, pc.cast<float>(), .05f)) {
          continue;
        }
        group.observations.push_back({i, obs});
      }
    }
  }
  std::vector<int64_t> ids;
  std::map<TrackId, int> point_index;
  for (const auto& [id, point] : points) {
    point_index.emplace(id, ids.size());
    ids.push_back(static_cast<int64_t>(id));
  }
  const int history_dimension = 6 * (world_from_imu.size() - 1), dimension = history_dimension + 3 * ids.size();
  std::vector<Eigen::MatrixXd> rows;
  std::vector<Eigen::VectorXd> residuals;
  int total = 0;
  size_t used = 0;
  const double cauchy = pnp::VisualCauchyParameters(1.f, 0).c;
  for (const auto& [id, group] : groups) {
    if (group.observations.empty()) {
      continue;
    }
    const int n = 2 * group.observations.size();
    Eigen::MatrixXd pose = Eigen::MatrixXd::Zero(n, history_dimension), point = Eigen::MatrixXd::Zero(n, 3);
    Eigen::VectorXd residual(n);
    int at = 0;
    for (const auto& observation : group.observations) {
      const auto& obs = observation.value;
      const auto& wi = world_from_imu[observation.node];
      const Eigen::Isometry3d ci = (rig.camera_from_rig[obs.cam_id] * cal.rig_from_imu()).cast<double>();
      const Eigen::Vector3d pi = wi.linear().transpose() * (group.point - wi.translation()), pc = ci * pi;
      Eigen::LLT<Eigen::Matrix2d> whiten(9. * obs.xy_info.cast<double>());
      if (whiten.info() != Eigen::Success) {
        return false;
      }
      const Eigen::Vector2d r = whiten.matrixU() * (pc.hnormalized() - obs.xy.cast<double>());
      const double weight = 1. / std::sqrt(1. + cauchy * r.squaredNorm());
      const Eigen::Matrix<double, 2, 3> j = weight * (whiten.matrixU() * Projection(pc)).eval() * ci.linear();
      point.middleRows(at, 2) = j * wi.linear().transpose();
      if (observation.node) {
        Eigen::Matrix<double, 3, 6> tangent;
        tangent.leftCols<3>() = Skew(pi);
        tangent.rightCols<3>() = -wi.linear().transpose() * wi.linear();
        pose.block(at, 6 * (observation.node - 1), 2, 6) = j * tangent;
      }
      residual.segment<2>(at) = weight * r;
      at += 2;
    }
    const auto wanted = point_index.find(id);
    Eigen::MatrixXd kept;
    Eigen::VectorXd r;
    if (wanted == point_index.end()) {
      gpu_window::project_out(point, pose, residual, kept, r);
      Eigen::MatrixXd full = Eigen::MatrixXd::Zero(kept.rows(), dimension);
      full.leftCols(history_dimension) = kept;
      kept = std::move(full);
    } else {
      ++used;
      kept = Eigen::MatrixXd::Zero(n, dimension);
      kept.leftCols(history_dimension) = pose;
      kept.middleCols(history_dimension + 3 * wanted->second, 3) = point;
      r = residual;
    }
    total += kept.rows();
    rows.push_back(std::move(kept));
    residuals.push_back(std::move(r));
  }
  Factor all;
  all.rows = Eigen::MatrixXd::Zero(total, dimension);
  all.residual.resize(total);
  int at = 0;
  for (size_t k = 0; k < rows.size(); ++k) {
    all.rows.middleRows(at, rows[k].rows()) = rows[k];
    all.residual.segment(at, residuals[k].size()) = residuals[k];
    at += rows[k].rows();
  }
  std::vector<int> kept;
  for (int k = 0; k < 6; ++k) {
    kept.push_back(6 * (root_node - 1) + k);
  }
  for (int k = history_dimension; k < dimension; ++k) {
    kept.push_back(k);
  }
  const auto compact = cuvslam::pnp::joint_geometry::compress(cuvslam::pnp::joint_geometry::eliminate(all, kept));
  Prior prior;
  prior.factor.rows = Eigen::MatrixXd::Zero(compact.rows.rows(), 15 + 3 * ids.size());
  prior.factor.rows.leftCols(6) = compact.rows.leftCols(6);
  prior.factor.rows.rightCols(3 * ids.size()) = compact.rows.rightCols(3 * ids.size());
  prior.factor.residual = compact.residual;
  prior.times = {root_time};
  prior.states = {root};
  prior.factor = Reanchor(prior.factor, prior.states);
  prior.ids = ids;
  for (auto id : ids) {
    prior.points.push_back(points.at(TrackId(id)).cast<double>());
  }
  prior.visual_absorbed_through_ns = root_time;  // Historical visuals, not future IMU edges.
  if (!prior.factor.valid()) {
    return false;
  }
  output = std::move(prior);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  TraceMessage("JOINT_GEOMETRY_SEED root=%lld historical=%zu points=%zu history_supported=%zu rows=%lld ms=%.6f",
               (long long)root_time, times.size(), ids.size(), used, (long long)output.factor.rows.rows(), ms);
  return true;
}
}  // namespace gpu_window::joint_reference
