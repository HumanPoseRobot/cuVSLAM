#include "joint_initial_graph.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <type_traits>
#include "common/log.h"
#include "joint_noise.h"

namespace gpu_window::posterior {
namespace {
using namespace cuvslam;
State Pack(const sba_imu::Pose& pose) {
  State state;
  CopyRows(state.R, pose.w_from_imu.linear().cast<double>().eval());
  CopyRows(state.p, pose.w_from_imu.translation().cast<double>().eval());
  CopyRows(state.v, pose.velocity.cast<double>().eval());
  CopyRows(state.bg, pose.gyro_bias.cast<double>().eval());
  CopyRows(state.ba, pose.acc_bias.cast<double>().eval());
  return state;
}

template <class T>
void Write(std::ostream& out, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
template <class T>
T Read(std::istream& in) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) {
    throw std::runtime_error("truncated joint graph snapshot");
  }
  return value;
}
template <class T>
void WriteVector(std::ostream& out, const std::vector<T>& values) {
  Write(out, uint64_t(values.size()));
  for (const auto& value : values) {
    Write(out, value);
  }
}
template <class T>
std::vector<T> ReadVector(std::istream& in) {
  const auto count = Read<uint64_t>(in);
  if (count > 1000000) {
    throw std::runtime_error("joint snapshot vector size exceeds schema");
  }
  std::vector<T> result;
  result.reserve(count);
  for (uint64_t i = 0; i < count; ++i) {
    result.push_back(Read<T>(in));
  }
  return result;
}
void WritePoints(std::ostream& out, const std::vector<Eigen::Vector3d>& points) {
  Write(out, uint64_t(points.size()));
  for (const auto& point : points) {
    for (int k = 0; k < 3; ++k) {
      Write(out, point[k]);
    }
  }
}
std::vector<Eigen::Vector3d> ReadPoints(std::istream& in) {
  const auto count = Read<uint64_t>(in);
  if (count > 1000000) {
    throw std::runtime_error("joint snapshot point count exceeds schema");
  }
  std::vector<Eigen::Vector3d> points(count);
  for (auto& point : points) {
    for (int k = 0; k < 3; ++k) {
      point[k] = Read<double>(in);
    }
  }
  return points;
}
void WriteGravity(std::ostream& out, const Eigen::Vector3d& gravity) {
  for (int k = 0; k < 3; ++k) {
    Write(out, gravity[k]);
  }
}
Eigen::Vector3d ReadGravity(std::istream& in) {
  Eigen::Vector3d gravity;
  for (int k = 0; k < 3; ++k) {
    gravity[k] = Read<double>(in);
  }
  return gravity;
}
void WritePrior(std::ostream& out, const Prior& prior) {
  WriteVector(out, prior.times);
  WriteVector(out, prior.ids);
  WriteVector(out, prior.nodes);
  WritePoints(out, prior.points);
  WriteGravity(out, prior.gravity);
  Write(out, prior.fixed_pose_time);
  Write(out, prior.visual_through);
  Write(out, prior.imu_left_through);
  WriteVector(out, prior.consumed_visual);
  const auto& factor = prior.factor();
  Write(out, int64_t(factor.rows.rows()));
  Write(out, int64_t(factor.rows.cols()));
  for (int col = 0; col < factor.rows.cols(); ++col) {
    for (int row = 0; row < factor.rows.rows(); ++row) {
      Write(out, factor.rows(row, col));
    }
  }
  for (int row = 0; row < factor.residual.size(); ++row) {
    Write(out, factor.residual[row]);
  }
  WriteVector(out, factor.evidence);
}
Prior ReadPrior(std::istream& in) {
  Prior prior;
  Factor factor;
  prior.times = ReadVector<int64_t>(in);
  prior.ids = ReadVector<int64_t>(in);
  prior.nodes = ReadVector<State>(in);
  prior.points = ReadPoints(in);
  prior.gravity = ReadGravity(in);
  prior.fixed_pose_time = Read<int64_t>(in);
  prior.visual_through = Read<int64_t>(in);
  prior.imu_left_through = Read<int64_t>(in);
  prior.consumed_visual = ReadVector<VisualKey>(in);
  const auto rows = Read<int64_t>(in), cols = Read<int64_t>(in);
  if (rows < 0 || cols < 0 || rows > 100000 || cols > 100000 || (cols && rows > 50000000 / cols)) {
    throw std::runtime_error("joint prior snapshot dimensions exceed schema");
  }
  factor.rows.resize(rows, cols);
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      factor.rows(row, col) = Read<double>(in);
    }
  }
  factor.residual.resize(rows);
  for (int row = 0; row < rows; ++row) {
    factor.residual[row] = Read<double>(in);
  }
  factor.evidence = ReadVector<uint64_t>(in);
  prior.SetFactor(std::move(factor));
  return prior;
}
}  // namespace

Graph BuildInitializationGraph(const map::UnifiedMap::SubMap& history, const std::vector<sba_imu::Pose>& initialized,
                               const CurrentFrame& current, const camera::Rig& rig,
                               const imu::ImuCalibration& calibration, const imu::ImuMeasurementStorage& storage,
                               const Eigen::Vector3d& gravity) {
  if (history.consecutive_keyframes.size() != initialized.size() || initialized.empty() ||
      history.landmark_and_obs.size() != initialized.size()) {
    throw std::invalid_argument("initialization graph history is incomplete");
  }
  Graph graph;
  graph.gravity = gravity;
  graph.fixed_pose_time = current.time_ns;
  std::map<TrackId, Vector3T> seeds;
  std::vector<std::vector<camera::Observation>> observations;
  for (size_t i = 0; i < initialized.size(); ++i) {
    graph.times.push_back(history.consecutive_keyframes[i].keyframe->time_ns());
    graph.nodes.push_back(Pack(initialized[i]));
    observations.emplace_back();
    for (const auto& entry : history.landmark_and_obs[i]) {
      if (const auto point = entry.landmark->get_pose()) {
        seeds.try_emplace(entry.landmark->id(), *point);
        observations.back().insert(observations.back().end(), entry.observations.begin(), entry.observations.end());
      }
    }
  }
  if (current.time_ns < graph.times.back()) {
    throw std::invalid_argument("initialization current clock precedes its keyframes");
  }
  if (current.time_ns > graph.times.back()) {
    graph.times.push_back(current.time_ns);
    auto root = current.pose;
    const auto& anchor = initialized.back();
    sba_imu::IMUPreintegration pi(calibration, storage, anchor.gyro_bias, anchor.acc_bias,
                                  graph.times[graph.times.size() - 2], current.time_ns);
    sba_imu::Pose predicted;
    if (!anchor.predict_pose(gravity.cast<float>(), pi, predicted)) {
      throw std::runtime_error("initialization tail has no valid inertial prediction");
    }
    // The accepted current visual pose fixes the coordinate gauge. Its velocity
    // and sensor bias only seed the solve; they are NOT independent priors.
    root.velocity = predicted.velocity;
    root.gyro_bias = anchor.gyro_bias;
    root.acc_bias = anchor.acc_bias;
    graph.nodes.push_back(Pack(root));
    observations.emplace_back();
  }
  for (const auto& [id, point] : current.points) {
    seeds.try_emplace(id, point);
  }
  observations.back().insert(observations.back().end(), current.observations.begin(), current.observations.end());
  std::map<TrackId, int> indices;
  std::set<std::tuple<int, int, TrackId>> seen;
  size_t duplicate = 0, domain = 0;
  for (size_t node = 0; node < observations.size(); ++node) {
    for (const auto& observation : observations[node]) {
      const auto point = seeds.find(observation.id);
      if (point == seeds.end() || observation.cam_id >= rig.num_cameras || !observation.xy.allFinite()) {
        ++domain;
        continue;
      }
      if (!seen.emplace(node, int(observation.cam_id), observation.id).second) {
        ++duplicate;
        continue;
      }
      const auto ci = (rig.camera_from_rig[observation.cam_id] * calibration.rig_from_imu()).cast<double>();
      const auto& state = graph.nodes[node];
      const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(state.R);
      const Eigen::Vector3d pc =
          ci * (rotation.transpose() * (point->second.cast<double>() - Eigen::Map<const Eigen::Vector3d>(state.p)));
      if (!pc.allFinite() || !(pc.z() > 0.)) {
        ++domain;
        continue;
      }
      auto [found, inserted] = indices.try_emplace(observation.id, graph.points.size());
      if (inserted) {
        graph.ids.push_back(static_cast<int64_t>(observation.id));
        graph.points.push_back(point->second.cast<double>());
      }
      Visual visual;
      visual.node = node;
      visual.point = found->second;
      CopyRows(visual.obs.Rci, ci.linear());
      CopyRows(visual.obs.tci, ci.translation());
      CopyRows(visual.obs.xy, observation.xy.cast<double>().eval());
      CopyRows(visual.obs.sqrt_info, VisualWhitening(StoredSymmetric(observation.xy_info)));
      visual.obs.min_depth = 0.;  // Actual projection domain, not a scene-dependent depth clamp.
      graph.visuals.push_back(visual);
      graph.visual_cameras.push_back(int(observation.cam_id));
    }
  }
  for (size_t i = 0; i + 1 < graph.nodes.size(); ++i) {
    const auto& state = graph.nodes[i];
    const Vector3T bg = Eigen::Map<const Eigen::Vector3d>(state.bg).cast<float>();
    const Vector3T ba = Eigen::Map<const Eigen::Vector3d>(state.ba).cast<float>();
    sba_imu::IMUPreintegration integration(calibration, storage, bg, ba, graph.times[i], graph.times[i + 1]);
    graph.edges.push_back(CalibratedEdge(integration, calibration, i, i + 1));
  }
  TraceMessage("JOINT_INITIAL_GRAPH states=%zu points=%zu visual=%zu imu=%zu duplicates=%zu domain=%zu root=%lld",
               graph.nodes.size(), graph.points.size(), graph.visuals.size(), graph.edges.size(), duplicate, domain,
               static_cast<long long>(current.time_ns));
  return graph;
}

void SaveGraph(const Graph& graph, const std::string& path) {
  if (std::filesystem::exists(path)) {
    throw std::runtime_error("refusing to overwrite a frozen joint graph");
  }
  std::ofstream out(path + ".tmp", std::ios::binary);
  Write(out, uint64_t(0x4a4f494e54475231));
  Write(out, uint32_t(0x01020304));
  Write(out, uint32_t(sizeof(State)));
  Write(out, uint32_t(sizeof(Edge)));
  Write(out, uint32_t(sizeof(Visual)));
  WriteVector(out, graph.times);
  WriteVector(out, graph.ids);
  WriteVector(out, graph.nodes);
  WritePoints(out, graph.points);
  WriteVector(out, graph.edges);
  WriteVector(out, graph.visuals);
  WriteVector(out, graph.visual_cameras);
  WriteGravity(out, graph.gravity);
  Write(out, graph.fixed_pose_time);
  Write(out, graph.huber);
  Write(out, graph.iterations);
  WritePrior(out, graph.prior);
  out.close();
  if (!out) {
    throw std::runtime_error("cannot write frozen joint graph");
  }
  std::filesystem::rename(path + ".tmp", path);
}

Graph LoadGraph(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (Read<uint64_t>(in) != 0x4a4f494e54475231 || Read<uint32_t>(in) != 0x01020304 ||
      Read<uint32_t>(in) != sizeof(State) || Read<uint32_t>(in) != sizeof(Edge) ||
      Read<uint32_t>(in) != sizeof(Visual)) {
    throw std::runtime_error("incompatible frozen joint graph schema");
  }
  Graph graph;
  graph.times = ReadVector<int64_t>(in);
  graph.ids = ReadVector<int64_t>(in);
  graph.nodes = ReadVector<State>(in);
  graph.points = ReadPoints(in);
  graph.edges = ReadVector<Edge>(in);
  graph.visuals = ReadVector<Visual>(in);
  graph.visual_cameras = ReadVector<int>(in);
  graph.gravity = ReadGravity(in);
  graph.fixed_pose_time = Read<int64_t>(in);
  graph.huber = Read<double>(in);
  graph.iterations = Read<int>(in);
  graph.prior = ReadPrior(in);
  if (in.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error("frozen joint graph has trailing data");
  }
  return graph;
}
}  // namespace gpu_window::posterior
