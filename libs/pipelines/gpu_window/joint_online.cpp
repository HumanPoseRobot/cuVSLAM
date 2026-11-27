#include "joint_online.h"
#include <array>
#include <cstring>
#include <fstream>
#include <mutex>
#include "finite_geometry.h"
#include "history_archive.h"
#include "observed_history_support.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include "common/log.h"
#include "joint_noise.h"
#include "pipelines/triangulator.h"

namespace gpu_window::posterior {
namespace {

// Diagnostic-only ownership ledger. One file per run; no numerical state is changed.

// A04 evidence capture only: actual tracker observations once per rig frame.
// Frame header: int64 timestamp,count; each record: int64 track,camera + 6 doubles.
void CaptureBearings(int64_t time, const std::vector<cuvslam::camera::Observation>& observations) {
  const char* path = std::getenv("VIO_RAW_BEARINGS_FILE");
  if (!path) {
    return;
  }
  struct Record {
    int64_t track, camera;
    double xy_info[6];
  };
  static_assert(sizeof(Record) == 64);
  std::vector<Record> records;
  records.reserve(observations.size());
  for (const auto& o : observations) {
    records.push_back({static_cast<int64_t>(o.id),
                       static_cast<int64_t>(o.cam_id),
                       {o.xy.x(), o.xy.y(), o.xy_info(0, 0), o.xy_info(0, 1), o.xy_info(1, 0), o.xy_info(1, 1)}});
  }
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);
  std::ofstream file(path, std::ios::binary | std::ios::app);
  const std::array<int64_t, 2> header{time, static_cast<int64_t>(records.size())};
  file.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
  file.write(reinterpret_cast<const char*>(records.data()), records.size() * sizeof(Record));
  if (!file) {
    TraceMessage("BEARINGS_IO_FAILURE");
  }
}

void CaptureRawGraph(const Graph& graph, const char* stage) {
  const char* dir = std::getenv("VIO_RAW_FACTOR_DIR");
  if (!dir) {
    return;
  }
  Graph raw = graph;
  raw.prior = Prior{};  // full initial prior and selected live priors are captured separately
  SaveGraph(raw, std::string(dir) + "/" + stage + "_" + std::to_string(graph.times.back()) + ".bin");
}

using OwnershipRecord = std::array<int64_t, 6>;
void OwnershipFlush(const std::vector<OwnershipRecord>& records) {
  const char* path = std::getenv("VIO_EVIDENCE_AUDIT_FILE");
  if (!path || records.empty()) {
    return;
  }
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);
  std::ofstream file(path, std::ios::binary | std::ios::app);
  file.write(reinterpret_cast<const char*>(records.data()), records.size() * sizeof(OwnershipRecord));
  if (!file) {
    TraceMessage("OWNERSHIP_IO_FAILURE");
  }
}
void OwnershipStage(const Graph& graph, const char* stage) {
  if (!std::getenv("VIO_EVIDENCE_AUDIT_FILE")) {
    return;
  }
  std::set<VisualKey> keys;
  size_t duplicate = 0, overlap = 0, imu_overlap = 0;
  for (size_t i = 0; i < graph.visuals.size(); ++i) {
    const auto& v = graph.visuals[i];
    const VisualKey key{graph.times.at(v.node), graph.ids.at(v.point), graph.visual_cameras.at(i)};
    duplicate += !keys.insert(key).second;
    overlap += !graph.prior.empty() &&
               (key.time <= graph.prior.visual_through ||
                std::binary_search(graph.prior.consumed_visual.begin(), graph.prior.consumed_visual.end(), key));
  }
  for (const auto& edge : graph.edges) {
    imu_overlap += !graph.prior.empty() && graph.times.at(edge.left) <= graph.prior.imu_left_through;
  }
  TraceMessage(
      "OWNERSHIP_STAGE t=%lld stage=%s states=%zu points=%zu visual=%zu prior_keys=%zu duplicate=%zu overlap=%zu "
      "imu_overlap=%zu",
      static_cast<long long>(graph.times.back()), stage, graph.nodes.size(), graph.points.size(), graph.visuals.size(),
      graph.prior.consumed_visual.size(), duplicate, overlap, imu_overlap);
}
// Diagnostic snapshots only. All solver inputs and stopping policies are kept.
// A crossing test on graph timestamps avoids process-global counters and works
// with dropped frames; requested times are never substituted for actual stamps.
bool ExactAuditRequested(int64_t left, int64_t right) {
  const char* text = std::getenv("VIO_EXACT_GRAPH_TIMES_NS");
  if (!text) return false;
  while (*text) {
    char* end = nullptr;
    const auto value = std::strtoll(text, &end, 10);
    if (end == text || (*end && *end != ',')) {
      throw std::invalid_argument("invalid exact graph audit timestamp list");
    }
    if (left < value && value <= right) return true;
    text = *end ? end + 1 : end;
  }
  return false;
}
void ExactAuditSave(const Graph& graph, const char* stage) {
  const char* dir = std::getenv("VIO_EXACT_GRAPH_DIR");
  if (!dir) return;
  try {
    const auto file = std::string(dir) + "/" + stage + "_" + std::to_string(graph.times.back()) + ".bin";
    SaveGraph(graph, file);
    TraceMessage("EXACT_GRAPH_SAVED stage=%s t=%lld states=%zu points=%zu visuals=%zu", stage,
                 (long long)graph.times.back(), graph.nodes.size(), graph.points.size(), graph.visuals.size());
  } catch (const std::exception& error) {
    TraceMessage("EXACT_GRAPH_FAILED stage=%s detail=%s", stage, error.what());
  }
}
using namespace cuvslam;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Clock = std::chrono::steady_clock;

bool HistoricalConditionalsEnabled() {
  const char* text = std::getenv("VIO_HISTORICAL_CONDITIONAL");
  if (!text || !std::strcmp(text, "0")) {
    return false;
  }
  if (!std::strcmp(text, "1")) {
    return true;
  }
  throw std::invalid_argument("invalid VIO_HISTORICAL_CONDITIONAL");
}

sba_imu::Pose PoseView(const State& state, const Matrix* covariance) {
  sba_imu::Pose pose;
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(state.R);
  pose.w_from_imu = Isometry3T::Identity();
  pose.w_from_imu.linear() = rotation.cast<float>();
  pose.w_from_imu.translation() = Eigen::Map<const Eigen::Vector3d>(state.p).cast<float>();
  pose.velocity = Eigen::Map<const Eigen::Vector3d>(state.v).cast<float>();
  pose.gyro_bias = Eigen::Map<const Eigen::Vector3d>(state.bg).cast<float>();
  pose.acc_bias = Eigen::Map<const Eigen::Vector3d>(state.ba).cast<float>();
  // This FP32 field is only the legacy POSE-information view. The full joint
  // posterior is owned by this backend and is never reconstructed from it.
  pose.info.setZero();
  if (covariance && covariance->rows() == 15) {
    const Eigen::Matrix<double, 6, 6> marginal = covariance->topLeftCorner<6, 6>();
    Eigen::LLT<Eigen::Matrix<double, 6, 6>> factor(marginal);
    if (factor.info() == Eigen::Success) {
      const auto information = factor.solve(Eigen::Matrix<double, 6, 6>::Identity()).eval();
      if (information.allFinite() && information.cwiseAbs().maxCoeff() <= std::numeric_limits<float>::max()) {
        pose.info.topLeftCorner<6, 6>() = information.cast<float>();
      }
    }
  }
  return pose;
}

bool Owned(const Graph& graph, const VisualKey& key) {
  if (!graph.prior.empty() &&
      (key.time <= graph.prior.visual_through ||
       std::binary_search(graph.prior.consumed_visual.begin(), graph.prior.consumed_visual.end(), key))) {
    return true;
  }
  for (size_t i = 0; i < graph.visuals.size(); ++i) {
    const auto& value = graph.visuals[i];
    if (graph.times[value.node] == key.time && graph.ids[value.point] == key.point &&
        graph.visual_cameras[i] == key.camera) {
      return true;
    }
  }
  return false;
}

struct RawFrame {
  int64_t time;
  std::vector<camera::Observation> observations;
};

void AddView(Graph& graph, int node, const camera::Rig& rig, const imu::ImuCalibration& calibration,
             const std::vector<camera::Observation>& observations, const std::unordered_map<TrackId, Vector3T>& seeds,
             int source = 0) {
  const bool audit = std::getenv("VIO_EVIDENCE_AUDIT_FILE") != nullptr;
  std::vector<OwnershipRecord> records;
  std::map<int64_t, int> indices;
  for (size_t i = 0; i < graph.ids.size(); ++i) {
    indices.emplace(graph.ids[i], i);
  }
  const auto& state = graph.nodes[node];
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(state.R);
  for (const auto& observation : observations) {
    const int64_t id = static_cast<int64_t>(observation.id);
    const VisualKey key{graph.times[node], id, int(observation.cam_id)};
    const auto record = [&](int reason) {
      if (audit) {
        records.push_back({graph.times.back(), key.time, key.point, key.camera, source, reason});
      }
    };
    if (observation.cam_id >= rig.num_cameras) {
      record(1);
      continue;
    }
    if (Owned(graph, key)) {
      record(2);
      continue;
    }
    auto found = indices.find(id);
    if (found == indices.end()) {
      const auto seed = seeds.find(observation.id);
      if (seed == seeds.end() || !seed->second.allFinite()) {
        record(3);
        continue;
      }
      indices.emplace(id, graph.points.size());
      graph.ids.push_back(id);
      graph.points.push_back(seed->second.cast<double>());
      found = indices.find(id);
    }
    const auto ci = (rig.camera_from_rig[observation.cam_id] * calibration.rig_from_imu()).cast<double>();
    const Eigen::Vector3d pc =
        ci * (rotation.transpose() * (graph.points[found->second] - Eigen::Map<const Eigen::Vector3d>(state.p)));
    if (!pc.allFinite() || !(pc.z() > 0.) || !observation.xy.allFinite()) {
      record(4);
      continue;
    }
    Visual visual;
    visual.node = node;
    visual.point = found->second;
    CopyRows(visual.obs.Rci, ci.linear());
    CopyRows(visual.obs.tci, ci.translation());
    CopyRows(visual.obs.xy, observation.xy.cast<double>().eval());
    CopyRows(visual.obs.sqrt_info, VisualWhitening(StoredSymmetric(observation.xy_info)));
    visual.obs.min_depth = 0.;
    graph.visuals.push_back(visual);
    graph.visual_cameras.push_back(int(observation.cam_id));
    record(0);
  }
  OwnershipFlush(records);
}

void AddStereoSeeds(Graph& graph, int node, const camera::Rig& rig, const imu::ImuCalibration& calibration,
                    const std::vector<camera::Observation>& observations,
                    std::unordered_map<TrackId, Vector3T>& seeds) {
  const auto pose = PoseView(graph.nodes[node], nullptr);
  const Isometry3T world_from_rig = pose.w_from_imu * calibration.rig_from_imu().inverse();
  pipelines::MulticamTriangulator triangulator(rig);
  for (const auto& landmark : triangulator.triangulate(world_from_rig, observations)) {
    seeds.try_emplace(landmark.id, landmark.point_w);
  }
}

// Unconsumed birth observations may be added at a retained keyframe. Old map
// means are seeds only; no zero-uncertainty 3D measurements are manufactured.
void AddMapBirths(Graph& graph, const map::UnifiedMap& map, const camera::Rig& rig,
                  const imu::ImuCalibration& calibration) {
  const auto sub = map.get_recent_submap(map.capacity());
  for (size_t i = 0; i < sub.consecutive_keyframes.size(); ++i) {
    const auto time = sub.consecutive_keyframes[i].keyframe->time_ns();
    const auto node = std::find(graph.times.begin(), graph.times.end(), time);
    if (node == graph.times.end()) {
      continue;
    }
    std::unordered_map<TrackId, Vector3T> seeds;
    std::vector<camera::Observation> observations;
    for (const auto& entry : sub.landmark_and_obs[i]) {
      if (const auto point = entry.landmark->get_pose()) {
        seeds.try_emplace(entry.landmark->id(), *point);
        observations.insert(observations.end(), entry.observations.begin(), entry.observations.end());
      }
    }
    AddView(graph, node - graph.times.begin(), rig, calibration, observations, seeds, 2);
  }
}

Graph RetainSuffix(const Graph& graph, int count, history::Archive* archive) {
  if (graph.nodes.size() <= size_t(count)) {
    return graph;
  }
  const int retired = graph.nodes.size() - count;
  auto boundary = graph;
  boundary.visuals.clear();
  boundary.visual_cameras.clear();
  for (size_t i = 0; i < graph.visuals.size(); ++i) {
    if (graph.visuals[i].node < retired) {
      boundary.visuals.push_back(graph.visuals[i]);
      boundary.visual_cameras.push_back(graph.visual_cameras[i]);
    }
  }
  boundary.edges.erase(std::remove_if(boundary.edges.begin(), boundary.edges.end(),
                                      [&](const auto& edge) { return edge.left >= retired && edge.right >= retired; }),
                       boundary.edges.end());
  std::set<int64_t> live_ids;
  for (const auto& visual : graph.visuals) {
    if (visual.node >= retired) {
      live_ids.insert(graph.ids[visual.point]);
    }
  }
  std::vector<int64_t> times(graph.times.begin() + retired, graph.times.end()), ids(live_ids.begin(), live_ids.end());
  const auto consumed = Linearize(boundary, false);
  if (archive) {
    archive->Capture(graph, consumed, times, ids);
  }
  const auto prior = Marginalize(graph, consumed, times, ids, graph.times[retired - 1], graph.times[retired - 1]);
  Graph result;
  result.prior = prior;
  result.times = times;
  result.ids = ids;
  result.nodes.assign(graph.nodes.begin() + retired, graph.nodes.end());
  result.gravity = graph.gravity;
  result.fixed_pose_time = graph.fixed_pose_time;
  result.huber = graph.huber;
  result.iterations = graph.iterations;
  for (auto id : ids) {
    const auto found = std::find(graph.ids.begin(), graph.ids.end(), id);
    result.points.push_back(graph.points[found - graph.ids.begin()]);
  }
  for (auto edge : graph.edges) {
    if (edge.left >= retired && edge.right >= retired) {
      edge.left -= retired;
      edge.right -= retired;
      result.edges.push_back(edge);
    }
  }
  for (size_t i = 0; i < graph.visuals.size(); ++i) {
    auto visual = graph.visuals[i];
    if (visual.node >= retired) {
      visual.node -= retired;
      visual.point = std::find(ids.begin(), ids.end(), graph.ids[visual.point]) - ids.begin();
      result.visuals.push_back(visual);
      result.visual_cameras.push_back(graph.visual_cameras[i]);
    }
  }
  return result;
}

bool VisualConsensus(const Graph& graph, const camera::Rig& rig, int required, int64_t time) {
  std::vector<int> counts(rig.num_cameras, 0);
  std::vector<Eigen::Vector2d> lo(rig.num_cameras, Eigen::Vector2d::Constant(1e30));
  std::vector<Eigen::Vector2d> hi(rig.num_cameras, Eigen::Vector2d::Constant(-1e30));
  std::set<int64_t> unique, informed;
  if (!graph.prior.empty()) {
    const int navigation = 15 * graph.prior.nodes.size();
    for (size_t i = 0; i < graph.prior.ids.size(); ++i) {
      const Matrix jacobian = graph.prior.factor().rows.middleCols(navigation + 3 * i, 3);
      Eigen::ColPivHouseholderQR<Matrix> qr(jacobian);
      qr.setThreshold(256. * std::numeric_limits<double>::epsilon() * std::max(jacobian.rows(), jacobian.cols()));
      if (qr.rank() == 3) {
        informed.insert(graph.prior.ids[i]);
      }
    }
  }
  for (const auto& visual : graph.visuals) {
    if (graph.times[visual.node] != time) {
      informed.insert(graph.ids[visual.point]);
    }
  }
  int inliers = 0;
  for (size_t i = 0; i < graph.visuals.size(); ++i) {
    const auto& visual = graph.visuals[i];
    if (graph.times[visual.node] != time || !informed.count(graph.ids[visual.point])) {
      continue;
    }
    const auto& node = graph.nodes[visual.node];
    const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(node.R), camera(visual.obs.Rci);
    const Eigen::Vector3d pc =
        camera * rotation.transpose() * (graph.points[visual.point] - Eigen::Map<const Eigen::Vector3d>(node.p)) +
        Eigen::Map<const Eigen::Vector3d>(visual.obs.tci);
    if (!(pc.z() > 0.) || !pc.allFinite()) {
      continue;
    }
    const Eigen::Vector2d error = pc.hnormalized() - Eigen::Map<const Eigen::Vector2d>(visual.obs.xy);
    const Eigen::Map<const Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> whiten(visual.obs.sqrt_info);
    Vector2T predicted, observed;
    const int cam = graph.visual_cameras[i];
    if (!rig.intrinsics[cam]->denormalizePoint(pc.hnormalized().cast<float>(), predicted) ||
        !rig.intrinsics[cam]->denormalizePoint(Eigen::Map<const Eigen::Vector2d>(visual.obs.xy).cast<float>(),
                                               observed)) {
      continue;
    }
    if ((whiten * error).squaredNorm() > 9.21 || (predicted - observed).squaredNorm() > 64.) {
      continue;
    }
    ++inliers;
    ++counts[cam];
    unique.insert(graph.ids[visual.point]);
    lo[cam] = lo[cam].cwiseMin(observed.cast<double>());
    hi[cam] = hi[cam].cwiseMax(observed.cast<double>());
  }
  int supported = 0;
  for (int c = 0; c < rig.num_cameras; ++c) {
    supported += counts[c] >= 5 && (hi[c] - lo[c]).minCoeff() >= 16.;
  }
  // Preserve the old observation-count/coverage policy. No room-specific
  // velocity, bias, height or correction-magnitude bounds are added.
  const int minimum = std::max(20, (required + 1) / 2);
  const bool accept = inliers >= minimum && unique.size() >= size_t(std::max(12, minimum / 2)) && supported >= 2;
  TraceMessage("JOINT_POSTERIOR_SUPPORT t=%lld accept=%d inliers=%d unique=%zu cameras=%d",
               static_cast<long long>(time), int(accept), inliers, unique.size(), supported);
  return accept;
}

State Predict(const State& left, const sba_imu::IMUPreintegration& pi, const Eigen::Vector3d& gravity) {
  State result = left;
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(left.R);
  const Vector3T bg = Eigen::Map<const Eigen::Vector3d>(left.bg).cast<float>();
  const Vector3T ba = Eigen::Map<const Eigen::Vector3d>(left.ba).cast<float>();
  const Eigen::Matrix3d r = rotation * pi.GetDeltaRotation(bg).cast<double>();
  const Eigen::Vector3d velocity = Eigen::Map<const Eigen::Vector3d>(left.v) + gravity * pi.dT_s +
                                   rotation * pi.GetDeltaVelocity(bg, ba).cast<double>();
  const Eigen::Vector3d position =
      Eigen::Map<const Eigen::Vector3d>(left.p) + Eigen::Map<const Eigen::Vector3d>(left.v) * pi.dT_s +
      .5 * gravity * pi.dT_s * pi.dT_s + rotation * pi.GetDeltaPosition(bg, ba).cast<double>();
  CopyRows(result.R, r);
  CopyRows(result.p, position);
  CopyRows(result.v, velocity);
  return result;
}

bool Publishable(const Graph& graph) {
  const double limit = std::numeric_limits<float>::max();
  for (const auto& point : graph.points) {
    if (!point.allFinite() || point.cwiseAbs().maxCoeff() > limit) {
      return false;
    }
  }
  for (const auto& node : graph.nodes) {
    const auto pose = PoseView(node, nullptr);
    if (!pose.w_from_imu.matrix().allFinite() || !pose.velocity.allFinite() || !pose.gyro_bias.allFinite() ||
        !pose.acc_bias.allFinite()) {
      return false;
    }
  }
  return graph.gravity.allFinite() && graph.gravity.cwiseAbs().maxCoeff() <= limit;
}
}  // namespace

sba_imu::Pose StatePose(const State& state) { return PoseView(state, nullptr); }

struct Online::Impl {
  Graph active, provisional, pending, publish;
  const bool historical_conditionals = HistoricalConditionalsEnabled();
  history::Archive history, provisional_history, pending_history;
  std::vector<RawFrame> raw, provisional_raw, pending_raw;
  Matrix covariance, pending_covariance;
  std::map<int64_t, map::State> expected_states;
  std::map<int64_t, Vector3T> expected_points;
  std::optional<Vector3T> expected_gravity;
  sba_imu::Pose prediction;
  int64_t prediction_time = -1;
  bool ready = false, have_provisional = false, have_pending = false;
};
Online::Online() : impl_(std::make_unique<Impl>()) {
  // Reject invalid model configuration at construction, before map mutation.
  (void)VisualWhitening(Eigen::Matrix2d::Identity());
}
Online::~Online() = default;
bool Online::ready() const { return impl_->ready; }
void Online::reset() { impl_ = std::make_unique<Impl>(); }
void Online::discard() { impl_->have_pending = false; }

bool Online::initialize(Graph graph, const CurrentFrame& current, const camera::Rig& rig,
                        const imu::ImuCalibration& calibration, Initialization& result) {
  const auto start = Clock::now();
  auto seeds = current.points;
  AddStereoSeeds(graph, graph.nodes.size() - 1, rig, calibration, current.observations, seeds);
  AddView(graph, graph.nodes.size() - 1, rig, calibration, current.observations, seeds, 3);
  CaptureBearings(current.time_ns, current.observations);
  graph.iterations = 40;
  ExactAuditSave(graph, "initial_input");
  finite_geometry::Stats finite;
  const auto solve = finite_geometry::OptimizeFinite(graph, nullptr, finite);
  if (finite_geometry::Enabled()) {
    TraceMessage("FINITE_INIT t=%lld points=%zu deferred=%zu boundary=%d rank=%d numerical=%d geometry_ms=%.6f",
                 (long long)current.time_ns, graph.points.size(), finite.deferred.size(), finite.boundary,
                 finite.unobservable, finite.numerical, finite.geometry_ms);
    for (const auto& item : finite.deferred) {
      TraceMessage("FINITE_DEFER t=%lld id=%lld reason=%d", (long long)current.time_ns, (long long)item.first,
                   int(item.second));
    }
  }
  ExactAuditSave(graph, "initial_optimized");
  if (!solve.success || !Publishable(graph)) {
    return false;
  }
  std::set<int64_t> live;
  for (const auto& visual : graph.visuals) {
    if (visual.node == int(graph.nodes.size()) - 1) {
      live.insert(graph.ids[visual.point]);
    }
  }
  const std::vector<int64_t> ids(live.begin(), live.end());
  const auto prior = Marginalize(graph, Linearize(graph, false), {graph.times.back()}, ids,
                                 graph.times[graph.times.size() - 2], graph.times[graph.times.size() - 2]);
  Graph stored;
  stored.times = prior.times;
  stored.nodes = prior.nodes;
  stored.ids = prior.ids;
  stored.points = prior.points;
  stored.gravity = prior.gravity;
  stored.fixed_pose_time = prior.fixed_pose_time;
  stored.prior = prior;
  Matrix covariance;
  Vector gradient;
  if (!CurrentMarginal(stored, current.time_ns, covariance, gradient)) {
    return false;
  }
  // Gravity stays a random variable, including its cross-covariance with v/b.
  std::vector<int> gravity_columns{int(prior.factor().rows.cols()) - 2, int(prior.factor().rows.cols()) - 1};
  if (!cuvslam::pnp::joint_geometry::marginal(prior.factor(), gravity_columns).observable) {
    return false;
  }
  ExactAuditSave(stored, "initial_transferred");
  OwnershipStage(stored, "initial_transferred");
  result.complete = std::move(graph);
  result.live = std::move(stored);
  result.root_covariance = covariance;
  result.solve = solve;
  TraceMessage("JOINT_POSTERIOR_INIT t=%lld before=%.9g after=%.9g iterations=%d states=%zu points=%zu ms=%.6f",
               static_cast<long long>(current.time_ns), solve.before, solve.after, solve.iterations,
               result.complete.nodes.size(), result.complete.points.size(),
               std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  return true;
}

void Online::adopt(Initialization&& result, const CurrentFrame& current) {
  auto& d = *impl_;
  if (d.historical_conditionals) {
    history::Archive archive;
    archive.Capture(result.complete, Linearize(result.complete, false), result.live.times, result.live.ids);
    d.history = std::move(archive);
    const auto stats = d.history.GetStats();
    TraceMessage("HISTORY_INIT t=%lld records=%llu bytes=%llu", (long long)current.time_ns,
                 (unsigned long long)stats.live_records, (unsigned long long)stats.coefficient_bytes);
  }
  d.active = std::move(result.live);
  d.raw = {{current.time_ns, current.observations}};
  d.covariance = Matrix::Zero(15, 15);
  d.covariance.bottomRightCorner<9, 9>() = result.root_covariance;
  d.ready = true;
  d.have_provisional = d.have_pending = false;
}

bool Online::solve(map::UnifiedMap& map, const camera::Rig& rig, const imu::ImuCalibration& calibration,
                   const imu::ImuMeasurementStorage& storage, const std::unordered_map<TrackId, Vector3T>& points,
                   const std::vector<camera::Observation>& observations, const pipelines::InertialPnPSettings& settings,
                   int64_t time, bool is_keyframe, sba_imu::Pose& candidate, sba_imu::Pose& prediction,
                   bool& prediction_valid) {
  auto& d = *impl_;
  prediction_valid = false;
  if (!d.ready) {
    return false;
  }
  d.have_pending = false;
  const auto begin = Clock::now();
  const auto gpu_before = cuvslam::pnp::joint_geometry::dense::GetGpuMetrics();
  try {
    d.expected_states.clear();
    d.expected_points.clear();
    d.expected_gravity = map.get_gravity();
    const auto snapshot = map.get_recent_submap(map.capacity());
    for (const auto& key : snapshot.consecutive_keyframes) {
      d.expected_states.emplace(key.keyframe->time_ns(), key.keyframe->get_state());
    }
    for (const auto& frame : snapshot.landmark_and_obs) {
      for (const auto& entry : frame) {
        if (const auto point = entry.landmark->get_pose()) {
          d.expected_points.try_emplace(static_cast<int64_t>(entry.landmark->id()), *point);
        }
      }
    }
    Graph graph = d.have_provisional ? d.provisional : d.active;
    auto raw = d.have_provisional ? d.provisional_raw : d.raw;
    auto history = d.have_provisional ? d.provisional_history : d.history;
    if (graph.times.empty() || time <= graph.times.back()) {
      throw std::invalid_argument("non-increasing online posterior clock");
    }
    const auto& left = graph.nodes.back();
    const Vector3T bg = Eigen::Map<const Eigen::Vector3d>(left.bg).cast<float>();
    const Vector3T ba = Eigen::Map<const Eigen::Vector3d>(left.ba).cast<float>();
    sba_imu::IMUPreintegration integration(calibration, storage, bg, ba, graph.times.back(), time);
    const auto edge = CalibratedEdge(integration, calibration, graph.nodes.size() - 1, graph.nodes.size());
    const auto predicted = Predict(left, integration, graph.gravity);
    graph.nodes.push_back(predicted);
    graph.times.push_back(time);
    graph.edges.push_back(edge);
    prediction = PoseView(predicted, nullptr);
    d.prediction = prediction;
    d.prediction_time = time;
    prediction_valid = true;
    AddMapBirths(graph, map, rig, calibration);
    auto seeds = points;
    AddStereoSeeds(graph, graph.nodes.size() - 1, rig, calibration, observations, seeds);
    CaptureBearings(time, observations);
    raw.push_back({time, observations});
    // When a point first acquires a stereo seed, earlier bearings in the active
    // lag also become real factors; they never become fixed-pose pseudo-priors.
    for (const auto& frame : raw) {
      const auto found = std::find(graph.times.begin(), graph.times.end(), frame.time);
      if (found != graph.times.end()) {
        AddView(graph, found - graph.times.begin(), rig, calibration, frame.observations, seeds, 1);
      }
    }
    if (d.historical_conditionals) {
      const auto observed = history::ObservedPointIds(graph);
      size_t deferred = 0;
      for (auto id : graph.ids) {
        if (history.OwnsPoint(id) && !observed.count(id)) { ++deferred; }
      }
      TraceMessage("HISTORY_SUPPORT t=%lld observed=%zu deferred_unobserved=%zu", (long long)time, observed.size(), deferred);
      const auto before = history.GetStats();
      const auto start = Clock::now();
      const auto stats = history.Restore(graph, &observed);
      TraceMessage(
          "HISTORY_RESTORE t=%lld restored=%llu ancestors=%llu records=%llu bytes=%llu seed_fallback=%llu ms=%.6f",
          (long long)time, (unsigned long long)(stats.restored - before.restored),
          (unsigned long long)(stats.ancestors - before.ancestors), (unsigned long long)stats.live_records,
          (unsigned long long)stats.coefficient_bytes,
          (unsigned long long)(stats.seed_domain_fallbacks - before.seed_domain_fallbacks),
          std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    OwnershipStage(graph, "input");
    CaptureRawGraph(graph, "input");
    d.provisional = graph;  // raw evidence + prediction seeds, never a rejected NLS state
    d.provisional_raw = raw;
    d.provisional_history = history;
    d.have_provisional = true;
    graph.iterations = settings.max_iteration;
    if (const char* dir = std::getenv("VIO_POSTERIOR_ONLINE_GRAPH_DIR")) {
      static bool captured = false;
      if (!captured && graph.nodes.size() == 4) {
        SaveGraph(graph, std::string(dir) + "/online4.bin");
        captured = true;
      }
    }
    const bool exact_audit = ExactAuditRequested(graph.times[graph.times.size() - 2], time);
    if (exact_audit) ExactAuditSave(graph, "window_input");
    const auto optimize_start = Clock::now();
    const double prepare_ms = std::chrono::duration<double, std::milli>(optimize_start - begin).count();
    // No mean/factor mutation occurs between this solve and posterior
    // extraction. Reuse only this call's final, undamped linearization.
    const bool reuse = std::getenv("VIO_POSTERIOR_REUSE_LINEARIZATION");
    Linearization final_linearization;
    finite_geometry::Stats finite;
    const auto report = finite_geometry::OptimizeFinite(graph, reuse ? &final_linearization : nullptr, finite);
    if (finite_geometry::Enabled()) {
      TraceMessage("FINITE_FRAME t=%lld points=%zu deferred=%zu boundary=%d rank=%d numerical=%d geometry_ms=%.6f",
                   (long long)time, graph.points.size(), finite.deferred.size(), finite.boundary, finite.unobservable,
                   finite.numerical, finite.geometry_ms);
      for (const auto& item : finite.deferred) {
        TraceMessage("FINITE_DEFER t=%lld id=%lld reason=%d", (long long)time, (long long)item.first, int(item.second));
      }
    }
    const auto optimized = Clock::now();
    CaptureRawGraph(graph, "optimized");
    if (exact_audit) ExactAuditSave(graph, "window_optimized");
    if (!report.success || !Publishable(graph) || !VisualConsensus(graph, rig, settings.min_observations, time)) {
      return false;
    }
    const auto support_end = Clock::now();
    Matrix covariance;
    Vector gradient;
    if (!CurrentMarginal(graph, time, covariance, gradient, reuse ? &final_linearization : nullptr)) {
      return false;
    }
    const auto covariance_end = Clock::now();
    int length = 4;
    if (const char* value = std::getenv("CUVSLAM_WINDOW_SIZE")) {
      length = std::stoi(value);
    }
    if (length < 2) {
      throw std::invalid_argument("invalid joint lag length");
    }
    auto retained = RetainSuffix(graph, length - 1, d.historical_conditionals ? &history : nullptr);
    OwnershipStage(retained, "retained");
    if (exact_audit) ExactAuditSave(retained, "window_retained");
    const auto retained_end = Clock::now();
    TraceMessage(
        "POSTERIOR_TIMING t=%lld prepare=%.6f optimize=%.6f linearize=%.6f assemble=%.6f solve=%.6f retract=%.6f "
        "support=%.6f covariance=%.6f marginalize=%.6f states=%zu points=%zu visuals=%zu iterations=%d",
        (long long)time, prepare_ms, std::chrono::duration<double, std::milli>(optimized - optimize_start).count(),
        report.linearize_ms, report.assemble_ms, report.linear_solve_ms, report.retract_ms,
        std::chrono::duration<double, std::milli>(support_end - optimized).count(),
        std::chrono::duration<double, std::milli>(covariance_end - support_end).count(),
        std::chrono::duration<double, std::milli>(retained_end - covariance_end).count(), graph.nodes.size(),
        graph.points.size(), graph.visuals.size(), report.iterations);
    const auto gpu_after = cuvslam::pnp::joint_geometry::dense::GetGpuMetrics();
    TraceMessage(
        "POSTERIOR_GPU t=%lld gram=%llu solve=%llu project=%llu compress=%llu failures=%llu rank_fallbacks=%llu "
        "allocations=%llu",
        (long long)time, (unsigned long long)(gpu_after.gram - gpu_before.gram),
        (unsigned long long)(gpu_after.solve - gpu_before.solve),
        (unsigned long long)(gpu_after.project - gpu_before.project),
        (unsigned long long)(gpu_after.compress - gpu_before.compress),
        (unsigned long long)(gpu_after.failures - gpu_before.failures),
        (unsigned long long)(gpu_after.rank_fallbacks - gpu_before.rank_fallbacks),
        (unsigned long long)(gpu_after.allocations - gpu_before.allocations));
    d.publish = graph;
    d.pending = std::move(retained);
    d.pending_history = std::move(history);
    d.pending_raw.clear();
    for (const auto& frame : raw) {
      if (frame.time >= d.pending.times.front()) {
        d.pending_raw.push_back(frame);
      }
    }
    d.pending_covariance = covariance;
    d.have_pending = true;
    candidate = PoseView(graph.nodes.back(), &covariance);
    TraceMessage("JOINT_POSTERIOR_FRAME t=%lld key=%d before=%.9g after=%.9g iterations=%d frames=%zu points=%zu",
                 static_cast<long long>(time), int(is_keyframe), report.before, report.after, report.iterations,
                 graph.nodes.size(), graph.points.size());
    return true;
  } catch (const std::exception& error) {
    TraceMessage("JOINT_POSTERIOR_REJECT t=%lld detail=%s", static_cast<long long>(time), error.what());
    return false;
  }
}

bool Online::commit(map::UnifiedMap& map, const imu::ImuCalibration& calibration, int64_t time) {
  auto& d = *impl_;
  if (!d.have_pending) {
    return true;
  }
  if (d.pending.times.back() != time) {
    d.have_pending = false;
    return false;
  }
  const auto sub = map.get_recent_submap(map.capacity());
  const auto gravity = map.get_gravity();
  if (!gravity || !d.expected_gravity || !(*gravity == *d.expected_gravity)) {
    d.have_pending = false;
    return false;
  }
  // A new current keyframe and capacity eviction are permitted. Changes to
  // any still-present pre-existing mean invalidate this transaction.
  for (const auto& key : sub.consecutive_keyframes) {
    const auto expected = d.expected_states.find(key.keyframe->time_ns());
    if (expected != d.expected_states.end()) {
      const auto state = key.keyframe->get_state();
      const auto& old = expected->second;
      if (!(state.rig_from_w.matrix() == old.rig_from_w.matrix()) || !(state.velocity == old.velocity) ||
          !(state.gyro_bias == old.gyro_bias) || !(state.acc_bias == old.acc_bias)) {
        d.have_pending = false;
        return false;
      }
    }
  }
  for (const auto& frame : sub.landmark_and_obs) {
    for (const auto& entry : frame) {
      const auto expected = d.expected_points.find(static_cast<int64_t>(entry.landmark->id()));
      if (expected != d.expected_points.end()) {
        const auto actual = entry.landmark->get_pose();
        if (!actual || !(*actual == expected->second)) {
          d.have_pending = false;
          return false;
        }
      }
    }
  }
  std::map<int64_t, State> states;
  for (size_t i = 0; i < d.publish.times.size(); ++i) {
    states.emplace(d.publish.times[i], d.publish.nodes[i]);
  }
  std::map<int64_t, Eigen::Vector3d> points;
  for (size_t i = 0; i < d.publish.ids.size(); ++i) {
    points.emplace(d.publish.ids[i], d.publish.points[i]);
  }
  std::map<int64_t, sba_imu::IMUPreintegration> refreshed;
  for (const auto& key : sub.consecutive_keyframes) {
    const auto found = states.find(key.keyframe->time_ns());
    if (found != states.end() && key.preintegration) {
      auto integration = *key.preintegration;
      const auto pose = PoseView(found->second, nullptr);
      integration.SetNewBias(pose.gyro_bias, pose.acc_bias);
      integration.Reintegrate(calibration);
      if (!integration.dR.allFinite() || !integration.dV.allFinite() || !integration.dP.allFinite()) {
        d.have_pending = false;
        return false;
      }
      refreshed.emplace(key.keyframe->time_ns(), std::move(integration));
    }
  }
  // All numerical work precedes mutations. The serial VIO caller owns map
  // writes here; SLAM/PGO and legacy SBA are disabled for this reference mode.
  for (const auto& key : sub.consecutive_keyframes) {
    const auto found = states.find(key.keyframe->time_ns());
    if (found != states.end()) {
      const auto pose = PoseView(found->second, nullptr);
      key.keyframe->set_state(map::State{(pose.w_from_imu * calibration.rig_from_imu().inverse()).inverse(),
                                         pose.velocity, pose.acc_bias, pose.gyro_bias});
      if (key.preintegration) {
        *key.preintegration = refreshed.at(key.keyframe->time_ns());
      }
    }
  }
  for (const auto& frame : sub.landmark_and_obs) {
    for (const auto& entry : frame) {
      const auto found = points.find(static_cast<int64_t>(entry.landmark->id()));
      if (found != points.end()) {
        entry.landmark->set_pose(found->second.cast<float>());
      }
    }
  }
  map.set_gravity(d.pending.gravity.cast<float>());
  d.active = std::move(d.pending);
  d.history = std::move(d.pending_history);
  d.covariance = std::move(d.pending_covariance);
  d.raw = std::move(d.pending_raw);
  d.have_provisional = d.have_pending = false;
  TraceMessage("JOINT_POSTERIOR_COMMIT t=%lld states=%zu points=%zu", static_cast<long long>(time),
               d.active.nodes.size(), d.active.points.size());
  return true;
}

bool Online::pose_at(int64_t time, sba_imu::Pose& pose) const {
  const auto& d = *impl_;
  if (!d.ready) {
    return false;
  }
  const auto found = std::find(d.active.times.begin(), d.active.times.end(), time);
  if (found == d.active.times.end()) {
    return false;
  }
  pose =
      PoseView(d.active.nodes[found - d.active.times.begin()], time == d.active.times.back() ? &d.covariance : nullptr);
  return true;
}
bool Online::prediction_at(int64_t time, sba_imu::Pose& pose) const {
  if (impl_->prediction_time != time) {
    return false;
  }
  pose = impl_->prediction;
  return true;
}
std::optional<Matrix3T> Online::position_covariance(int64_t time) const {
  const auto& d = *impl_;
  if (!d.ready || d.active.times.back() != time || d.covariance.rows() != 15) {
    return std::nullopt;
  }
  const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(d.active.nodes.back().R);
  return (rotation * d.covariance.block<3, 3>(3, 3) * rotation.transpose()).cast<float>();
}
}  // namespace gpu_window::posterior

namespace gpu_window::posterior {
Graph RetainForProfile(const Graph& g, int n) { return RetainSuffix(g, n, nullptr); }
}  // namespace gpu_window::posterior
