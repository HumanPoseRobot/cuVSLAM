#include "window_backend.h"
#include "window_commit_validation.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include "common/log.h"
#include "window_fp64.h"
#include "window_linear.h"
#include "window_math.h"
#include "window_validation.h"
namespace cuvslam::sba_imu {
using WMatrix = Eigen::MatrixXd;
using WVector = Eigen::VectorXd;
using WinProblem = gpu_window::Problem;
using WinResult = gpu_window::Result;
using WinEvaluation = gpu_window::Evaluation;
using gpu_window::assemble;
using gpu_window::compress_prior;
using gpu_window::Dim;
using gpu_window::eliminate_first;
using gpu_window::Evaluate;
using gpu_window::Optimize;
using gpu_window::pinverse;
using gpu_window::position_consistent;
using gpu_window::project_out;
using gpu_window::Slots;
using gpu_window::square_root;
namespace {
using namespace gpu_window;
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
bool anchor_enabled() {
  const char* value = std::getenv("CUVSLAM_GRAVITY_ANCHOR");
  return value && std::strcmp(value, "1") == 0;
}
int window_limit() {
  // Navigation kernels launch 128 threads and use one for the right-hand side,
  // so eight keyframes (120 states) still fit. Frozen cameras do not consume those slots.
  if (anchor_enabled()) {
    return Slots;
  }
  const char* value = std::getenv("CUVSLAM_WINDOW_SIZE");
  if (value && std::strcmp(value, "3") == 0) {
    return 3;
  }
  if (value && std::strcmp(value, "4") == 0) {
    return 4;
  }
  return Slots;
}
bool fp64_window() {
  const char* s = std::getenv("CUVSLAM_WINDOW_SOLVER");
  return s && (std::strcmp(s, "cpu_fp64") == 0 || std::strcmp(s, "schur_fp64") == 0);
}
bool fp64_gpu() {
  const char* s = std::getenv("CUVSLAM_WINDOW_SOLVER");
  return s && std::strcmp(s, "schur_fp64") == 0;
}
bool cpu_window() {
  const char* value = std::getenv("CUVSLAM_WINDOW_SOLVER");
  return value && std::strcmp(value, "cpu") == 0;
}
bool schur_window() {
  const char* value = std::getenv("CUVSLAM_WINDOW_SOLVER");
  return value && std::strcmp(value, "schur") == 0;
}
WinEvaluation evaluate_window(const WinProblem& p, const WinResult& r) {
  // Share the existing posterior evaluation path in the FP64 ablation.
  return (cpu_window() || fp64_window()) ? EvaluateCpu(p, r) : Evaluate(p, r);
}
template <class M>
void copy_matrix(double* p, const M& m) {
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      p[i * m.cols() + j] = m(i, j);
    }
  }
}
gpu_window::State pack_state(const Pose& p) {
  gpu_window::State s;
  copy_matrix(s.R, p.w_from_imu.linear());
  copy_matrix(s.p, p.w_from_imu.translation());
  copy_matrix(s.v, p.velocity);
  copy_matrix(s.bg, p.gyro_bias);
  copy_matrix(s.ba, p.acc_bias);
  return s;
}
void update_pose(Pose& p, const float* x) {
  auto a = pack_state(p);
  double z[15];
  for (int k = 0; k < 15; ++k) {
    z[k] = x[k];
  }
  double R[9], t[3], v[3];
  gpu_window::pose(a, z, R, t, v);
  for (int i = 0; i < 3; ++i) {
    p.w_from_imu.translation()[i] = t[i];
    p.velocity[i] = v[i];
    p.gyro_bias[i] += z[9 + i];
    p.acc_bias[i] += z[12 + i];
    for (int j = 0; j < 3; ++j) {
      p.w_from_imu.linear()(i, j) = R[3 * i + j];
    }
  }
}
template <int N>
void whiten(double* dst, const Eigen::Matrix<double, N, N>& m) {
  WMatrix W;
  WVector c;
  square_root(m, WVector::Zero(N), W, c);
  copy_matrix(dst, W);
}
struct Node {
  int64_t t;
  Pose pose;
  std::vector<camera::Observation> observations;
};
struct FixedView {
  int64_t t = 0;
  gpu_window::State pose;
  std::vector<camera::Observation> observations;
};
struct Graph {
  std::vector<Node> nodes;
  std::vector<FixedView> fixed;
  std::map<TrackId, Vector3T> points;
  std::array<gpu_window::State, Slots> anchors{};
  WMatrix W = WMatrix::Zero(Dim, Dim);
  WVector c = WVector::Zero(Dim);
};
constexpr size_t kMaxFixed = 16;

int shared_tracks(const std::vector<camera::Observation>& obs, const std::set<TrackId>& live) {
  std::set<TrackId> seen;
  int count = 0;
  for (const auto& o : obs) {
    if (live.count(o.id) && seen.insert(o.id).second) {
      ++count;
    }
  }
  return count;
}
void trim_fixed(Graph& g) {
  while (g.fixed.size() > kMaxFixed) {
    std::set<TrackId> live;
    for (const auto& node : g.nodes) {
      for (const auto& o : node.observations) {
        live.insert(o.id);
      }
    }
    auto worst = g.fixed.begin();
    int worst_score = shared_tracks(worst->observations, live);
    for (auto it = g.fixed.begin(); it != g.fixed.end(); ++it) {
      const int score = shared_tracks(it->observations, live);
      if (score < worst_score || (score == worst_score && it->t < worst->t)) {
        worst = it;
        worst_score = score;
      }
    }
    g.fixed.erase(worst);
  }
}
template <class SubMap>
void reseed_fixed(Graph& g, const SubMap& sub, const imu::ImuCalibration& cal) {
  if (!anchor_enabled() || g.nodes.empty() || sub.consecutive_keyframes.size() != sub.landmark_and_obs.size()) {
    return;
  }
  std::set<int64_t> have;
  for (const auto& node : g.nodes) {
    have.insert(node.t);
  }
  for (const auto& view : g.fixed) {
    have.insert(view.t);
  }
  std::set<TrackId> live;
  for (const auto& node : g.nodes) {
    for (const auto& o : node.observations) {
      live.insert(o.id);
    }
  }
  const size_t before = g.fixed.size();
  for (size_t i = 0; i < sub.consecutive_keyframes.size(); ++i) {
    const auto kf = sub.consecutive_keyframes[i].keyframe;
    if (!kf || have.count(kf->time_ns())) {
      continue;
    }
    const int64_t stamp = kf->time_ns();
    FixedView view;
    const auto state = kf->get_state();
    Pose pose;
    pose.w_from_imu = state.rig_from_w.inverse() * cal.rig_from_imu();
    pose.velocity = state.velocity;
    pose.gyro_bias = state.gyro_bias;
    pose.acc_bias = state.acc_bias;
    view.t = stamp;
    view.pose = pack_state(pose);
    for (const auto& entry : sub.landmark_and_obs[i]) {
      const auto point = entry.landmark->get_pose();
      if (!point) {
        continue;
      }
      for (const auto& o : entry.observations) {
        if (!live.count(o.id)) {
          continue;
        }
        view.observations.push_back(o);
        g.points.emplace(o.id, *point);
      }
    }
    if (shared_tracks(view.observations, live) < 8) {
      continue;
    }
    g.fixed.push_back(std::move(view));
    have.insert(stamp);
  }
  trim_fixed(g);
  if (g.fixed.size() != before) {
    std::fprintf(stderr, "GRAVITY_ANCHOR reseed fixed=%zu\n", g.fixed.size());
  }
}
struct MapSupport {
  int before = 0;
  int after = 0;

  bool preserved() const {
    // Allow reprojections near the inlier threshold to change classification.
    return before < 20 || after >= before - std::max(2, before / 20);
  }
};

bool reprojection_inlier(const camera::Rig& rig, const Isometry3T& rig_from_world,
                         const camera::Observation& observation, const Vector3T& point) {
  if (observation.cam_id >= rig.num_cameras) {
    return false;
  }
  const auto camera = observation.cam_id;
  const Vector3T pc = rig.camera_from_rig[camera] * rig_from_world * point;
  if (!pc.allFinite() || pc.z() <= .05f) {
    return false;
  }
  const Vector2T error = pc.hnormalized() - observation.xy;
  return error.dot(9.f * observation.xy_info * error) <= 9.21f &&
         error.cwiseProduct(rig.intrinsics[camera]->getFocal()).squaredNorm() <= 64.f;
}

bool map_support_preserved(const WindowBackend::FrameInput& input, const Graph& candidate,
                           const std::map<TrackId, Vector3T>& candidate_points, bool enforce) {
  const auto& rig = input.rig;
  const auto& cal = input.calibration;
  const auto to_rig = [&](const Pose& pose) -> Isometry3T { return cal.rig_from_imu() * pose.w_from_imu.inverse(); };
  const auto old_current = to_rig(input.tracked);
  const auto new_current = to_rig(candidate.nodes.back().pose);
  MapSupport current;
  for (const auto& observation : input.observations) {
    const auto old_point = input.landmarks.find(observation.id);
    if (old_point == input.landmarks.end()) {
      continue;
    }
    const auto new_point = candidate_points.find(observation.id);
    current.before += reprojection_inlier(rig, old_current, observation, old_point->second);
    current.after += reprojection_inlier(rig, new_current, observation,
                                         new_point == candidate_points.end() ? old_point->second : new_point->second);
  }

  // Older keyframes can observe the same landmarks without contributing to
  // this window's objective. Check their support before changing shared geometry.
  const auto submap = input.map.get_recent_submap(20);
  MapSupport historical;
  bool historical_ok = true;
  for (size_t i = 0; i < submap.consecutive_keyframes.size(); ++i) {
    const auto& keyframe = submap.consecutive_keyframes[i].keyframe;
    const auto node = std::find_if(candidate.nodes.begin(), candidate.nodes.end(),
                                   [&](const Node& value) { return value.t == keyframe->time_ns(); });
    const Isometry3T old_pose = keyframe->get_state().rig_from_w;
    const Isometry3T new_pose = node == candidate.nodes.end() ? old_pose : to_rig(node->pose);
    MapSupport row;
    for (const auto& entry : submap.landmark_and_obs[i]) {
      const auto old_point = entry.landmark->get_pose();
      if (!old_point) {
        continue;
      }
      const auto new_point = candidate_points.find(entry.landmark->id());
      for (const auto& observation : entry.observations) {
        row.before += reprojection_inlier(rig, old_pose, observation, *old_point);
        row.after += reprojection_inlier(rig, new_pose, observation,
                                         new_point == candidate_points.end() ? *old_point : new_point->second);
      }
    }
    historical.before += row.before;
    historical.after += row.after;
    historical_ok &= row.preserved();
  }
  const bool accepted = current.preserved() && historical_ok;
  TraceMessage(
      "WINDOW_MAP_GUARD t=%lld enforce=%d accept=%d current_before=%d current_after=%d "
      "history_before=%d history_after=%d history_ok=%d points=%zu",
      (long long)input.time_ns, int(enforce), int(accepted), current.before, current.after, historical.before,
      historical.after, int(historical_ok), candidate_points.size());
  return !enforce || accepted;
}

WinProblem pack(Graph& g, const imu::ImuCalibration& cal, const camera::Rig& rig, const Vector3T& gravity,
                int iterations, double huber, std::vector<TrackId>& ids) {
  int invalid_depth = 0;
  for (auto& node : g.nodes) {
    auto& observations = node.observations;
    observations.erase(std::remove_if(observations.begin(), observations.end(),
                                      [&](const auto& o) {
                                        auto it = g.points.find(o.id);
                                        if (it == g.points.end() || o.cam_id >= rig.num_cameras) {
                                          ++invalid_depth;
                                          return true;
                                        }
                                        const Vector3T pc = rig.camera_from_rig[o.cam_id] * cal.rig_from_imu() *
                                                            node.pose.w_from_imu.inverse() * it->second;
                                        if (!pc.allFinite() || pc.z() <= .05f) {
                                          ++invalid_depth;
                                          return true;
                                        }
                                        return false;
                                      }),
                       observations.end());
  }
  if (invalid_depth) {
    TraceMessage("WINDOW_INVALID_DEPTH removed=%d", invalid_depth);
  }
  WinProblem p;
  p.count = g.nodes.size();
  p.iterations = iterations;
  p.huber = huber;
  p.anchors = g.anchors;
  copy_matrix(p.W.data(), g.W);
  copy_matrix(p.offset.data(), g.c);
  for (int i = 0; i < Slots; ++i) {
    if (i < p.count) {
      p.nodes[i] = pack_state(g.nodes[i].pose);
    } else {
      p.anchors[i] = p.nodes[i];
      for (int j = 0; j < 15; ++j) {
        p.W[(i * 15 + j) * Dim + i * 15 + j] = 1;
      }
    }
  }
  std::map<TrackId, int> counts;
  for (auto& n : g.nodes) {
    for (auto& o : n.observations) {
      ++counts[o.id];
    }
  }
  for (auto& view : g.fixed) {
    for (auto& o : view.observations) {
      if (g.points.count(o.id)) {
        ++counts[o.id];
      }
    }
  }
  std::map<TrackId, int> index;
  for (auto& kv : g.points) {
    if (counts[kv.first] >= 2) {
      index[kv.first] = ids.size();
      ids.push_back(kv.first);
      p.points.push_back({kv.second[0], kv.second[1], kv.second[2]});
    }
  }
  for (int i = 0; i < p.count; ++i) {
    for (auto& o : g.nodes[i].observations) {
      auto it = index.find(o.id);
      if (it == index.end()) {
        continue;
      }
      Visual v;
      v.node = i;
      v.point = it->second;
      copy_matrix(v.obs.xy, o.xy);
      auto ci = Isometry3T(rig.camera_from_rig[o.cam_id] * cal.rig_from_imu());
      copy_matrix(v.obs.Rci, ci.linear());
      copy_matrix(v.obs.tci, ci.translation());
      whiten<2>(v.obs.sqrt_info, 9. * o.xy_info.template cast<double>());
      v.obs.min_depth = .05;
      p.visuals.push_back(v);
    }
  }
  for (auto& view : g.fixed) {
    Matrix3T rotation;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        rotation(row, col) = view.pose.R[3 * row + col];
      }
    }
    const Vector3T translation(view.pose.p[0], view.pose.p[1], view.pose.p[2]);
    for (auto& o : view.observations) {
      auto it = index.find(o.id);
      if (it == index.end() || o.cam_id >= rig.num_cameras || !g.points.count(o.id)) {
        continue;
      }
      const Isometry3T camera(rig.camera_from_rig[o.cam_id] * cal.rig_from_imu());
      const Vector3T pc = camera.linear() * (rotation.transpose() * (g.points[o.id] - translation)) + camera.translation();
      if (!pc.allFinite() || pc.z() <= .05f) {
        continue;
      }
      Visual v;
      v.node = -1;
      v.point = it->second;
      v.fixed = view.pose;
      copy_matrix(v.obs.xy, o.xy);
      copy_matrix(v.obs.Rci, camera.linear());
      copy_matrix(v.obs.tci, camera.translation());
      whiten<2>(v.obs.sqrt_info, 9. * o.xy_info.template cast<double>());
      v.obs.min_depth = .05;
      p.visuals.push_back(v);
    }
  }
  for (int i = 0; i + 1 < p.count; ++i) {
    auto& l = g.nodes[i].pose;
    auto& r = g.nodes[i + 1].pose;
    auto& pi = l.preintegration;
    Vector3T db, da;
    pi.GetDeltaBias(l.gyro_bias, l.acc_bias, db, da);
    if (db.norm() > 1e-4 || da.norm() > 1e-4) {
      pi.SetNewBias(l.gyro_bias, l.acc_bias);
      pi.Reintegrate(cal);
    }
    Edge e;
    e.left = i;
    e.right = i + 1;
    auto& d = e.data;
    auto a = pack_state(l), b = pack_state(r);
    std::copy(a.R, a.R + 9, d.R[0]);
    std::copy(b.R, b.R + 9, d.R[1]);
    std::copy(a.p, a.p + 3, d.p[0]);
    std::copy(b.p, b.p + 3, d.p[1]);
    std::copy(a.v, a.v + 3, d.v[0]);
    std::copy(b.v, b.v + 3, d.v[1]);
    copy_matrix(d.bg, l.gyro_bias);
    copy_matrix(d.ba, l.acc_bias);
    copy_matrix(d.bg0, pi.GetOriginalGyroBias());
    copy_matrix(d.ba0, pi.GetOriginalAccBias());
    copy_matrix(d.dR, pi.dR);
    copy_matrix(d.dV, pi.dV);
    copy_matrix(d.dP, pi.dP);
    copy_matrix(d.JRg, pi.JRg);
    copy_matrix(d.JVg, pi.JVg);
    copy_matrix(d.JVa, pi.JVa);
    copy_matrix(d.JPg, pi.JPg);
    copy_matrix(d.JPa, pi.JPa);
    copy_matrix(d.gravity, gravity);
    d.dt = pi.dT_s;
    Matrix9T inf;
    pi.InfoMatrix(inf);
    Eigen::Matrix<double, 9, 9> scaled = inf.cast<double>();
    double gyro_rw = 0.002;
    double acc_rw = 0.002;
    if (anchor_enabled()) {
      // Rotation stays at full preintegration weight. Position and velocity
      // drop to 0.05 so a fixed view can update velocity through (\u0394p)/dt.
      constexpr double kTrans = 0.05;
      Eigen::Matrix<double, 9, 9> S = Eigen::Matrix<double, 9, 9>::Identity();
      S.block<3, 3>(3, 3) *= std::sqrt(kTrans);
      S.block<3, 3>(6, 6) *= std::sqrt(kTrans);
      scaled = S * scaled * S;
      gyro_rw = 1.0;
      acc_rw = kTrans;
    } else {
      scaled *= 0.002;
    }
    whiten<9>(d.imu_sqrt, scaled);
    Matrix3T bg, ba;
    pi.InfoGyroRWMatrix(bg);
    pi.InfoAccRWMatrix(ba);
    Eigen::Matrix<double, 6, 6> rw = Eigen::Matrix<double, 6, 6>::Zero();
    rw.topLeftCorner<3, 3>() = gyro_rw * bg.cast<double>();
    rw.bottomRightCorner<3, 3>() = acc_rw * ba.cast<double>();
    whiten<6>(e.rw, rw);
    p.edges.push_back(e);
  }
  return p;
}
double cost(const WinEvaluation& e, double huber) {
  double s = 0;
  for (auto x : e.edge) {
    s += .5 * x * x;
  }
  for (auto x : e.prior) {
    s += .5 * x * x;
  }
  for (size_t i = 0; i < e.visual.size(); i += 2) {
    double n = std::hypot(e.visual[i], e.visual[i + 1]);
    s += n > huber ? huber * n - .5 * huber * huber : .5 * n * n;
  }
  return s;
}

// Prepare the retained graph before exposing the current candidate to the
// frontend. A failed posterior/marginalization must never publish a pose whose
// map or prior cannot subsequently be committed.
bool prepare_retained(Graph& g, const imu::ImuCalibration& cal, const camera::Rig& rig, const Vector3T& gravity,
                      int iterations, double huber, int64_t t) {
  if (g.nodes.size() == static_cast<size_t>(window_limit())) {
    FixedView frozen;
    const bool anchoring = anchor_enabled();
    if (anchoring) {
      // Keep this pose as a visual anchor. Its pixels must not also enter the marginal prior.
      frozen.t = g.nodes.front().t;
      frozen.pose = pack_state(g.nodes.front().pose);
      frozen.observations = g.nodes.front().observations;
      g.nodes.front().observations.clear();
    }
    std::vector<TrackId> ids;
    auto p = pack(g, cal, rig, gravity, iterations, huber, ids);
    WinResult zero;
    zero.points.resize(3 * ids.size());
    auto linear = evaluate_window(p, zero);
    if (!valid_evaluation(linear)) {
      TraceMessage("WINDOW_TRANSACTION_REJECT t=%lld reason=invalid_marginal_jacobian", (long long)t);
      return false;
    }
    const auto point_start = Clock::now();
    const auto point_reduced = assemble(p, linear, true);
    const double point_ms = elapsed(point_start);
    const auto marginal_start = Clock::now();
    auto reduced = eliminate_first(point_reduced, p.count);
    const int n = 15 * (p.count - 1);
    WMatrix W;
    WVector c;
    compress_prior(reduced.rows.leftCols(n), reduced.residual, W, c);
    if (!W.allFinite() || !c.allFinite() || W.rows() != n || W.cols() != n || c.size() != n) {
      TraceMessage("WINDOW_TRANSACTION_REJECT t=%lld reason=invalid_marginal_prior", (long long)t);
      return false;
    }
    const double marginal_ms = elapsed(marginal_start);
    TraceMessage("SCALE_MARG t=%lld states=%d points=%zu point_ms=%.6f marginal_ms=%.6f", (long long)t, p.count,
                 ids.size(), point_ms, marginal_ms);
    if (anchoring) {
      g.nodes.erase(g.nodes.begin());
      if (!frozen.observations.empty()) {
        g.fixed.push_back(std::move(frozen));
      }
      trim_fixed(g);
      std::set<TrackId> used;
      for (const auto& node : g.nodes) {
        for (const auto& o : node.observations) {
          used.insert(o.id);
        }
      }
      for (const auto& view : g.fixed) {
        for (const auto& o : view.observations) {
          used.insert(o.id);
        }
      }
      for (auto it = g.points.begin(); it != g.points.end();) {
        if (!used.count(it->first)) {
          it = g.points.erase(it);
        } else {
          ++it;
        }
      }
      static int freezes = 0;
      if (freezes++ % 20 == 0) {
        std::fprintf(stderr, "GRAVITY_ANCHOR freeze fixed=%zu points=%zu\n", g.fixed.size(), g.points.size());
      }
    } else {
      std::set<TrackId> remove;
      for (auto& o : g.nodes.front().observations) {
        remove.insert(o.id);
      }
      g.nodes.erase(g.nodes.begin());
      for (auto& node : g.nodes) {
        auto& v = node.observations;
        v.erase(std::remove_if(v.begin(), v.end(), [&](auto& o) { return remove.count(o.id); }), v.end());
      }
      for (auto id : remove) {
        g.points.erase(id);
      }
    }
    g.W.setZero();
    g.c.setZero();
    g.W.topLeftCorner(n, n) = W;
    g.c.head(n) = c;
    for (int i = 0; i < p.count - 1; ++i) {
      g.anchors[i] = pack_state(g.nodes[i].pose);
    }
    TraceMessage("WINDOW_MARG t=%lld retained=%zu points=%zu fixed=%zu prior_norm=%.9g linear_norm=%.9g", (long long)t,
                 g.nodes.size(), g.points.size(), g.fixed.size(), g.W.norm(), g.c.norm());
  }
  return g.W.allFinite() && g.c.allFinite();
}

}  // namespace
struct WindowBackend::Impl {
  Graph active, pending;
  std::vector<Node> pending_publish_nodes;
  std::map<TrackId, Vector3T> pending_points;
  std::map<int64_t, map::State> map_anchors;
  std::map<int64_t, map::State> pending_expected;
  bool pending_map_write = true;
  bool have_pending = false;
  Vector3T gravity;
  imu::ImuCalibration cal;
  camera::Rig rig;
  int iterations = 7;
  double huber = 1;
  size_t commits = 0;
};
WindowBackend::WindowBackend() : impl_(std::make_unique<Impl>()) {}
WindowBackend::~WindowBackend() = default;
void WindowBackend::discard_pending() {
  impl_->have_pending = false;
  impl_->pending = Graph{};
  impl_->pending_points.clear();
  impl_->pending_publish_nodes.clear();
  impl_->pending_expected.clear();
}
void WindowBackend::reset() { impl_ = std::make_unique<Impl>(); }
bool WindowBackend::solve(const FrameInput& input, Pose& result) {
  auto& map = input.map;
  const auto& cal = input.calibration;
  const auto& rig = input.rig;
  const auto& storage = input.measurements;
  const auto& landmarks = input.landmarks;
  const auto& obs = input.observations;
  const auto& gravity = input.gravity;
  const auto& settings = input.settings;
  const auto& previous = input.previous;
  const auto& tracked = input.tracked;
  const int64_t previous_t = input.previous_time_ns, t = input.time_ns;
  if (!valid_state(pack_state(previous)) || !valid_state(pack_state(tracked)) || !tracked.info.allFinite() ||
      !gravity.allFinite() || gravity.norm() < 1e-6) {
    discard_pending();
    TraceMessage("WINDOW_NUMERIC_REJECT t=%lld stage=frame_input", (long long)t);
    return false;
  }

  discard_pending();
  auto& d = *impl_;
  if (previous_t < 0 || t <= previous_t) {
    reset();
    return false;
  }
  if (!d.active.nodes.empty() && (d.gravity - gravity).norm() > 1e-4) {
    const char* gravity_scheme = std::getenv("CUVSLAM_GRAVITY_SCHEME");
    const float alignment = d.gravity.normalized().dot(gravity.normalized());
    const bool small_tangent =
        gravity_scheme && std::strcmp(gravity_scheme, "tangent") == 0 && alignment > 0.9993908f;
    if (!small_tangent) {
      reset();
      return solve(input, result);
    }
    d.gravity = gravity;
  }
  if (!d.active.nodes.empty()) {
    auto recent = map.get_recent_submap(20);
    if (!recent.consecutive_keyframes.empty()) {
      const int64_t published = recent.consecutive_keyframes.back().keyframe->time_ns();
      if (published > d.active.nodes.back().t) {
        // A rejected window leaves the validated frontend free to publish its
        // keyframe. Do not bridge subsequent IMU factors across that skipped
        // keyframe indefinitely using the old local linearization/prior.
        // Re-seed from the authoritative published state, with a fresh prior.
        TraceMessage("WINDOW_RESEED t=%lld reason=published_keyframe_ahead active=%lld published=%lld", (long long)t,
                     (long long)d.active.nodes.back().t, (long long)published);
        reset();
        return solve(input, result);
      }
    }
    bool changed = false;
    for (auto& k : recent.consecutive_keyframes) {
      for (auto& n : d.active.nodes) {
        if (k.keyframe->time_ns() != n.t) {
          continue;
        }
        auto state = k.keyframe->get_state();
        auto anchor = d.map_anchors.find(n.t);
        if (anchor == d.map_anchors.end()) {
          continue;
        }
        const auto& expected = anchor->second;
        changed = changed || (state.rig_from_w.matrix() - expected.rig_from_w.matrix()).norm() > 1e-4f ||
                  (state.velocity - expected.velocity).norm() > 1e-4f ||
                  (state.gyro_bias - expected.gyro_bias).norm() > 1e-4f ||
                  (state.acc_bias - expected.acc_bias).norm() > 1e-4f;
      }
    }
    if (changed) {
      TraceMessage("WINDOW_RESET reason=external_map_correction t=%lld", (long long)t);
      reset();
      return solve(input, result);
    }
  }
  d.cal = cal;
  d.rig = rig;
  d.gravity = gravity;
  d.iterations = settings.max_iteration;
  d.huber = settings.robustifier_scale;
  if (d.active.nodes.empty()) {
    Node n{previous_t, previous, {}};
    auto seed = map.get_recent_submap(1);
    if (!seed.consecutive_keyframes.empty()) {
      auto k = seed.consecutive_keyframes.back().keyframe;
      auto state = k->get_state();
      n.t = k->time_ns();
      n.pose.w_from_imu = state.rig_from_w.inverse() * cal.rig_from_imu();
      n.pose.velocity = state.velocity;
      n.pose.gyro_bias = state.gyro_bias;
      n.pose.acc_bias = state.acc_bias;
      for (auto& row : seed.landmark_and_obs) {
        for (auto& entry : row) {
          auto point = entry.landmark->get_pose();
          if (!point) {
            continue;
          }
          d.active.points.emplace(entry.landmark->id(), *point);
          for (auto& o : entry.observations) {
            n.observations.push_back(o);
          }
        }
      }
    }
    d.active.nodes.push_back(n);
    d.active.anchors[0] = pack_state(n.pose);
    WVector diag(15);
    diag << 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 25, 25, 25, 1e4, 1e4, 1e4, 25, 25, 25;
    d.active.W.topLeftCorner(15, 15) = diag.cwiseSqrt().asDiagonal();
    if (anchor_enabled()) {
      reseed_fixed(d.active, map.get_recent_submap(20), cal);
    }
  }
  Graph g = d.active;
  auto& last = g.nodes.back();
  if (t <= last.t) {
    reset();
    return false;
  }
  last.pose.preintegration = IMUPreintegration(cal, storage, last.pose.gyro_bias, last.pose.acc_bias, last.t, t);
  double dt = (t - last.t) * 1e-9;
  if (std::abs(last.pose.preintegration.dT_s - dt) > 1e-5 || last.pose.preintegration.empty()) {
    reset();
    return false;
  }
  Pose predicted = tracked;
  Node current{t, predicted, {}};
  const Isometry3T rig_from_world = cal.rig_from_imu() * predicted.w_from_imu.inverse();
  std::set<std::pair<TrackId, int>> seen;
  for (auto& o : obs) {
    auto it = landmarks.find(o.id);
    if (it == landmarks.end() || o.cam_id >= rig.num_cameras || !seen.emplace(o.id, int(o.cam_id)).second) {
      continue;
    }
    auto point = g.points.count(o.id) ? g.points[o.id] : it->second;
    if ((rig.camera_from_rig[o.cam_id] * rig_from_world * point).z() <= .05) {
      continue;
    }
    g.points.emplace(o.id, point);
    current.observations.push_back(o);
  }
  if (current.observations.size() < size_t(settings.min_observations)) {
    return false;
  }
  g.nodes.push_back(current);
  std::vector<TrackId> ids;
  WinResult solved;
  double after = 0;
  for (int pass = 0; pass < 2; ++pass) {
    ids.clear();
    auto p = pack(g, cal, rig, gravity, settings.max_iteration, settings.robustifier_scale, ids);
    if (p.visuals.size() < size_t(settings.min_observations) ||
        g.nodes.back().observations.size() < size_t(settings.min_observations)) {
      return false;
    }
    if (const char* reason = invalid_problem(p)) {
      TraceMessage("WINDOW_NUMERIC_REJECT t=%lld stage=input reason=%s", (long long)t, reason);
      return false;
    }
    const auto optimizer_start = Clock::now();
    if (const char* directory = std::getenv("CUVSLAM_WINDOW_SNAPSHOT_DIR")) {
      // Capture immutable input before either optimizer or outlier pruning runs.
      if (t >= 188000000000LL && t <= 202000000000LL) {
        std::string path =
            std::string(directory) + "/window_" + std::to_string(t) + "_" + std::to_string(pass) + ".bin";
        gpu_window::fp64::SaveProblem(p, path.c_str());
      }
    }
    if (fp64_window()) {
      solved = gpu_window::fp64::Optimize(p, fp64_gpu());
    } else if (cpu_window()) {
      solved = OptimizeCpu(p);
    } else {
      solved = schur_window() ? OptimizeSchur(p) : Optimize(p);
    }
    const double optimizer_ms = elapsed(optimizer_start);
    TraceMessage("SCALE_SOLVE t=%lld pass=%d states=%d points=%zu obs=%zu ms=%.6f schur_ms=%.6f iterations=%d",
                 (long long)t, pass, p.count, ids.size(), p.visuals.size(), optimizer_ms, solved.landmark_ms,
                 solved.iterations);
    if (!valid_result(solved, ids.size())) {
      TraceMessage("WINDOW_NUMERIC_REJECT t=%lld stage=result", (long long)t);
      return false;
    }
    auto evaluation = evaluate_window(p, solved);
    after = cost(evaluation, p.huber);
    if (!std::isfinite(after) || after > solved.before + 1e-4 * std::max(1.f, solved.before)) {
      TraceMessage("WINDOW_REJECT t=%lld reason=objective before=%.9g after=%.9g", (long long)t, solved.before, after);
      return false;
    }
    for (size_t i = 0; i < g.nodes.size(); ++i) {
      update_pose(g.nodes[i].pose, solved.x.data() + 15 * i);
    }
    for (size_t i = 0; i < ids.size(); ++i) {
      for (int k = 0; k < 3; ++k) {
        g.points[ids[i]][k] += solved.points[3 * i + k];
      }
    }
    for (const auto& node : g.nodes) {
      if (!valid_state(pack_state(node.pose))) {
        TraceMessage("WINDOW_NUMERIC_REJECT t=%lld stage=updated_pose", (long long)t);
        return false;
      }
    }
    for (const auto& point : g.points) {
      if (!point.second.allFinite()) {
        TraceMessage("WINDOW_NUMERIC_REJECT t=%lld stage=updated_point", (long long)t);
        return false;
      }
    }
    int removed = 0, behind = 0;
    // Classify individual reprojections, rather than rejecting a whole window
    // because one triangulated outlier crosses the camera plane. 9.21 is the
    // 99% two-dimensional Gaussian residual threshold under the same visual info.
    for (auto& node : g.nodes) {
      auto& observations = node.observations;
      observations.erase(std::remove_if(observations.begin(), observations.end(),
                                        [&](const auto& o) {
                                          auto it = g.points.find(o.id);
                                          if (it == g.points.end()) {
                                            ++removed;
                                            return true;
                                          }
                                          const Vector3T pc = rig.camera_from_rig[o.cam_id] * cal.rig_from_imu() *
                                                              node.pose.w_from_imu.inverse() * it->second;
                                          if (!pc.allFinite() || pc.z() <= .05f) {
                                            ++removed;
                                            ++behind;
                                            return true;
                                          }
                                          const Vector2T error = pc.hnormalized() - o.xy;
                                          const double score = error.cast<double>().dot(
                                              9. * o.xy_info.template cast<double>() * error.cast<double>());
                                          if (!std::isfinite(score) || score > 9.21) {
                                            ++removed;
                                            return true;
                                          }
                                          return false;
                                        }),
                         observations.end());
    }
    TraceMessage("WINDOW_PRUNE t=%lld pass=%d removed=%d behind=%d current_obs=%zu", (long long)t, pass, removed,
                 behind, g.nodes.back().observations.size());
    if (g.nodes.back().observations.size() < size_t(settings.min_observations)) {
      return false;
    }
    if (!removed) {
      break;
    }
  }
  // Restore the existing repaired backend's same-timestamp position-consistency gate.
  // Tracking and window factors share measurements: this is an engineering gate,
  // not an independent-sensor chi-square claim. Reject the entire transaction,
  // retaining the old window and the separately computed foreground estimate.
  Eigen::MatrixXd tracking_info = tracked.info.cast<double>();
  Eigen::LDLT<Eigen::MatrixXd> posterior(tracking_info);
  if (posterior.info() != Eigen::Success || !posterior.isPositive() || posterior.vectorD().minCoeff() <= 0) {
    TraceMessage("WINDOW_GATE t=%lld accept=0 reason=invalid_tracking_covariance", (long long)t);
    return false;
  }
  Eigen::MatrixXd select = Eigen::MatrixXd::Zero(15, 3);
  select.block<3, 3>(3, 0).setIdentity();
  Eigen::Matrix3d local_cov = posterior.solve(select).block<3, 3>(3, 0);
  Eigen::Matrix3d rotation = tracked.w_from_imu.linear().cast<double>();
  Eigen::Matrix3d covariance = rotation * local_cov * rotation.transpose();
  Eigen::Vector3d delta =
      (g.nodes.back().pose.w_from_imu.translation() - tracked.w_from_imu.translation()).cast<double>();
  double score = 0;
  bool accept = position_consistent(covariance, delta, score);
  TraceMessage("WINDOW_GATE t=%lld accept=%d score=%.9g dp=%.9g", (long long)t, int(accept), score, delta.norm());
  if (!accept) {
    return false;
  }
  // Re-express final normal equations in the newly anchored tangent coordinates.
  ids.clear();
  auto final = pack(g, cal, rig, gravity, settings.max_iteration, settings.robustifier_scale, ids);
  WinResult zero;
  zero.points.resize(3 * ids.size());
  auto linear = evaluate_window(final, zero);
  if (!valid_evaluation(linear)) {
    TraceMessage("WINDOW_NUMERIC_REJECT t=%lld stage=posterior_jacobian", (long long)t);
    return false;
  }
  const auto elimination_start = Clock::now();
  auto reduced = assemble(final, linear, false);
  TraceMessage("SCALE_POINT_QR t=%lld stage=posterior states=%d points=%zu ms=%.6f", (long long)t, final.count,
               ids.size(), elapsed(elimination_start));
  int n = 15 * (g.nodes.size() - 1);
  WMatrix final_rows;
  WVector final_residual;
  project_out(reduced.rows.leftCols(n), reduced.rows.middleCols(n, 15), reduced.residual, final_rows, final_residual);
  WMatrix info = final_rows.transpose() * final_rows;
  if (!info.allFinite() || info.cwiseAbs().maxCoeff() > std::numeric_limits<float>::max()) {
    TraceMessage("WINDOW_NUMERIC_REJECT t=%lld stage=posterior", (long long)t);
    return false;
  }
  g.nodes.back().pose.info = info.cast<float>();
  result = g.nodes.back().pose;
  if (!result.info.allFinite() || !result.velocity.allFinite() || !result.w_from_imu.matrix().allFinite()) {
    return false;
  }
  const bool enforce_map = true;  // This safety variant never bypasses map acceptance.
  std::map<TrackId, Vector3T> supported_points;
  for (const auto id : ids) {
    supported_points.emplace(id, g.points.at(id));
  }
  const bool map_safe = map_support_preserved(input, g, enforce_map ? supported_points : g.points, enforce_map);
  auto publish_points = enforce_map ? std::move(supported_points) : g.points;
  auto visible_points = publish_points;
  for (const auto& point : input.landmarks) {
    visible_points.emplace(point.first, point.second);
  }
  if (input.accept_candidate && !input.accept_candidate(result, enforce_map ? visible_points : g.points)) {
    return false;
  }
  // A rejected map update also rejects its local solution and derived prior.
  // Keeping only the local branch makes subsequent factors reference geometry
  // different from the frontend's published map.
  if (!map_safe) {
    TraceMessage("WINDOW_TRANSACTION_REJECT t=%lld reason=map_support", (long long)t);
    return false;
  }
  const auto publish_nodes = g.nodes;
  if (!prepare_retained(g, cal, rig, gravity, settings.max_iteration, settings.robustifier_scale, t)) {
    return false;
  }
  // Capture the authoritative map version used by this solve. A commit may
  // follow keyframe insertion, but must not overwrite an intervening correction.
  d.pending_expected.clear();
  for(const auto& k:map.get_recent_submap(20).consecutive_keyframes) {
    const auto kt=k.keyframe->time_ns();
    if(std::any_of(publish_nodes.begin(),publish_nodes.end(),[&](const Node& n){return n.t==kt;})) {
      d.pending_expected.emplace(kt,k.keyframe->get_state());
    }
  }
  d.pending_map_write = true;
  d.pending_publish_nodes = publish_nodes;
  d.pending_points = std::move(publish_points);
  d.pending = std::move(g);
  d.have_pending = true;
  TraceMessage("WINDOW_SOLVE t=%lld states=%zu points=%zu obs=%zu iterations=%d before=%.9g after=%.9g", (long long)t,
               d.pending.nodes.size(), ids.size(), final.visuals.size(), solved.iterations, solved.before, after);
  return true;
}
bool WindowBackend::commit(map::UnifiedMap& map, const imu::ImuCalibration& cal, int64_t t) {
  auto& d = *impl_;
  if (!d.have_pending) {
    return true;
  }
  if (d.pending.nodes.empty() || d.pending.nodes.back().t != t) {
    TraceMessage("WINDOW_TRANSACTION_REJECT t=%lld reason=commit_timestamp", (long long)t);
    discard_pending();
    return false;
  }
  // Validate every destination before the first mutation. The caller must not
  // publish a window candidate if this transaction fails.
  const auto destinations=map.get_recent_submap(20);
  std::map<int64_t,map::State> states;
  for(const auto& k:destinations.consecutive_keyframes) {states.emplace(k.keyframe->time_ns(),k.keyframe->get_state());}
  for(const auto& n:d.pending_publish_nodes) {
    auto it=states.find(n.t);
    if(it==states.end()) {
      TraceMessage("WINDOW_TRANSACTION_REJECT t=%lld reason=commit_missing_keyframe",(long long)t);
      discard_pending();return false;
    }
    auto expected=d.pending_expected.find(n.t);
    if(expected!=d.pending_expected.end()) {
      const auto& a=it->second;const auto& b=expected->second;
      if(!SameCommitState(a,b)) {
        TraceMessage("WINDOW_TRANSACTION_REJECT t=%lld reason=commit_map_changed",(long long)t);
        discard_pending();return false;
      }
    }
  }
  std::map<int64_t,IMUPreintegration> refreshed;
  for(const auto& k:destinations.consecutive_keyframes) {
    if(!k.preintegration) {continue;}
    for(const auto& n:d.pending_publish_nodes) {
      if(k.keyframe->time_ns()!=n.t) {continue;}
      auto pi=*k.preintegration;
      pi.SetNewBias(n.pose.gyro_bias,n.pose.acc_bias);pi.Reintegrate(cal);
      if(!pi.dR.allFinite() || !pi.dV.allFinite() || !pi.dP.allFinite() || !std::isfinite(pi.dT_s)) {
        TraceMessage("WINDOW_TRANSACTION_REJECT t=%lld reason=commit_preintegration",(long long)t);
        discard_pending();return false;
      }
      refreshed.emplace(n.t,std::move(pi));
    }
  }
  Graph g = std::move(d.pending);
  d.have_pending = false;
  // All fallible numerical work completed in solve(), before pose acceptance.
  const auto publish_nodes = std::move(d.pending_publish_nodes);
  const auto publish_points = std::move(d.pending_points);
  auto publish = [&]() {
    auto sub = map.get_recent_submap(20);
    for (auto& k : sub.consecutive_keyframes) {
      for (auto& n : publish_nodes) {
        if (k.keyframe->time_ns() == n.t) {
          if(k.preintegration) {
            auto pi=refreshed.find(n.t);
            if(pi!=refreshed.end()) {*k.preintegration=pi->second;}
          }
          k.keyframe->set_state(map::State{(n.pose.w_from_imu * cal.rig_from_imu().inverse()).inverse(),
                                           n.pose.velocity, n.pose.acc_bias, n.pose.gyro_bias});
        }
      }
    }
    for (auto& row : sub.landmark_and_obs) {
      for (auto& v : row) {
        auto it = publish_points.find(v.landmark->id());
        if (it != publish_points.end()) {
          v.landmark->set_pose(it->second);
        }
      }
    }
  };
  const auto publish_start = Clock::now();
  if (d.pending_map_write) {
    publish();
  }
  // Local state and map are committed from the same validated candidate.
  d.map_anchors.clear();
  for (const auto& keyframe : map.get_recent_submap(20).consecutive_keyframes) {
    const auto time = keyframe.keyframe->time_ns();
    if (std::any_of(g.nodes.begin(), g.nodes.end(), [&](const Node& node) { return node.t == time; })) {
      d.map_anchors.emplace(time, keyframe.keyframe->get_state());
    }
  }
  TraceMessage("SCALE_MAP_PUBLISH t=%lld ms=%.6f", (long long)t, elapsed(publish_start));
  // Drop unreferenced temporary points to keep memory bounded.
  std::set<TrackId> used;
  for (auto& n : g.nodes) {
    for (auto& o : n.observations) {
      used.insert(o.id);
    }
  }
  for (auto it = g.points.begin(); it != g.points.end();) {
    if (!used.count(it->first)) {
      it = g.points.erase(it);
    } else {
      ++it;
    }
  }
  d.active = std::move(g);
  ++d.commits;
  TraceMessage("WINDOW_COMMIT t=%lld keyframes=%zu commits=%zu", (long long)t, d.active.nodes.size(), d.commits);
  return true;
}
}  // namespace cuvslam::sba_imu
