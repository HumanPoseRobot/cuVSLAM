#include "pnp/geometry_profile.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include "common/log.h"
#include "pipelines/gpu_window/window_linear.h"
#include "pnp/cauchy_parameters.h"
#include "pnp/joint_geometry.h"

namespace cuvslam::pnp {
namespace geometry_profile_math {
Step Solve(const std::vector<Rows>& blocks, double lambda) {
  Step step;
  Eigen::Matrix<double, 6, 1> scale = Eigen::Matrix<double, 6, 1>::Zero();
  for (const auto& b : blocks) {
    scale += b.pose.array().square().colwise().sum().matrix().transpose();
  }
  std::vector<Eigen::MatrixXd> kept;
  std::vector<Eigen::VectorXd> residuals;
  int total = 6;
  for (const auto& b : blocks) {
    Eigen::MatrixXd a = b.pose, point = b.point;
    Eigen::VectorXd r = b.residual;
    if (point.cols()) {
      const Eigen::Vector3d ps = point.array().square().colwise().sum().matrix().transpose();
      const int n = point.rows();
      point.conservativeResize(n + 3, 3);
      point.bottomRows(3).setZero();
      point.bottomRows(3).diagonal() = (lambda * ps.array().max(1e-12)).sqrt().matrix();
      a.conservativeResize(n + 3, 6);
      a.bottomRows(3).setZero();
      r.conservativeResize(n + 3);
      r.tail(3).setZero();
      Eigen::MatrixXd projected;
      Eigen::VectorXd rr;
      gpu_window::project_out(point, a, r, projected, rr);
      kept.push_back(std::move(projected));
      residuals.push_back(std::move(rr));
    } else {
      kept.push_back(a);
      residuals.push_back(r);
    }
    total += kept.back().rows();
  }
  Eigen::MatrixXd a = Eigen::MatrixXd::Zero(total, 6);
  Eigen::VectorXd r = Eigen::VectorXd::Zero(total);
  int at = 0;
  for (size_t k = 0; k < kept.size(); ++k) {
    a.middleRows(at, kept[k].rows()) = kept[k];
    r.segment(at, residuals[k].size()) = residuals[k];
    at += kept[k].rows();
  }
  a.bottomRows(6).diagonal() = (lambda * scale.array().max(1e-12)).sqrt().matrix();
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a);
  if (qr.rank() != 6) {
    return step;
  }
  step.pose = qr.solve(-r);
  if (!step.pose.allFinite()) {
    return step;
  }
  step.points.resize(blocks.size(), Eigen::Vector3d::Zero());
  for (size_t k = 0; k < blocks.size(); ++k) {
    const auto& b = blocks[k];
    if (b.point.cols()) {
      Eigen::MatrixXd point = b.point;
      Eigen::VectorXd rhs = -(b.residual + b.pose * step.pose);
      const Eigen::Vector3d ps = point.array().square().colwise().sum().matrix().transpose();
      const int n = point.rows();
      point.conservativeResize(n + 3, 3);
      point.bottomRows(3).setZero();
      point.bottomRows(3).diagonal() = (lambda * ps.array().max(1e-12)).sqrt().matrix();
      rhs.conservativeResize(n + 3);
      rhs.tail(3).setZero();
      step.points[k] = point.colPivHouseholderQr().solve(rhs);
      if (!step.points[k].allFinite()) {
        return Step{};
      }
    }
    const Eigen::VectorXd delta = b.pose * step.pose + (b.point.cols() ? Eigen::VectorXd(b.point * step.points[k])
                                                                       : Eigen::VectorXd::Zero(b.residual.size()));
    step.predicted -= b.residual.dot(delta) + .5 * delta.squaredNorm();
  }
  step.valid = std::isfinite(step.predicted) && step.predicted > 0.;
  return step;
}
}  // namespace geometry_profile_math
namespace {
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;
Eigen::Matrix3d Skew(const Eigen::Vector3d& p) {
  Eigen::Matrix3d s;
  s << 0, -p.z(), p.y(), p.z(), 0, -p.x(), -p.y(), p.x(), 0;
  return s;
}
Eigen::Isometry3d Retract(const Eigen::Isometry3d& pose, const Vec6& x) {
  const Eigen::Vector3d w = x.head<3>();
  const double theta = w.norm();
  const Eigen::Matrix3d W = Skew(w), W2 = W * W;
  const double a = theta < 1e-7 ? 1. - theta * theta / 6. : std::sin(theta) / theta;
  const double b = theta < 1e-7 ? .5 - theta * theta / 24. : (1. - std::cos(theta)) / (theta * theta);
  const double c = theta < 1e-7 ? 1. / 6. - theta * theta / 120. : (theta - std::sin(theta)) / (theta * theta * theta);
  Eigen::Isometry3d d = Eigen::Isometry3d::Identity();
  d.linear() = Eigen::Matrix3d::Identity() + a * W + b * W2;
  d.translation() = (Eigen::Matrix3d::Identity() + b * W + c * W2) * x.tail<3>();
  return pose * d;
}
Eigen::Matrix<double, 2, 3> Projection(const Eigen::Vector3d& p) {
  Eigen::Matrix<double, 2, 3> j;
  j << 1. / p.z(), 0., -p.x() / (p.z() * p.z()), 0., 1. / p.z(), -p.y() / (p.z() * p.z());
  return j;
}
struct Group {
  TrackId id;
  Eigen::Vector3d base, q = Eigen::Vector3d::Zero();
  Eigen::Matrix3d prior = Eigen::Matrix3d::Zero();
  Eigen::Vector3d offset = Eigen::Vector3d::Zero(), prior_origin = Eigen::Vector3d::Zero();
  bool uncertain = false, ray = false;
  gpu_window::SourceRay chart;
  std::vector<camera::Observation> observations;
};
Eigen::Vector3d Point(const Group& g, const Eigen::Vector3d& q) {
  if (!g.uncertain) {
    return g.base;
  }
  double delta[3];
  gpu_window::source_point_delta(g.ray ? &g.chart : nullptr, q.data(), delta);
  return g.base + Eigen::Map<const Eigen::Vector3d>(delta);
}
Eigen::Matrix3d PointJacobian(const Group& g) {
  if (!g.ray) {
    return Eigen::Matrix3d::Identity();
  }
  Eigen::Matrix3d out;
  for (int c = 0; c < 3; ++c) {
    gpu_vi::Jet q[3], delta[3];
    for (int k = 0; k < 3; ++k) {
      q[k] = {g.q[k], k == c ? 1. : 0.};
    }
    gpu_window::source_point_delta(&g.chart, q, delta);
    for (int k = 0; k < 3; ++k) {
      out(k, c) = delta[k].d;
    }
  }
  return out;
}
size_t Support(const camera::Rig& rig, const std::vector<Group>& groups, const Eigen::Isometry3d& pose,
               Eigen::MatrixXd* jacobian = nullptr) {
  size_t total = 0;
  for (const auto& g : groups) {
    total += g.observations.size();
  }
  if (jacobian) {
    jacobian->resize(2 * total, 6);
  }
  size_t count = 0;
  for (const auto& g : groups) {
    for (const auto& o : g.observations) {
      const Eigen::Vector3d pc = rig.camera_from_rig[o.cam_id].cast<double>() * pose * g.base;
      Vector2T a, b;
      if (pc.z() <= .001 || !pc.allFinite() ||
          !rig.intrinsics[o.cam_id]->denormalizePoint(pc.hnormalized().cast<float>(), a) ||
          !rig.intrinsics[o.cam_id]->denormalizePoint(o.xy, b)) {
        continue;
      }
      const Eigen::Vector2d r = pc.hnormalized() - o.xy.cast<double>();
      if (r.dot(9. * o.xy_info.cast<double>() * r) <= 9.21 && (a - b).squaredNorm() <= 64.) {
        if (jacobian) {
          Eigen::Matrix<double, 3, 6> tangent;
          tangent.leftCols<3>() = -Skew(g.base);
          tangent.rightCols<3>().setIdentity();
          Eigen::LLT<Eigen::Matrix2d> info(9. * o.xy_info.cast<double>());
          jacobian->middleRows(2 * count, 2) = info.matrixU() * Projection(pc) *
                                               (rig.camera_from_rig[o.cam_id].linear().cast<double>() * pose.linear()) *
                                               tangent;
        }
        ++count;
      }
    }
  }
  if (jacobian) {
    jacobian->conservativeResize(2 * count, 6);
  }
  return count;
}
}  // namespace
void GeometryProfile::reset() {
  map_time_ = -1;
  priors_.clear();
}
void GeometryProfile::update(const map::UnifiedMap& map, const camera::Rig& rig) {
  auto sub = map.get_recent_submap(map.capacity());
  if (sub.consecutive_keyframes.empty()) {
    reset();
    return;
  }
  const auto stamp = sub.consecutive_keyframes.back().keyframe->time_ns();
  // A keyframe timestamp is not a map revision: existing poses, landmarks,
  // observations or rig calibration can change without appending a keyframe.
  // Rebuild from the current call rather than reusing a stale conditional prior.
  // The caller must serialize map writes during this read, as for the existing
  // PnP/map snapshot path. Per-object locks do not provide a graph-wide epoch.
  const auto previous_stamp = map_time_;
  auto previous_priors = std::move(priors_);
  map_time_ = stamp;
  priors_.clear();
  struct History {
    Eigen::Vector3d point;
    std::vector<Eigen::Vector3d> rows;
    std::vector<double> residual;
    gpu_window::SourceRay chart;
  };
  std::map<TrackId, History> history;
  for (size_t k = 0; k < sub.consecutive_keyframes.size(); ++k) {
    const auto pose = sub.consecutive_keyframes[k].keyframe->get_state().rig_from_w.cast<double>();
    for (const auto& entry : sub.landmark_and_obs[k]) {
      const auto point = entry.landmark->get_pose();
      if (!point) {
        continue;
      }
      auto [hit, inserted] = history.try_emplace(entry.landmark->id());
      auto& h = hit->second;
      if (inserted) {
        h.point = point->cast<double>();
      }
      for (const auto& o : entry.observations) {
        if (o.cam_id >= rig.num_cameras) {
          continue;
        }
        const auto cw = rig.camera_from_rig[o.cam_id].cast<double>() * pose;
        const Eigen::Vector3d pc = cw * h.point;
        if (!pc.allFinite() || pc.z() <= .05) {
          continue;
        }
        const Eigen::Vector2d r = pc.hnormalized() - o.xy.cast<double>();
        const Eigen::Matrix2d info = 9. * o.xy_info.cast<double>();
        Eigen::LLT<Eigen::Matrix2d> ll(info);
        if (ll.info() != Eigen::Success || r.dot(info * r) > 9.21) {
          continue;
        }
        const Eigen::Matrix<double, 2, 3> j = ll.matrixU() * Projection(pc) * cw.linear();
        const Eigen::Vector2d wr = ll.matrixU() * r;
        for (int row = 0; row < 2; ++row) {
          h.rows.push_back(j.row(row).transpose());
          h.residual.push_back(wr[row]);
        }
        if (h.chart.inverse_depth == 0.) {
          const Eigen::Matrix3d Rwc = cw.linear().inverse();
          Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(h.chart.Rwc) = Rwc;
          Eigen::Map<Eigen::Vector3d>(h.chart.center) = -Rwc * cw.translation();
          h.chart.inverse_depth = 1. / pc.z();
          h.chart.ray[0] = pc.x() / pc.z();
          h.chart.ray[1] = pc.y() / pc.z();
          h.chart.source_xy[0] = o.xy.x();
          h.chart.source_xy[1] = o.xy.y();
          h.chart.node = int(k);
        }
      }
    }
  }
  for (auto& [id, h] : history) {
    if (h.rows.size() < 3) {
      continue;
    }
    Eigen::MatrixXd a(h.rows.size(), 3);
    Eigen::VectorXd r(h.rows.size());
    for (size_t k = 0; k < h.rows.size(); ++k) {
      a.row(k) = h.rows[k];
      r[k] = h.residual[k];
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(a);
    Prior p;
    p.origin = h.point;
    p.rows = qr.matrixQR().topRows(3).triangularView<Eigen::Upper>();
    p.residual = (qr.householderQ().adjoint() * r).head(3);
    p.chart = h.chart;
    priors_.emplace(id, p);
  }
  if (std::getenv("VIO_GEOMETRY_MEAN_AUDIT")) {
    size_t changed = 0;
    for (const auto& [id, prior] : priors_) {
      const auto old = previous_priors.find(id);
      if (old == previous_priors.end() || (prior.origin.array() != old->second.origin.array()).any() ||
          (prior.rows.array() != old->second.rows.array()).any() ||
          (prior.residual.array() != old->second.residual.array()).any()) {
        ++changed;
      }
    }
    for (const auto& [id, prior] : previous_priors) {
      if (priors_.find(id) == priors_.end()) {
        ++changed;
      }
    }
    TraceMessage("GEOMETRY_PRIOR_REBUILD stamp=%lld same_stamp=%d previous=%zu current=%zu changed=%zu",
                 (long long)stamp, int(previous_stamp == stamp), previous_priors.size(), priors_.size(), changed);
  }
}
bool GeometryProfile::refine(const map::UnifiedMap& map, const camera::Rig& rig,
                             const std::unordered_map<TrackId, Vector3T>& landmarks,
                             const std::vector<camera::Observation>& observations, size_t required, bool source_ray,
                             int64_t time_ns, Isometry3T& output, Matrix6T* information) {
  if (std::getenv("VIO_GEOMETRY_JOINT")) {
    return RefineJointGeometry(map, rig, landmarks, observations, required, time_ns, output, information);
  }
  const char* mode = std::getenv("VIO_GEOMETRY_USE_CENTERED");
  const char* selected = std::getenv("VIO_GEOMETRY_CENTERED_FRAME_NS");
  if (mode && (!selected || time_ns == std::strtoll(selected, nullptr, 10))) {
    return refineWithPriorMean(map, rig, landmarks, observations, required, source_ray, time_ns, output, information,
                               PriorMean::PreserveMap);
  }
  if (std::getenv("VIO_GEOMETRY_MEAN_AUDIT")) {
    auto centered = output;
    Matrix6T centered_information = Matrix6T::Zero();
    const bool centered_ok = refineWithPriorMean(map, rig, landmarks, observations, required, source_ray, time_ns,
                                                 centered, &centered_information, PriorMean::PreserveMap);
    const auto input = output;
    const bool accepted = refineWithPriorMean(map, rig, landmarks, observations, required, source_ray, time_ns, output,
                                              information, PriorMean::Conditional);
    const Eigen::Vector3d conditional_delta =
        output.inverse().translation().cast<double>() - input.inverse().translation().cast<double>();
    const Eigen::Vector3d centered_delta =
        centered.inverse().translation().cast<double>() - input.inverse().translation().cast<double>();
    TraceMessage(
        "GEOMETRY_MEAN_PAIR t=%lld conditional_accept=%d centered_accept=%d "
        "conditional_dxyz=%.12g,%.12g,%.12g centered_dxyz=%.12g,%.12g,%.12g",
        (long long)time_ns, int(accepted), int(centered_ok), conditional_delta.x(), conditional_delta.y(),
        conditional_delta.z(), centered_delta.x(), centered_delta.y(), centered_delta.z());
    if (const char* directory = std::getenv("VIO_GEOMETRY_MEAN_SNAPSHOT_DIR")) {
      std::ofstream file(std::string(directory) + "/mean_" + std::to_string(time_ns) + ".json");
      file << std::setprecision(17) << "{\"time_ns\":" << time_ns << ",\"conditional_accept\":" << int(accepted)
           << ",\"centered_accept\":" << int(centered_ok);
      const auto write_pose = [&](const char* name, const Isometry3T& pose) {
        file << ",\"" << name << "\":[";
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            if (r || c) {
              file << ',';
            }
            file << pose.matrix()(r, c);
          }
        }
        file << ']';
      };
      write_pose("input_rig_from_world", input);
      write_pose("conditional_rig_from_world", output);
      write_pose("centered_rig_from_world", centered);
      file << '}';
    }
    return accepted;
  }
  return refineWithPriorMean(map, rig, landmarks, observations, required, source_ray, time_ns, output, information,
                             PriorMean::Conditional);
}
bool GeometryProfile::refineWithPriorMean(const map::UnifiedMap& map, const camera::Rig& rig,
                                          const std::unordered_map<TrackId, Vector3T>& landmarks,
                                          const std::vector<camera::Observation>& observations, size_t required,
                                          bool source_ray, int64_t time_ns, Isometry3T& output, Matrix6T* information,
                                          PriorMean mean) {
  const auto started = std::chrono::steady_clock::now();
  update(map, rig);
  std::map<TrackId, Group> by_id;
  for (const auto& o : observations) {
    auto it = landmarks.find(o.id);
    if (it == landmarks.end() || o.cam_id >= rig.num_cameras) {
      continue;
    }
    const Eigen::Vector3d visible =
        rig.camera_from_rig[o.cam_id].cast<double>() * output.cast<double>() * it->second.cast<double>();
    if (!visible.allFinite() || visible.z() <= .001) {
      continue;
    }
    auto [entry, inserted] = by_id.try_emplace(o.id);
    auto& g = entry->second;
    if (inserted) {
      g.id = o.id;
      g.base = it->second.cast<double>();
      const auto prior = priors_.find(o.id);
      if (prior != priors_.end()) {
        g.uncertain = true;
        g.prior = prior->second.rows;
        g.offset = prior->second.residual;
        g.prior_origin = prior->second.origin;
        if (mean == PriorMean::PreserveMap) {
          // Exact recentering: R * (X - origin) + offset == R * (X - base).
          // Keep the same rows/information; alter only the prior's center.
          g.offset = -g.prior * (g.base - g.prior_origin);
        }
        g.chart = prior->second.chart;
        g.ray = source_ray;
        // The chart must start at this call's exact map point.
        const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Rwc(g.chart.Rwc);
        const Eigen::Vector3d pc = Rwc.inverse() * (g.base - Eigen::Map<const Eigen::Vector3d>(g.chart.center));
        if (pc.z() > .05 && pc.allFinite()) {
          g.chart.ray[0] = pc.x() / pc.z();
          g.chart.ray[1] = pc.y() / pc.z();
          g.chart.inverse_depth = 1. / pc.z();
        } else {
          g.ray = false;
        }
      }
    }
    g.observations.push_back(o);
  }
  std::vector<Group> groups;
  size_t uncertain = 0;
  for (auto& [id, g] : by_id) {
    std::sort(g.observations.begin(), g.observations.end(),
              [](const auto& a, const auto& b) { return a.cam_id < b.cam_id; });
    uncertain += g.uncertain;
    groups.push_back(std::move(g));
  }
  if (!uncertain) {
    return false;
  }
  Eigen::Isometry3d pose = output.cast<double>();
  const auto initial = pose;
  const size_t before_support = Support(rig, groups, pose);
  const double cauchy = VisualCauchyParameters(1.f, 0).c;
  const double amplitude = 1. / cauchy;
  auto evaluate = [&](const Eigen::Isometry3d& T, const std::vector<Group>& gs,
                      std::vector<geometry_profile_math::Rows>* blocks) {
    double total = 0.;
    if (blocks) {
      blocks->clear();
    }
    for (const auto& g : gs) {
      const Eigen::Vector3d point = Point(g, g.q);
      if (!point.allFinite()) {
        return std::numeric_limits<double>::infinity();
      }
      const auto L = PointJacobian(g);
      const int count = 2 * g.observations.size() + (g.uncertain ? 3 : 0);
      geometry_profile_math::Rows b;
      b.pose = Eigen::MatrixXd::Zero(count, 6);
      b.point = Eigen::MatrixXd::Zero(count, g.uncertain ? 3 : 0);
      b.residual = Eigen::VectorXd::Zero(count);
      int row = 0;
      for (const auto& o : g.observations) {
        const auto ext = rig.camera_from_rig[o.cam_id].cast<double>();
        const auto cw = ext * T;
        const Eigen::Vector3d pc = cw * point;
        if (!pc.allFinite() || pc.z() <= .001) {
          return std::numeric_limits<double>::infinity();
        }
        const Eigen::Vector2d r = pc.hnormalized() - o.xy.cast<double>();
        Eigen::LLT<Eigen::Matrix2d> ll(9. * o.xy_info.cast<double>());
        if (ll.info() != Eigen::Success) {
          return std::numeric_limits<double>::infinity();
        }
        const Eigen::Vector2d wr = ll.matrixU() * r;
        const double squared = wr.squaredNorm();
        total += .5 * amplitude * std::log1p(cauchy * squared);
        const double weight = std::sqrt(amplitude * cauchy / (1. + cauchy * squared));
        if (blocks) {
          Eigen::Matrix<double, 3, 6> tangent;
          tangent.leftCols<3>() = -Skew(point);
          tangent.rightCols<3>().setIdentity();
          const Eigen::Matrix<double, 2, 3> projection = weight * (ll.matrixU() * Projection(pc)).eval();
          b.pose.middleRows(row, 2) = projection * cw.linear() * tangent;
          if (g.uncertain) {
            b.point.middleRows(row, 2) = projection * cw.linear() * L;
          }
          b.residual.segment<2>(row) = weight * wr;
        }
        row += 2;
      }
      if (g.uncertain) {
        const Eigen::Vector3d r = g.prior * (point - g.prior_origin) + g.offset;
        total += .5 * r.squaredNorm();
        if (blocks) {
          b.point.bottomRows(3) = g.prior * L;
          b.residual.tail(3) = r;
        }
      }
      if (blocks) {
        blocks->push_back(std::move(b));
      }
    }
    return total;
  };
  if (std::getenv("VIO_GEOMETRY_CHECK_JACOBIANS")) {
    // Diagnostic only: differentiate the actual objective and actual retraction,
    // holding observations, map points and whitening fixed.
    std::vector<geometry_profile_math::Rows> rows;
    evaluate(pose, groups, &rows);
    Vec6 analytic = Vec6::Zero();
    for (const auto& row : rows) {
      analytic += row.pose.transpose() * row.residual;
    }
    double worst = 0.;
    constexpr double epsilon = 1e-6;
    const auto compare = [&](double numeric, double exact) {
      worst = std::max(worst, std::abs(numeric - exact) / std::max({1., std::abs(numeric), std::abs(exact)}));
    };
    for (int c = 0; c < 6; ++c) {
      Vec6 direction = Vec6::Zero();
      direction[c] = epsilon;
      const double numeric =
          (evaluate(Retract(pose, direction), groups, nullptr) - evaluate(Retract(pose, -direction), groups, nullptr)) /
          (2. * epsilon);
      compare(numeric, analytic[c]);
    }
    for (size_t k = 0; k < groups.size(); ++k) {
      if (groups[k].uncertain) {
        const Eigen::Vector3d exact = rows[k].point.transpose() * rows[k].residual;
        for (int c = 0; c < 3; ++c) {
          auto plus = groups, minus = groups;
          plus[k].q[c] += epsilon;
          minus[k].q[c] -= epsilon;
          compare((evaluate(pose, plus, nullptr) - evaluate(pose, minus, nullptr)) / (2. * epsilon), exact[c]);
        }
      }
    }
    TraceMessage("GEOMETRY_GRADIENT_AUDIT ray=%d variables=%zu max_relative=%.12g", int(source_ray), 6 + 3 * uncertain,
                 worst);
    if (!std::isfinite(worst) || worst > 1e-5) {
      return false;
    }
  }
  double current = evaluate(pose, groups, nullptr), before = current, lambda = 1e-3;
  int iterations = 0;
  std::vector<geometry_profile_math::Rows> blocks;
  for (int iter = 0; iter < 15 && std::isfinite(current) && current > 1e-7; ++iter) {
    ++iterations;
    evaluate(pose, groups, &blocks);
    const auto step = geometry_profile_math::Solve(blocks, lambda);
    if (!step.valid) {
      lambda = std::min(1e6, lambda * 10.);
      continue;
    }
    auto trial_groups = groups;
    for (size_t k = 0; k < groups.size(); ++k) {
      if (groups[k].uncertain) {
        trial_groups[k].q += step.points[k];
      }
    }
    const auto trial_pose = Retract(pose, step.pose);
    const double updated = evaluate(trial_pose, trial_groups, nullptr);
    const double gain = (current - updated) / step.predicted;
    if (std::isfinite(updated) && updated <= current && gain >= .25) {
      pose = trial_pose;
      groups = std::move(trial_groups);
      current = updated;
      if (gain > .75) {
        lambda = std::max(1e-6, lambda * .3);
      }
      double norm = step.pose.squaredNorm();
      for (const auto& p : step.points) {
        norm += p.squaredNorm();
      }
      if (norm < 1e-10 || step.predicted / std::max(1., current) < 1e-6) {
        break;
      }
    } else {
      lambda = std::min(1e6, lambda * 10.);
    }
  }
  Eigen::MatrixXd original_jacobian;
  const size_t after_support = Support(rig, groups, pose, &original_jacobian);
  const bool original_rank = HasObservableVisualSupport(original_jacobian, after_support, required);
  evaluate(pose, groups, &blocks);
  int nrows = 0;
  std::vector<Eigen::MatrixXd> reduced;
  for (const auto& b : blocks) {
    Eigen::MatrixXd j;
    Eigen::VectorXd r;
    if (b.point.cols()) {
      gpu_window::project_out(b.point, b.pose, b.residual, j, r);
    } else {
      j = b.pose;
    }
    nrows += j.rows();
    reduced.push_back(j);
  }
  Eigen::MatrixXd observable(nrows, 6);
  int at = 0;
  for (const auto& j : reduced) {
    observable.middleRows(at, j.rows()) = j;
    at += j.rows();
  }
  const Eigen::Matrix<double, 6, 6> posterior = observable.transpose() * observable;
  bool rank = true;
  for (int col = 0; col < 6; ++col) {
    const double norm = observable.col(col).norm();
    if (!(norm > 0)) {
      rank = false;
      break;
    }
    observable.col(col) /= norm;
  }
  if (rank) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(observable);
    qr.setThreshold(64. * std::numeric_limits<double>::epsilon());
    rank = qr.rank() == 6;
  }
  const bool accepted = rank && original_rank && std::isfinite(current) && current < before &&
                        after_support >= std::max(required, before_support) && pose.matrix().allFinite();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  TraceMessage(
      "GEOMETRY_PROFILE t=%lld accept=%d ray=%d points=%zu uncertain=%zu before=%.12g after=%.12g support=%zu->%zu "
      "rank=%d dp=%.9g iterations=%d ms=%.6f prior_mean=%s",
      (long long)time_ns, int(accepted), int(source_ray), groups.size(), uncertain, before, current, before_support,
      after_support, int(rank), (pose.inverse().translation() - initial.inverse().translation()).norm(), iterations, ms,
      mean == PriorMean::Conditional ? "conditional" : "map");
  if (accepted) {
    if (information && !posterior.cast<float>().allFinite()) {
      return false;
    }
    output = pose.cast<float>();
    if (information) {
      *information = posterior.cast<float>();
    }
  }
  return accepted;
}
}  // namespace cuvslam::pnp
