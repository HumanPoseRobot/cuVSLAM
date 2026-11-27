#include "pnp/joint_geometry.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include "common/log.h"
#include "pnp/cauchy_parameters.h"
#include "pnp/joint_sqrt_prior.h"
namespace cuvslam::pnp {
namespace {
using Vec6 = Eigen::Matrix<double, 6, 1>;
struct JointStep {
  Eigen::VectorXd pose;
  std::vector<Eigen::Vector3d> points;
  double predicted = 0.;
  bool valid = false;
};
using geometry_profile_math::Rows;
JointStep SolveJoint(const std::vector<Rows>& blocks, double lambda) {
  JointStep step;
  if (blocks.empty()) {
    return step;
  }
  const int dimension = blocks.front().pose.cols();
  step.pose = Eigen::VectorXd::Zero(dimension);
  Eigen::VectorXd scale = Eigen::VectorXd::Zero(dimension);
  for (const auto& b : blocks) {
    scale += b.pose.array().square().colwise().sum().matrix().transpose();
  }
  Eigen::VectorXd reference_squared = Eigen::VectorXd::Zero(dimension);
  std::vector<Eigen::MatrixXd> kept;
  std::vector<Eigen::VectorXd> residuals;
  int total = dimension;
  for (const auto& b : blocks) {
    Eigen::MatrixXd a = b.pose, point = b.point;
    Eigen::VectorXd r = b.residual;
    if (point.cols()) {
      const Eigen::Vector3d ps = point.array().square().colwise().sum().matrix().transpose();
      const int n = point.rows();
      point.conservativeResize(n + 3, 3);
      point.bottomRows(3).setZero();
      point.bottomRows(3).diagonal() = (lambda * ps.array().max(1e-12)).sqrt().matrix();
      a.conservativeResize(n + 3, dimension);
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
  Eigen::MatrixXd a = Eigen::MatrixXd::Zero(total, dimension);
  Eigen::VectorXd r = Eigen::VectorXd::Zero(total);
  int at = 0;
  for (size_t k = 0; k < kept.size(); ++k) {
    a.middleRows(at, kept[k].rows()) = kept[k];
    r.segment(at, residuals[k].size()) = residuals[k];
    at += kept[k].rows();
  }
  a.bottomRows(dimension).diagonal() = (lambda * scale.array().max(1e-12)).sqrt().matrix();
  if (dimension > 0) {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(a);
    if (qr.rank() != dimension) {
      return step;
    }
    step.pose = qr.solve(-r);
    if (!step.pose.allFinite()) {
      return step;
    }
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
        return JointStep{};
      }
    }
    const Eigen::VectorXd delta = b.pose * step.pose + (b.point.cols() ? Eigen::VectorXd(b.point * step.points[k])
                                                                       : Eigen::VectorXd::Zero(b.residual.size()));
    step.predicted -= b.residual.dot(delta) + .5 * delta.squaredNorm();
  }
  step.valid = std::isfinite(step.predicted) && step.predicted > 0.;
  return step;
}
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

struct Observation {
  size_t node;
  camera::Observation value;
};
struct Group {
  TrackId id;
  Eigen::Vector3d point;
  std::vector<Observation> observations;
  bool current = false, history = false;
};
}  // namespace
bool RefineJointGeometry(const map::UnifiedMap& map, const camera::Rig& rig,
                         const std::unordered_map<TrackId, Vector3T>& landmarks,
                         const std::vector<camera::Observation>& observations, size_t required, int64_t time_ns,
                         Isometry3T& output, Matrix6T* information, JointGeometryReport* report) {
  const auto started = std::chrono::steady_clock::now();
  const auto sub = map.get_recent_submap(map.capacity());
  const size_t history_count = sub.consecutive_keyframes.size();
  if (!history_count) {
    return false;
  }
  std::vector<Eigen::Isometry3d> poses;
  for (const auto& k : sub.consecutive_keyframes) {
    poses.push_back(k.keyframe->get_pose().cast<double>());
  }
  poses.push_back(output.cast<double>());
  const int dimension = 6 * history_count;  // First historical pose fixes only the visual world gauge.
  std::map<TrackId, Group> by_id;
  for (size_t node = 0; node < history_count; ++node) {
    for (const auto& entry : sub.landmark_and_obs[node]) {
      const auto point = entry.landmark->get_pose();
      if (!point) {
        continue;
      }
      auto [it, inserted] = by_id.try_emplace(entry.landmark->id());
      auto& group = it->second;
      if (inserted) {
        group.id = entry.landmark->id();
        group.point = point->cast<double>();
      }
      for (const auto& obs : entry.observations) {
        if (obs.cam_id >= rig.num_cameras) {
          continue;
        }
        const Eigen::Vector3d pc = rig.camera_from_rig[obs.cam_id].cast<double>() * poses[node] * group.point;
        if (!pc.allFinite() || pc.z() <= .05) {
          continue;
        }
        const Eigen::Vector2d r = pc.hnormalized() - obs.xy.cast<double>();
        Eigen::LLT<Eigen::Matrix2d> whitening(9. * obs.xy_info.cast<double>());
        if (whitening.info() != Eigen::Success || r.dot(9. * obs.xy_info.cast<double>() * r) > 9.21) {
          continue;
        }
        group.observations.push_back({node, obs});
        group.history = true;
      }
    }
  }
  for (const auto& obs : observations) {
    const auto point = landmarks.find(obs.id);
    if (point == landmarks.end() || obs.cam_id >= rig.num_cameras) {
      continue;
    }
    const Eigen::Vector3d pc =
        rig.camera_from_rig[obs.cam_id].cast<double>() * poses.back() * point->second.cast<double>();
    if (!pc.allFinite() || pc.z() <= .001) {
      continue;
    }
    auto [it, inserted] = by_id.try_emplace(obs.id);
    auto& group = it->second;
    if (inserted) {
      group.id = obs.id;
      group.point = point->second.cast<double>();
    }
    // One shared point variable; no fixed-map fallback for missing history.
    group.observations.push_back({history_count, obs});
    group.current = true;
  }
  std::vector<Group> groups;
  size_t missing = 0;
  for (auto& [id, group] : by_id) {
    if (group.current && !group.history) {
      ++missing;
    }
    if (!group.observations.empty()) {
      groups.push_back(std::move(group));
    }
  }

  if (const char* directory = std::getenv("VIO_JOINT_GEOMETRY_SNAPSHOT_DIR")) {
    const char* selected = std::getenv("VIO_JOINT_GEOMETRY_SNAPSHOT_NS");
    if (!selected || time_ns == std::strtoll(selected, nullptr, 10)) {
      std::ofstream file(std::string(directory) + "/joint_" + std::to_string(time_ns) + ".json");
      file << std::setprecision(17);
      const auto array = [&](const auto& matrix) {
        file << '[';
        for (int r = 0; r < matrix.rows(); ++r) {
          for (int c = 0; c < matrix.cols(); ++c) {
            if (r || c) {
              file << ',';
            }
            file << matrix(r, c);
          }
        }
        file << ']';
      };
      const auto obs = [&](const camera::Observation& observation) {
        file << "{\"id\":" << observation.id << ",\"cam\":" << int(observation.cam_id) << ",\"xy\":";
        array(observation.xy);
        file << ",\"info\":";
        array(observation.xy_info);
        file << '}';
      };
      file << "{\"time_ns\":" << time_ns << ",\"required\":" << required << ",\"times\":[";
      for (size_t k = 0; k < history_count; ++k) {
        if (k) {
          file << ',';
        }
        file << sub.consecutive_keyframes[k].keyframe->time_ns();
      }
      file << "],\"poses\":[";
      for (size_t k = 0; k < poses.size(); ++k) {
        if (k) {
          file << ',';
        }
        array(poses[k].matrix());
      }
      file << "],\"camera_from_rig\":[";
      for (int k = 0; k < rig.num_cameras; ++k) {
        if (k) {
          file << ',';
        }
        array(rig.camera_from_rig[k].matrix());
      }
      file << "],\"groups\":[";
      for (size_t k = 0; k < groups.size(); ++k) {
        const auto& group = groups[k];
        if (k) {
          file << ',';
        }
        file << "{\"id\":" << group.id << ",\"point\":";
        array(group.point);
        file << ",\"observations\":[";
        for (size_t j = 0; j < group.observations.size(); ++j) {
          if (j) {
            file << ',';
          }
          file << "{\"node\":" << group.observations[j].node << ",\"value\":";
          obs(group.observations[j].value);
          file << '}';
        }
        file << "]}";
      }
      file << "],\"landmarks\":[";
      bool first = true;
      for (const auto& [id, point] : landmarks) {
        if (!first) {
          file << ',';
        }
        first = false;
        file << "{\"id\":" << id << ",\"point\":";
        array(point);
        file << '}';
      }
      file << "],\"observations\":[";
      for (size_t k = 0; k < observations.size(); ++k) {
        if (k) {
          file << ',';
        }
        obs(observations[k]);
      }
      file << "]}";
    }
  }

  const double cauchy = VisualCauchyParameters(1.f, 0).c;
  auto evaluate = [&](const std::vector<Eigen::Isometry3d>& state, const std::vector<Group>& points,
                      std::vector<geometry_profile_math::Rows>* rows) {
    if (rows) {
      rows->clear();
    }
    double total = 0.;
    for (const auto& group : points) {
      geometry_profile_math::Rows block;
      const int n = 2 * group.observations.size();
      block.pose = Eigen::MatrixXd::Zero(n, dimension);
      block.point = Eigen::MatrixXd::Zero(n, 3);
      block.residual = Eigen::VectorXd::Zero(n);
      int row = 0;
      for (const auto& observation : group.observations) {
        const auto& obs = observation.value;
        const Eigen::Isometry3d cw = rig.camera_from_rig[obs.cam_id].cast<double>() * state[observation.node];
        const Eigen::Vector3d pc = cw * group.point;
        if (!pc.allFinite() || pc.z() <= .001) {
          return std::numeric_limits<double>::infinity();
        }
        Eigen::LLT<Eigen::Matrix2d> whitening(9. * obs.xy_info.cast<double>());
        if (whitening.info() != Eigen::Success) {
          return std::numeric_limits<double>::infinity();
        }
        const Eigen::Vector2d residual = whitening.matrixU() * (pc.hnormalized() - obs.xy.cast<double>());
        total += .5 / cauchy * std::log1p(cauchy * residual.squaredNorm());
        if (rows) {
          const double weight = 1. / std::sqrt(1. + cauchy * residual.squaredNorm());
          const Eigen::Matrix<double, 2, 3> J = weight * (whitening.matrixU() * Projection(pc)).eval() * cw.linear();
          block.point.middleRows(row, 2) = J;
          if (observation.node) {
            Eigen::Matrix<double, 3, 6> tangent;
            tangent.leftCols<3>() = -Skew(group.point);
            tangent.rightCols<3>().setIdentity();
            block.pose.block(row, 6 * (observation.node - 1), 2, 6) = J * tangent;
          }
          block.residual.segment<2>(row) = weight * residual;
        }
        row += 2;
      }
      if (rows) {
        rows->push_back(std::move(block));
      }
    }
    return total;
  };
  auto support = [&](const Eigen::Isometry3d& pose) {
    size_t count = 0;
    for (const auto& obs : observations) {
      const auto point = landmarks.find(obs.id);
      if (point == landmarks.end() || obs.cam_id >= rig.num_cameras) {
        continue;
      }
      const Eigen::Vector3d pc = rig.camera_from_rig[obs.cam_id].cast<double>() * pose * point->second.cast<double>();
      Vector2T a, b;
      if (!pc.allFinite() || pc.z() <= .001 ||
          !rig.intrinsics[obs.cam_id]->denormalizePoint(pc.hnormalized().cast<float>(), a) ||
          !rig.intrinsics[obs.cam_id]->denormalizePoint(obs.xy, b)) {
        continue;
      }
      const Eigen::Vector2d r = pc.hnormalized() - obs.xy.cast<double>();
      count += r.dot(9. * obs.xy_info.cast<double>() * r) <= 9.21 && (a - b).squaredNorm() <= 64.;
    }
    return count;
  };

  if (std::getenv("VIO_GEOMETRY_CHECK_JACOBIANS")) {
    std::vector<geometry_profile_math::Rows> linear;
    evaluate(poses, groups, &linear);
    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(dimension);
    for (const auto& block : linear) {
      gradient += block.pose.transpose() * block.residual;
    }
    constexpr double epsilon = 1e-6;
    double worst = 0.;
    const auto compare = [&](double actual, double reference) {
      worst = std::max(worst, std::abs(actual - reference) / std::max({1., std::abs(actual), std::abs(reference)}));
    };
    for (int col = 0; col < dimension; ++col) {
      auto plus = poses, minus = poses;
      Vec6 d = Vec6::Zero();
      d[col % 6] = epsilon;
      const size_t node = 1 + col / 6;
      plus[node] = Retract(poses[node], d);
      minus[node] = Retract(poses[node], -d);
      compare((evaluate(plus, groups, nullptr) - evaluate(minus, groups, nullptr)) / (2 * epsilon), gradient[col]);
    }
    for (size_t point = 0; point < groups.size(); ++point) {
      const Eigen::Vector3d exact = linear[point].point.transpose() * linear[point].residual;
      for (int col = 0; col < 3; ++col) {
        auto plus = groups, minus = groups;
        plus[point].point[col] += epsilon;
        minus[point].point[col] -= epsilon;
        compare((evaluate(poses, plus, nullptr) - evaluate(poses, minus, nullptr)) / (2 * epsilon), exact[col]);
      }
    }
    TraceMessage("JOINT_GEOMETRY_GRADIENT t=%lld max_relative=%.12g", (long long)time_ns, worst);
    if (!std::isfinite(worst) || worst > 1e-5) {
      return false;
    }
  }

  const auto initial = poses.back();
  const auto source_poses = poses;
  std::vector<Eigen::Vector3d> source_points;
  for (const auto& group : groups) {
    source_points.push_back(group.point);
  }
  const size_t before_support = support(initial);
  double current = evaluate(poses, groups, nullptr), before = current, lambda = 1e-3;
  if (!std::isfinite(current)) {
    return false;
  }
  std::vector<geometry_profile_math::Rows> blocks;
  for (int iter = 0; iter < 15 && std::isfinite(current) && current > 1e-7; ++iter) {
    evaluate(poses, groups, &blocks);
    const auto step = SolveJoint(blocks, lambda);
    if (!step.valid) {
      lambda = std::min(1e6, lambda * 10.);
      continue;
    }
    auto trial = poses;
    auto trial_points = groups;
    for (size_t node = 1; node < trial.size(); ++node) {
      trial[node] = Retract(poses[node], step.pose.segment<6>(6 * (node - 1)));
    }
    for (size_t k = 0; k < groups.size(); ++k) {
      trial_points[k].point += step.points[k];
    }
    const double updated = evaluate(trial, trial_points, nullptr);
    const double gain = (current - updated) / step.predicted;
    if (std::isfinite(updated) && updated <= current && gain >= .25) {
      poses = std::move(trial);
      groups = std::move(trial_points);
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

  bool covariance_only = false;
  const size_t proposal_support = support(poses.back());
  if (proposal_support < std::max(required, before_support) && !std::getenv("VIO_GEOMETRY_DISABLE_COV_ONLY")) {
    // Reject only the proposed mean. Keep the independently admitted frontend
    // pose and estimate local joint information there, with latent geometry
    // reprofiled. This information is not inserted as a zero-gradient factor:
    // the initialized full backend owns the raw image factors and their RHS.
    covariance_only = true;
    poses = source_poses;
    for (size_t k = 0; k < groups.size(); ++k) {
      groups[k].point = source_points[k];
    }
    current = evaluate(poses, groups, nullptr);
    lambda = 1e-3;
    for (int iter = 0; iter < 15 && std::isfinite(current) && current > 1e-7; ++iter) {
      evaluate(poses, groups, &blocks);
      for (auto& block : blocks) {
        block.pose = block.pose.leftCols(dimension - 6).eval();
      }
      const auto step = SolveJoint(blocks, lambda);
      if (!step.valid) {
        lambda = std::min(1e6, lambda * 10.);
        continue;
      }
      auto trial = poses;
      auto trial_points = groups;
      for (size_t node = 1; node < history_count; ++node) {
        trial[node] = Retract(poses[node], step.pose.segment<6>(6 * (node - 1)));
      }
      for (size_t k = 0; k < groups.size(); ++k) {
        trial_points[k].point += step.points[k];
      }
      const double updated = evaluate(trial, trial_points, nullptr);
      const double gain = (current - updated) / step.predicted;
      if (std::isfinite(updated) && updated <= current && gain >= .25) {
        poses = std::move(trial);
        groups = std::move(trial_points);
        current = updated;
        if (gain > .75) {
          lambda = std::max(1e-6, lambda * .3);
        }
        double norm = step.pose.squaredNorm();
        for (const auto& point : step.points) {
          norm += point.squaredNorm();
        }
        if (norm < 1e-10 || step.predicted / std::max(1., current) < 1e-6) {
          break;
        }
      } else {
        lambda = std::min(1e6, lambda * 10.);
      }
    }
    TraceMessage("JOINT_GEOMETRY_COV_ONLY t=%lld rejected_support=%zu->%zu mean_unchanged=1 fixed_map_fallback=0",
                 (long long)time_ns, before_support, proposal_support);
  }

  evaluate(poses, groups, &blocks);
  joint_geometry::Factor reduced;
  Eigen::VectorXd reference_squared = Eigen::VectorXd::Zero(dimension);
  std::vector<Eigen::MatrixXd> kept;
  std::vector<Eigen::VectorXd> residuals;
  int count = 0;
  for (const auto& block : blocks) {
    reference_squared += block.pose.array().square().colwise().sum().matrix().transpose();
    Eigen::MatrixXd a;
    Eigen::VectorXd r;
    gpu_window::project_out(block.point, block.pose, block.residual, a, r);
    count += a.rows();
    kept.push_back(std::move(a));
    residuals.push_back(std::move(r));
  }
  reduced.rows.resize(count, dimension);
  reduced.residual.resize(count);
  int at = 0;
  for (size_t k = 0; k < kept.size(); ++k) {
    reduced.rows.middleRows(at, kept[k].rows()) = kept[k];
    reduced.residual.segment(at, residuals[k].size()) = residuals[k];
    at += kept[k].rows();
  }
  std::vector<int> current_columns;
  for (int k = dimension - 6; k < dimension; ++k) {
    current_columns.push_back(k);
  }
  const Eigen::VectorXd reference_scale = reference_squared.cwiseSqrt();
  const Eigen::VectorXd current_scale = reference_scale.tail(6);
  const auto posterior = joint_geometry::marginal(reduced, current_columns, &current_scale);
  const size_t after_support = support(poses.back());

  Eigen::MatrixXd support_jacobian = Eigen::MatrixXd::Zero(2 * observations.size(), 6);
  int support_rows = 0;
  for (const auto& obs : observations) {
    const auto point = landmarks.find(obs.id);
    if (point == landmarks.end() || obs.cam_id >= rig.num_cameras) {
      continue;
    }
    const Eigen::Vector3d base = point->second.cast<double>();
    const auto ext = rig.camera_from_rig[obs.cam_id].cast<double>();
    const Eigen::Vector3d pc = ext * poses.back() * base;
    Vector2T a, b;
    if (!pc.allFinite() || pc.z() <= .001 ||
        !rig.intrinsics[obs.cam_id]->denormalizePoint(pc.hnormalized().cast<float>(), a) ||
        !rig.intrinsics[obs.cam_id]->denormalizePoint(obs.xy, b)) {
      continue;
    }
    const Eigen::Vector2d r = pc.hnormalized() - obs.xy.cast<double>();
    if (r.dot(9. * obs.xy_info.cast<double>() * r) > 9.21 || (a - b).squaredNorm() > 64.) {
      continue;
    }
    Eigen::LLT<Eigen::Matrix2d> info(9. * obs.xy_info.cast<double>());
    Eigen::Matrix<double, 3, 6> tangent;
    tangent.leftCols<3>() = -Skew(base);
    tangent.rightCols<3>().setIdentity();
    support_jacobian.middleRows(support_rows, 2) =
        info.matrixU() * Projection(pc) * (ext.linear() * poses.back().linear()) * tangent;
    support_rows += 2;
  }
  support_jacobian.conservativeResize(support_rows, 6);
  const bool original_rank = HasObservableVisualSupport(support_jacobian, after_support, required);

  const bool accepted = posterior.observable && original_rank && std::isfinite(current) && current <= before &&
                        after_support >= std::max(required, before_support) && poses.back().matrix().allFinite();
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  TraceMessage(
      "JOINT_GEOMETRY t=%lld accept=%d historical=%zu points=%zu missing=%zu gauge=%lld before=%.12g after=%.12g "
      "support=%zu->%zu dp=%.9g ms=%.6f",
      (long long)time_ns, int(accepted), history_count, groups.size(), missing,
      (long long)sub.consecutive_keyframes.front().keyframe->time_ns(), before, current, before_support, after_support,
      (poses.back().inverse().translation() - initial.inverse().translation()).norm(), ms);
  if (report) {
    report->covariance_only = covariance_only;
    if (posterior.gradient.size() == 6) {
      report->profile_gradient = posterior.gradient;
    }
    report->historical_poses = history_count;
    report->points = groups.size();
    report->missing_history = missing;
    std::vector<int> all_columns;
    for (int col = 0; col < dimension; ++col) {
      all_columns.push_back(col);
    }
    const auto navigation = joint_geometry::marginal(reduced, all_columns, &reference_scale);
    if (navigation.observable) {
      report->navigation_covariance = navigation.covariance;
      report->point_pose_covariance =
          Eigen::MatrixXd::Constant(3 * blocks.size(), dimension, std::numeric_limits<double>::quiet_NaN());
      for (size_t k = 0; k < blocks.size(); ++k) {
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(blocks[k].point);
        if (qr.rank() == 3) {
          report->point_pose_covariance.middleRows(3 * k, 3) = -qr.solve(blocks[k].pose) * navigation.covariance;
        }
      }
    }
  }
  if (accepted) {
    if (information && !posterior.information.cast<float>().allFinite()) {
      return false;
    }
    output = poses.back().cast<float>();
    if (information) {
      *information = posterior.information.cast<float>();
    }
  }
  return accepted;
}
}  // namespace cuvslam::pnp
