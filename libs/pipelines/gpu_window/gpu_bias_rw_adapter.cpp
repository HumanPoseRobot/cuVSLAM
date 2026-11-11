#include "gpu_bias_rw_adapter.h"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include "common/log.h"
#include "gpu_bias_rw_math.h"
namespace gpu_bias_rw {
Result OptimizeEliminated(const Data& data, const std::vector<Observation>& obs, float huber, int iterations,
                          bool reuse) {
  // Only the RW factor contains the current bias. Its conditional minimizer is
  // exactly b_right=b_left, regardless of RW precision. Eliminating this block
  // avoids putting nearly identical, strongly coupled bias columns in FP32 LM.
  // This is an exact reduction of this TWO-frame graph, not a frozen-bias model;
  // the multi-keyframe window retains its separate bias variables and RW edges.
  const auto reduced = gpu_vi::OptimizeGpu(data, obs, huber, iterations, reuse);
  Result out;
  std::copy(reduced.state, reduced.state + 24, out.state);
  for (int k = 0; k < 3; ++k) {
    out.state[24 + k] = data.bg[k] + reduced.state[18 + k] - data.right_bg[k];
    out.state[27 + k] = data.ba[k] + reduced.state[21 + k] - data.right_ba[k];
  }
  out.initial_cost = reduced.initial_cost;
  out.final_cost = reduced.final_cost;
  out.iterations = reduced.iterations;
  out.buffer_ms = reduced.buffer_ms;
  out.graph_ms = reduced.graph_ms;
  out.lm_ms = reduced.lm_ms;
  out.copy_ms = reduced.copy_ms;
  return out;
}
}  // namespace gpu_bias_rw
namespace cuvslam::sba_imu {
void Pack(const imu::ImuCalibration&, const StereoPnPInput&, const Pose&, const Pose&, float, gpu_vi::Data&,
          std::vector<gpu_vi::Observation>&);
void Unpack(const gpu_vi::Data&, const float*, const Pose&, const Pose&, Pose&, Pose&);
float EvaluateCost(const imu::ImuCalibration&, const StereoPnPInput&, const std::vector<bool>&, const Pose&,
                   const Pose&, const Pose&, const Pose&, const Pose&, float);
bool SolveGpuBiasRW(const imu::ImuCalibration& cal, const StereoPnPInput& problem, Pose& left, Pose& right,
                    float penalty) {
  if (problem.shared_bias || problem.freeze_bias || problem.prior_gyro != 0 || problem.prior_acc != 0 ||
      problem.translation_constraint != 0 || problem.robustifier_scale_pose > 0) {
    return false;
  }
  gpu_bias_rw::Data data;
  std::vector<gpu_vi::Observation> obs;
  Pack(cal, problem, left, right, penalty, data, obs);
  for (int k = 0; k < 3; ++k) {
    data.right_bg[k] = right.gyro_bias[k];
    data.right_ba[k] = right.acc_bias[k];
  }
  Matrix3T bg, ba;
  left.preintegration.InfoGyroRWMatrix(bg);
  left.preintegration.InfoAccRWMatrix(ba);
  Eigen::Matrix<double, 6, 6> rw = Eigen::Matrix<double, 6, 6>::Zero();
  rw.topLeftCorner<3, 3>() = 2. * penalty * bg.cast<double>();
  rw.bottomRightCorner<3, 3>() = 2. * penalty * ba.cast<double>();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(rw);
  if (es.info() != Eigen::Success || es.eigenvalues().minCoeff() <= 0) {
    return false;
  }
  const Eigen::Matrix<double, 6, 6> w = es.eigenvalues().cwiseSqrt().asDiagonal() * es.eigenvectors().transpose();
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      data.rw_sqrt[6 * i + j] = w(i, j);
    }
  }
  const auto solution =
      gpu_bias_rw::OptimizeEliminated(data, obs, problem.robustifier_scale, problem.max_iterations, true);
  Pose l, r;
  Unpack(data, solution.state, left, right, l, r);
  for (int k = 0; k < 3; ++k) {
    r.gyro_bias[k] = data.right_bg[k] + solution.state[24 + k];
    r.acc_bias[k] = data.right_ba[k] + solution.state[27 + k];
  }
  r.gyro_bias = l.gyro_bias;
  r.acc_bias = l.acc_bias;
  Pose zero;
  std::vector<bool> outliers(obs.size(), false);
  const double before = EvaluateCost(cal, problem, outliers, left, left, right, zero, zero, penalty);
  const double after = EvaluateCost(cal, problem, outliers, left, l, r, zero, zero, penalty);
  if (!std::isfinite(after) || after > before + 1e-4 * std::max(1., before)) {
    return false;
  }
  // Evaluate the SAME objective at the accepted LM state, preserving the old prior anchor.
  const auto e = gpu_bias_rw::EvaluateGpu(data, obs, solution.state);
  const int rows = 30 + 2 * obs.size();
  Eigen::MatrixXd J(rows, 30);
  int row = 0;
  auto add = [&](const std::vector<float>& residual, const std::vector<float>& jac, int block, bool robust) {
    for (size_t i = 0; i < residual.size(); i += block) {
      double norm = 0;
      for (int k = 0; k < block; ++k) {
        norm += double(residual[i + k]) * residual[i + k];
      }
      const double weight = robust && norm > problem.robustifier_scale * problem.robustifier_scale
                                ? std::sqrt(problem.robustifier_scale / std::sqrt(norm))
                                : 1.;
      for (int k = 0; k < block; ++k, ++row) {
        for (int c = 0; c < 30; ++c) {
          J(row, c) = weight * jac[(i + k) * 30 + c];
        }
      }
    }
  };
  add(e.imu, e.imu_j, 9, false);
  add(e.prior, e.prior_j, 15, false);
  add(e.rw, e.rw_j, 6, false);
  add(e.visual, e.visual_j, 2, true);
  if (row != rows || !J.allFinite()) {
    return false;
  }
  // Convert additive LM increments to the newly published pose tangent.
  // d Exp(phi)/d phi = Exp(phi)*Jr(phi); translations were in the old body frame.
  for (int side = 0; side < 2; ++side) {
    const int offset = 9 * side;
    Eigen::Vector3d phi;
    for (int k = 0; k < 3; ++k) {
      phi[k] = solution.state[offset + k];
    }
    const double theta = phi.norm();
    Eigen::Matrix3d K;
    K << 0, -phi.z(), phi.y(), phi.z(), 0, -phi.x(), -phi.y(), phi.x(), 0;
    const double a =
        theta < 1e-5 ? 1. / 12. + theta * theta / 720. : (1. - .5 * theta / std::tan(.5 * theta)) / (theta * theta);
    const Eigen::Matrix3d inverse_jr = Eigen::Matrix3d::Identity() + .5 * K + a * K * K;
    const Eigen::Matrix3d oldR = (side == 0 ? left : right).w_from_imu.linear().cast<double>();
    const Eigen::Matrix3d newR = (side == 0 ? l : r).w_from_imu.linear().cast<double>();
    J.middleCols(offset, 3) = (J.middleCols(offset, 3) * inverse_jr).eval();
    J.middleCols(offset + 3, 3) = (J.middleCols(offset + 3, 3) * (oldR.transpose() * newR)).eval();
  }
  // Eliminate the complete old state (including old biases) using square-root QR.
  // RW is already an explicit factor: do not add its process noise a second time.
  Eigen::MatrixXd old(rows, 15), kept(rows, 15);
  old.leftCols(9) = J.leftCols(9);
  old.rightCols(6) = J.middleCols(18, 6);
  kept.leftCols(9) = J.middleCols(9, 9);
  kept.rightCols(6) = J.rightCols(6);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(old);
  qr.setThreshold(1e-10);
  Eigen::MatrixXd transformed = qr.householderQ().adjoint() * kept;
  Eigen::MatrixXd tail = transformed.bottomRows(rows - qr.rank());
  Eigen::MatrixXd info = tail.transpose() * tail;
  if (!info.allFinite() || info.cwiseAbs().maxCoeff() > std::numeric_limits<float>::max()) {
    return false;
  }
  r.info = info.cast<float>();
  TraceMessage("GPU_BIAS_RW_ELIMINATED obs=%zu iterations=%d before=%.9g after=%.9g dbg=%.9g dba=%.9g", obs.size(),
               solution.iterations, before, after, (r.gyro_bias - l.gyro_bias).norm(),
               (r.acc_bias - l.acc_bias).norm());
  left = std::move(l);
  right = std::move(r);
  return true;
}
}  // namespace cuvslam::sba_imu
