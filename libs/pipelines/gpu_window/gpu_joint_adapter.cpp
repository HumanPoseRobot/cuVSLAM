#include "gpu_joint_adapter.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "common/log.h"
#include "gpu_joint_math.h"
#include "gpu_joint_types.h"
namespace cuvslam::sba_imu {
using Mat30 = Eigen::Matrix<float, 30, 30>;
using Vec30 = Eigen::Matrix<float, 30, 1>;
float EvaluateCost(const imu::ImuCalibration&, const StereoPnPInput&, const std::vector<bool>&, const Pose&,
                   const Pose&, const Pose&, const Pose&, const Pose&, float);
void build_hessian(const imu::ImuCalibration&, const StereoPnPInput&, const std::vector<bool>&, const Pose&,
                   const Pose&, const Pose&, float, Mat30&, Vec30&);
template <class M>
void put(double* dst, const M& src) {
  for (int i = 0; i < src.rows(); ++i)
    for (int j = 0; j < src.cols(); ++j) dst[i * src.cols() + j] = src(i, j);
}
template <int N>
void sqrt_info(double* dst, const Eigen::Matrix<double, N, N>& info) {
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> es((info + info.transpose()) * .5);
  if (es.info() != Eigen::Success || es.eigenvalues().minCoeff() < -1e-5 * std::max(1., info.norm()))
    throw std::runtime_error("Non-PSD VI information");
  Eigen::Matrix<double, N, N> w = es.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal() * es.eigenvectors().transpose();
  put(dst, w);
}
void Pack(const imu::ImuCalibration& cal, const StereoPnPInput& input, const Pose& l, const Pose& r, float penalty,
          gpu_vi::Data& d, std::vector<gpu_vi::Observation>& obs) {
  d = {};
  const Pose* ps[2] = {&l, &r};
  for (int i = 0; i < 2; ++i) {
    put(d.R[i], ps[i]->w_from_imu.linear());
    put(d.p[i], ps[i]->w_from_imu.translation());
    put(d.v[i], ps[i]->velocity);
  }
  put(d.bg, l.gyro_bias);
  put(d.ba, l.acc_bias);
  Vector3T bg0, ba0;
  l.preintegration.GetDeltaBias(Vector3T::Zero(), Vector3T::Zero(), bg0, ba0);
  put(d.bg0, -bg0);
  put(d.ba0, -ba0);
  const auto& pi = l.preintegration;
  put(d.dR, pi.dR);
  put(d.dV, pi.dV);
  put(d.dP, pi.dP);
  put(d.JRg, pi.JRg);
  put(d.JVg, pi.JVg);
  put(d.JVa, pi.JVa);
  put(d.JPg, pi.JPg);
  put(d.JPa, pi.JPa);
  put(d.gravity, input.gravity);
  d.dt = pi.dT_s;
  Matrix9T info;
  pi.InfoMatrix(info);
  Eigen::Matrix<double, 9, 9> weighted = 2. * penalty * info.cast<double>();
  const char* anchor = std::getenv("CUVSLAM_GRAVITY_ANCHOR");
  if (anchor && std::strcmp(anchor, "1") == 0) {
    // Position residual is p_j - p_i - v*dt. Scale only that block (x25 on the
    // existing 0.002, i.e. 0.05) so a visual pose update moves velocity.
    // Rotation and the velocity residual stay at the caller penalty.
    Eigen::Matrix<double, 9, 9> S = Eigen::Matrix<double, 9, 9>::Identity();
    S.block<3, 3>(6, 6) *= 5.;
    weighted = S * weighted * S;
  }
  sqrt_info<9>(d.imu_sqrt, weighted);
  sqrt_info<15>(d.prior_sqrt, l.info.cast<double>());
  for (size_t i = 0; i < input.point_ids.size(); ++i) {
    gpu_vi::Observation o{};
    put(o.point, input.points[input.point_ids[i]]);
    Isometry3T ci = input.rig.camera_from_rig[input.camera_ids[i]] * cal.rig_from_imu();
    put(o.Rci, ci.linear());
    put(o.tci, ci.translation());
    put(o.xy, input.observation_xys[i]);
    sqrt_info<2>(o.sqrt_info, input.observation_infos[i].cast<double>());
    o.min_depth = input.min_depth;
    obs.push_back(o);
  }
}
void Unpack(const gpu_vi::Data& d, const float* xf, const Pose& anchor_l, const Pose& anchor_r, Pose& l, Pose& r) {
  double x[24];
  for (int i = 0; i < 24; ++i) x[i] = xf[i];
  l = anchor_l;
  r = anchor_r;
  Pose* poses[2] = {&l, &r};
  for (int s = 0; s < 2; ++s) {
    double R[9], p[3], v[3];
    gpu_vi::pose(d, x + 9 * s, s, R, p, v);
    for (int i = 0; i < 3; ++i) {
      poses[s]->w_from_imu.translation()[i] = p[i];
      poses[s]->velocity[i] = v[i];
      poses[s]->gyro_bias[i] = d.bg[i] + x[18 + i];
      poses[s]->acc_bias[i] = d.ba[i] + x[21 + i];
      for (int j = 0; j < 3; ++j) poses[s]->w_from_imu.linear()(i, j) = R[3 * i + j];
    }
  }
}
double GpuCost(const gpu_vi::Evaluation& e, float huber) {
  double c = 0;
  for (float x : e.imu) c += .5 * x * x;
  for (float x : e.prior) c += .5 * x * x;
  for (size_t k = 0; k < e.visual.size(); k += 2) {
    double n = e.visual[k] * e.visual[k] + e.visual[k + 1] * e.visual[k + 1];
    c += huber > 0 && n > huber * huber ? huber * sqrt(n) - .5 * huber * huber : .5 * n;
  }
  return c;
}
void AuditGpuJoint(const imu::ImuCalibration& cal, const StereoPnPInput& input, const Pose& l, const Pose& r,
                   float penalty) {
  gpu_vi::Data d;
  std::vector<gpu_vi::Observation> obs;
  Pack(cal, input, l, r, penalty, d, obs);
  for (int trial = 0; trial < 2; ++trial) {
    float x[24]{};
    if (trial)
      for (int k = 0; k < 24; ++k) x[k] = ((k % 5) - 2) * .0001f;
    auto e = gpu_vi::EvaluateGpu(d, obs, x);
    Pose ll, rr;
    Unpack(d, x, l, r, ll, rr);
    Pose z;
    std::vector<bool> outliers(input.point_ids.size(), false);
    double cpu = EvaluateCost(cal, input, outliers, l, ll, rr, z, z, penalty);
    double gpu = GpuCost(e, input.robustifier_scale);
    double rel = abs(cpu - gpu) / std::max(1., abs(cpu));
    double hrel = -1, grel = -1, gscaled = -1, gabs = -1, gnorm = -1;
    if (!trial) {
      Eigen::Matrix<double, 24, 24> H = Eigen::Matrix<double, 24, 24>::Zero();
      Eigen::Matrix<double, 24, 1> g = Eigen::Matrix<double, 24, 1>::Zero();
      auto add = [&](const std::vector<float>& rs, const std::vector<float>& js, int n, bool robust) {
        for (size_t base = 0; base < rs.size(); base += n) {
          double sq = 0;
          for (int k = 0; k < n; ++k) sq += rs[base + k] * rs[base + k];
          double w = robust && input.robustifier_scale > 0 && sq > input.robustifier_scale * input.robustifier_scale
                         ? input.robustifier_scale / sqrt(sq)
                         : 1.;
          for (int k = 0; k < n; ++k) {
            Eigen::Matrix<double, 24, 1> j;
            for (int c = 0; c < 24; ++c) j[c] = js[(base + k) * 24 + c];
            H += w * j * j.transpose();
            g -= w * j * rs[base + k];
          }
        }
      };
      add(e.imu, e.imu_j, 9, false);
      add(e.prior, e.prior_j, 15, false);
      add(e.visual, e.visual_j, 2, true);
      Mat30 hc;
      Vec30 gc;
      build_hessian(cal, input, outliers, l, l, r, penalty, hc, gc);
      int ids[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 9, 10, 11, 12, 13, 14};
      Eigen::Matrix<double, 24, 24> ch;
      Eigen::Matrix<double, 24, 1> cg;
      for (int i = 0; i < 24; ++i) {
        cg[i] = 2 * gc[ids[i]];
        for (int j = 0; j < 24; ++j) ch(i, j) = 2 * hc(ids[i], ids[j]);
      }
      hrel = (H - ch).norm() / std::max(1., ch.norm());
      grel = (g - cg).norm() / std::max(1., cg.norm());
      Eigen::Matrix<double, 24, 1> scale = ch.diagonal().cwiseMax(1e-12).cwiseSqrt().cwiseInverse();
      gabs = ((g - cg).cwiseProduct(scale)).norm();
      gnorm = cg.cwiseProduct(scale).norm();
      gscaled = gabs / std::max(1., gnorm);
    }
    TraceMessage(
        "GPU_VI_AUDIT trial=%d cpu=%.9g gpu=%.9g cost_rel=%.9g hessian_rel=%.9g gradient_rel=%.9g "
        "gradient_scaled_rel=%.9g gradient_scaled_abs=%.9g gradient_scaled_norm=%.9g",
        trial, cpu, gpu, rel, hrel, grel, gscaled, gabs, gnorm);
    if (rel > 2e-3 || (trial == 0 && (hrel > 2e-3 || gscaled > 2e-3)))
      throw std::runtime_error("GPU VI equivalence gate failed");
  }
}
// Solver integration is enabled in stage 2 only after the shadow audit passes.
bool SolveGpuJoint(const imu::ImuCalibration& calib, const StereoPnPInput& problem, Pose& left, Pose& right,
                   float penalty, bool reuse_lm) {
  if (!problem.shared_bias || problem.prior_gyro != 0 || problem.prior_acc != 0 ||
      problem.translation_constraint != 0 || problem.robustifier_scale_pose > 0) {
    throw std::runtime_error("GPU VI backend received unsupported objective");
  }
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  gpu_vi::Data data;
  std::vector<gpu_vi::Observation> obs;
  Pack(calib, problem, left, right, penalty, data, obs);
  const auto packed = Clock::now();
  const Pose original_left = left, original_right = right;
  auto solution = gpu_vi::OptimizeGpu(data, obs, problem.robustifier_scale, problem.max_iterations, reuse_lm);
  const auto gpu_returned = Clock::now();
  Pose l, r;
  Unpack(data, solution.state, original_left, original_right, l, r);
  Pose zero;
  std::vector<bool> outliers(problem.point_ids.size(), false);
  const double before =
      EvaluateCost(calib, problem, outliers, original_left, original_left, original_right, zero, zero, penalty);
  const double after = EvaluateCost(calib, problem, outliers, original_left, l, r, zero, zero, penalty);
  const auto costs_ready = Clock::now();
  TraceMessage("GPU_VI_SOLVE iterations=%d cpu_initial=%.9g cpu_final=%.9g gpu_initial=%.9g gpu_final=%.9g",
               solution.iterations, before, after, solution.initial_cost, solution.final_cost);
  if (!std::isfinite(after) || after > before + 1e-4 * std::max(1., before)) {
    return false;
  }
  // Stage 2 preserves the independently tested CPU posterior assembly/Schur
  // update. Residual evaluation, state updates and LM solve run in cuNLS GPU.
  const auto checked = Clock::now();
  Mat30 full;
  Vec30 gradient;
  build_hessian(calib, problem, outliers, original_left, l, r, penalty, full, gradient);
  const auto hessian_ready = Clock::now();
  Eigen::MatrixXd h = full.topLeftCorner<24, 24>().cast<double>();
  h = (.5 * (h + h.transpose())).eval();
  // Eliminate old pose and velocity, retain right pose/velocity and common bias.
  std::array<int, 15> kept{15, 16, 17, 18, 19, 20, 21, 22, 23, 9, 10, 11, 12, 13, 14};
  Eigen::MatrixXd cross(9, 15), info(15, 15);
  for (int j = 0; j < 15; ++j) {
    cross.col(j) = h.block(0, kept[j], 9, 1);
    for (int k = 0; k < 15; ++k) info(k, j) = h(kept[k], kept[j]);
  }
  Eigen::LDLT<Eigen::MatrixXd> prior(h.topLeftCorner(9, 9));
  if (prior.info() != Eigen::Success || !prior.isPositive()) return false;
  info -= cross.transpose() * prior.solve(cross);
  Matrix3T gyro, acc;
  l.preintegration.InfoGyroRWMatrix(gyro);
  l.preintegration.InfoAccRWMatrix(acc);
  Eigen::MatrixXd bias_info = info.bottomRightCorner(6, 6);
  bias_info.topLeftCorner(3, 3) += penalty * gyro.cast<double>();
  bias_info.bottomRightCorner(3, 3) += penalty * acc.cast<double>();
  Eigen::LDLT<Eigen::MatrixXd> random_walk(bias_info);
  if (random_walk.info() != Eigen::Success || !random_walk.isPositive()) return false;
  // Woodbury update for adding the process noise to the retained bias covariance.
  cross = info.rightCols(6);
  info -= cross * random_walk.solve(cross.transpose());
  // build_hessian uses the derivative with respect to squared residuals. A
  // Gaussian prior is evaluated as 0.5 * e' I e, so the next frame consumes
  // 0.5 * I. Restore I = 2 * Schur before passing this quadratic forward.
  r.info = (info + info.transpose()).cast<float>();
  if (!r.info.allFinite()) return false;
  left = std::move(l);
  right = std::move(r);
  const auto done = Clock::now();
  const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  TraceMessage("GPU_VI_SEGMENTS sparse=%d obs=%zu iters=%d pack=%.6f buffer=%.6f graph=%.6f lm=%.6f copy=%.6f "
               "cleanup=%.6f cost_check=%.6f trace=%.6f hessian=%.6f posterior=%.6f total=%.6f",
               int(reuse_lm), obs.size(), solution.iterations, ms(start, packed), solution.buffer_ms, solution.graph_ms,
               solution.lm_ms, solution.copy_ms,
               ms(packed, gpu_returned) - solution.buffer_ms - solution.graph_ms - solution.lm_ms - solution.copy_ms,
               ms(gpu_returned, costs_ready), ms(costs_ready, checked), ms(checked, hessian_ready),
               ms(hessian_ready, done), ms(start, done) - ms(costs_ready, checked));
  return true;
}

}  // namespace cuvslam::sba_imu
