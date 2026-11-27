#pragma once
#include <Eigen/Cholesky>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <cerrno>
#include "camera/observation.h"
#include <stdexcept>
#include "common/imu_calibration.h"
#include "imu/imu_preintegration.h"
#include "window_types.h"

namespace gpu_window::posterior {
template <class Derived>
auto StoredSymmetric(const Eigen::MatrixBase<Derived>& input) {
  using Scalar = typename Derived::Scalar;
  using Result = Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
  const Result value = input.template cast<double>();
  const double error = (value - value.transpose()).norm();
  const double roundoff = 64. * Eigen::NumTraits<Scalar>::epsilon() * input.rows() * value.norm();
  if (!value.allFinite() || error > roundoff) {
    throw std::invalid_argument("sensor matrix asymmetry exceeds storage roundoff");
  }
  return Result(.5 * (value + value.transpose()));
}

template <int N>
Eigen::Matrix<double, N, N> CovarianceWhitening(const Eigen::Matrix<double, N, N>& covariance) {
  if (!covariance.allFinite()) {
    throw std::invalid_argument("nonfinite sensor covariance");
  }
  Eigen::LLT<Eigen::Matrix<double, N, N>> factor(StoredSymmetric(covariance));
  if (factor.info() != Eigen::Success) {
    throw std::runtime_error("sensor covariance is not positive definite");
  }
  // W C W^T = I, hence W^T W = C^-1. There is no empirical
  // information multiplier and no diagonal replacement for unknown covariance.
  return factor.matrixL().solve(Eigen::Matrix<double, N, N>::Identity());
}

template <int N>
Eigen::Matrix<double, N, N> InformationWhitening(const Eigen::Matrix<double, N, N>& information) {
  if (!information.allFinite()) {
    throw std::invalid_argument("nonfinite observation information");
  }
  Eigen::LLT<Eigen::Matrix<double, N, N>> factor(StoredSymmetric(information));
  if (factor.info() != Eigen::Success) {
    throw std::runtime_error("observation information is not positive definite");
  }
  return factor.matrixU();
}

// Diagnostic controls reproduce the documented rc2 information multipliers.
// They are OFF by default and are not a sensor-noise calibration or a release fix.
inline Eigen::Matrix2d VisualWhitening(const Eigen::Matrix2d& information) {
  Eigen::Matrix2d result = InformationWhitening<2>(information);
  const char* configured = std::getenv("VIO_POSTERIOR_VISUAL_SIGMA_PX");
  if (configured) {
    if (std::getenv("VIO_POSTERIOR_DIAG_RC2_VISUAL")) {
      throw std::invalid_argument("configured visual noise conflicts with legacy diagnostic");
    }
    char* end = nullptr;
    errno = 0;
    const double sigma = std::strtod(configured, &end);
    if (end == configured || *end != 0 || errno == ERANGE || !(sigma > 0.) || !std::isfinite(sigma)) {
      throw std::invalid_argument("visual pixel sigma must be finite and positive");
    }
    // Check the resulting metric before products can overflow/underflow.
    const double log_norm = std::log(result.cwiseAbs().maxCoeff()) +
                            std::log(cuvslam::camera::kDefaultObservationNoisePx) - std::log(sigma);
    if (log_norm > .5 * (std::log(std::numeric_limits<double>::max()) - std::log(2.)) ||
        log_norm < .5 * std::log(std::numeric_limits<double>::min())) {
      throw std::invalid_argument("visual pixel sigma exceeds numerical range");
    }
    const double scale = cuvslam::camera::kDefaultObservationNoisePx / sigma;
    result *= scale;
    const Eigen::Matrix2d metric = result.transpose() * result;
    Eigen::LLT<Eigen::Matrix2d> check(metric);
    if (!metric.allFinite() || check.info() != Eigen::Success) {
      throw std::invalid_argument("visual pixel sigma exceeds numerical range");
    }
  } else if (std::getenv("VIO_POSTERIOR_DIAG_RC2_VISUAL")) {
    result *= cuvslam::camera::kDefaultObservationNoisePx;
  }
  return result;
}
inline double DiagnosticImuWhiteningScale() {
  return std::getenv("VIO_POSTERIOR_DIAG_RC2_IMU") ? std::sqrt(.002) : 1.;
}

template <class Derived>
void CopyRows(double* destination, const Eigen::MatrixBase<Derived>& value) {
  for (int r = 0; r < value.rows(); ++r) {
    for (int c = 0; c < value.cols(); ++c) {
      destination[r * value.cols() + c] = value(r, c);
    }
  }
}

inline Edge CalibratedEdge(const cuvslam::sba_imu::IMUPreintegration& integration,
                           const cuvslam::imu::ImuCalibration& calibration, int left, int right) {
  using namespace cuvslam;
  if (integration.empty() || !(integration.dT_s > 0.) || !std::isfinite(integration.dT_s)) {
    throw std::invalid_argument("missing preintegration for joint posterior");
  }
  Edge edge;
  edge.left = left;
  edge.right = right;
  auto& d = edge.data;
  CopyRows(d.dR, integration.dR.cast<double>().eval());
  CopyRows(d.dV, integration.dV.cast<double>().eval());
  CopyRows(d.dP, integration.dP.cast<double>().eval());
  CopyRows(d.JRg, integration.JRg.cast<double>().eval());
  CopyRows(d.JVg, integration.JVg.cast<double>().eval());
  CopyRows(d.JVa, integration.JVa.cast<double>().eval());
  CopyRows(d.JPg, integration.JPg.cast<double>().eval());
  CopyRows(d.JPa, integration.JPa.cast<double>().eval());
  CopyRows(d.bg0, integration.GetOriginalGyroBias().cast<double>().eval());
  CopyRows(d.ba0, integration.GetOriginalAccBias().cast<double>().eval());
  d.dt = integration.dT_s;
  CopyRows(d.imu_sqrt, (DiagnosticImuWhiteningScale() * CovarianceWhitening<9>(StoredSymmetric(integration.cov_matrix_))).eval());
  // Calibration stores per-sample RW covariance. The integrator accumulates
  // covariance * frequency * dt, i.e. continuous RW density squared * dt.
  if (!(calibration.frequency() > 0.)) {
    throw std::invalid_argument("invalid calibrated sampling frequency");
  }
  // Use the actual accumulated matrices. Rebuilding them from a separately
  // rounded float dT changes the factor and is not a noise-model conversion.
  const Eigen::Matrix3d bg = StoredSymmetric(integration.gyro_random_walk_accum_cov_matrix_);
  const Eigen::Matrix3d ba = StoredSymmetric(integration.acc_random_walk_accum_cov_matrix_);
  Eigen::Matrix<double, 6, 6> whiten = Eigen::Matrix<double, 6, 6>::Zero();
  whiten.topLeftCorner<3, 3>() = CovarianceWhitening<3>(bg);
  whiten.bottomRightCorner<3, 3>() = CovarianceWhitening<3>(ba);
  CopyRows(edge.rw, (DiagnosticImuWhiteningScale() * whiten).eval());
  return edge;
}
}  // namespace gpu_window::posterior
