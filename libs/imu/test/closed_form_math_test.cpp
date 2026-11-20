#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include "imu/imu_preintegration.h"
namespace cuvslam::sba_imu {
TEST(ClosedFormMath, ConstantTurnMatchesIndependentScalarIntegrals) {
  for (double theta : {1e-8, .0999999, .1, .1000001, 1., 3.141592653589793, 6., 12.}) {
    const double dt = .2;
    imu::ImuMeasurementStorage samples(4);
    for (int64_t t : {0LL, 200000000LL}) {
      samples.push_back({t, Vector3T(2, 0, 0), Vector3T(0, 0, float(theta / dt))});
    }
    const imu::ImuCalibration cal(Isometry3T::Identity(), .01f, .001f, .1f, .01f, 5.f);
    IMUPreintegration p(cal, samples, Vector3T::Zero(), Vector3T::Zero(), 0, 200000000);
    const double actualTheta = double(float(theta / dt)) * double(float(dt));
    const double a = actualTheta;
    const double sine = std::sin(a), half = std::sin(.5 * a);
    const double oneMinusCos = 2. * half * half;
    const double thetaMinusSin =
        std::abs(a) < 1e-3 ? a * a * a * (1. / 6. - a * a / 120. + a * a * a * a / 5040.) : a - sine;
    const Eigen::Vector3d v(2. * double(float(dt)) * sine / a, 2. * double(float(dt)) * oneMinusCos / a, 0.);
    const Eigen::Vector3d q(2. * double(float(dt)) * double(float(dt)) * oneMinusCos / (a * a),
                            2. * double(float(dt)) * double(float(dt)) * thetaMinusSin / (a * a), 0.);
    SCOPED_TRACE(theta);
    EXPECT_LT((p.dV.cast<double>() - v).norm(), 2e-6);
    EXPECT_LT((p.dP.cast<double>() - q).norm(), 2e-7);
    EXPECT_LT((p.dR.cast<double>() - Eigen::AngleAxisd(a, Eigen::Vector3d::UnitZ()).toRotationMatrix()).norm(), 2e-6);
  }
}
TEST(ClosedFormMath, BiasJacobiansMatchIndependentDoubleIntegration) {
  imu::ImuMeasurementStorage samples(100);
  for (int64_t t = 0; t <= 300000000; t += 5000000) {
    samples.push_back({t, Vector3T(1, -9.3f, .7f), Vector3T(.8f, -.4f, .5f)});
  }
  const imu::ImuCalibration cal;
  const Vector3T bg(.01f, -.02f, .003f), ba(.1f, -.05f, .02f);
  const auto integrate = [&](const Vector3T& g, const Vector3T& a) {
    return IMUPreintegration(cal, samples, g, a, 0, 300000000);
  };
  const auto base = integrate(bg, ba);
  constexpr float eps = .001f;
  double worst = 0.;
  for (int k = 0; k < 3; ++k) {
    const Vector3T delta = eps * Vector3T::Unit(k);
    const auto gp = integrate(bg + delta, ba), gm = integrate(bg - delta, ba);
    const auto ap = integrate(bg, ba + delta), am = integrate(bg, ba - delta);
    const Eigen::Matrix3d dr = base.dR.cast<double>().transpose() * (gp.dR - gm.dR).cast<double>() / (2. * eps);
    const Eigen::Vector3d jr = .5 * Eigen::Vector3d(dr(2, 1) - dr(1, 2), dr(0, 2) - dr(2, 0), dr(1, 0) - dr(0, 1));
    const auto check = [&](const Eigen::Vector3d& numeric, const Eigen::Vector3d& exact) {
      worst = std::max(worst, (numeric - exact).norm() / std::max(.01, exact.norm()));
    };
    check(jr, base.JRg.col(k).cast<double>());
    check((gp.dV - gm.dV).cast<double>() / (2. * eps), base.JVg.col(k).cast<double>());
    check((gp.dP - gm.dP).cast<double>() / (2. * eps), base.JPg.col(k).cast<double>());
    check((ap.dV - am.dV).cast<double>() / (2. * eps), base.JVa.col(k).cast<double>());
    check((ap.dP - am.dP).cast<double>() / (2. * eps), base.JPa.col(k).cast<double>());
  }
  // Float-output differences lose significance for small point/gyro columns.
  // Keep that diagnostic, and use independent FP64 Simpson quadrature to
  // judge the analytic bias Jacobians instead of loosening its tolerance.
  RecordProperty("float_reintegration_difference_max_relative", worst);
  struct Reference {
    Eigen::Matrix3d R;
    Eigen::Vector3d V, P;
  };
  const double T = 60. * double(float(.005));
  const auto reference = [&](const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc) {
    Reference out;
    out.V.setZero();
    out.P.setZero();
    const Eigen::Vector3d omega = Vector3T(.8f, -.4f, .5f).cast<double>() - gyro;
    const Eigen::Vector3d force = Vector3T(1, -9.3f, .7f).cast<double>() - acc;
    const double speed = omega.norm();
    out.R = Eigen::AngleAxisd(speed * T, omega / speed).toRotationMatrix();
    constexpr int N = 2000;
    const double h = T / N;
    for (int i = 0; i <= N; ++i) {
      const double t = i * h, weight = (i == 0 || i == N) ? 1. : (i % 2 ? 4. : 2.);
      const Eigen::Vector3d rotated = Eigen::AngleAxisd(speed * t, omega / speed) * force;
      out.V += weight * rotated;
      out.P += weight * (T - t) * rotated;
    }
    out.V *= h / 3.;
    out.P *= h / 3.;
    return out;
  };
  const auto origin = reference(bg.cast<double>(), ba.cast<double>());
  EXPECT_LT((base.dV.cast<double>() - origin.V).norm(), 1e-5);
  EXPECT_LT((base.dP.cast<double>() - origin.P).norm(), 1e-6);
  worst = 0.;
  constexpr double epsilon = 1e-5;
  for (int k = 0; k < 3; ++k) {
    const Eigen::Vector3d shift = epsilon * Eigen::Vector3d::Unit(k);
    const auto gp = reference(bg.cast<double>() + shift, ba.cast<double>()),
               gm = reference(bg.cast<double>() - shift, ba.cast<double>());
    const auto ap = reference(bg.cast<double>(), ba.cast<double>() + shift),
               am = reference(bg.cast<double>(), ba.cast<double>() - shift);
    const Eigen::Matrix3d dr = origin.R.transpose() * (gp.R - gm.R) / (2. * epsilon);
    const Eigen::Vector3d jr = .5 * Eigen::Vector3d(dr(2, 1) - dr(1, 2), dr(0, 2) - dr(2, 0), dr(1, 0) - dr(0, 1));
    const auto check = [&](const Eigen::Vector3d& numeric, const Eigen::Vector3d& exact) {
      worst = std::max(worst, (numeric - exact).norm() / std::max(.01, exact.norm()));
    };
    check(jr, base.JRg.col(k).cast<double>());
    check((gp.V - gm.V) / (2. * epsilon), base.JVg.col(k).cast<double>());
    check((gp.P - gm.P) / (2. * epsilon), base.JPg.col(k).cast<double>());
    check((ap.V - am.V) / (2. * epsilon), base.JVa.col(k).cast<double>());
    check((ap.P - am.P) / (2. * epsilon), base.JPa.col(k).cast<double>());
  }
  RecordProperty("independent_double_bias_jacobian_max_relative", worst);
  EXPECT_LT(worst, 1e-5);
}
}  // namespace cuvslam::sba_imu
