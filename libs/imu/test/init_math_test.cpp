#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include "imu/inertial_optimization.h"
#include "imu/lm_model_reduction.h"

namespace cuvslam::sba_imu {
struct InertialOptimizerMathAccess {
  static void Build(const InertialOptimizer& o, const std::vector<Pose>& p, const Matrix3T& R, const Vector3T& bg,
                    const Vector3T& ba, float robust, Eigen::MatrixXf& H, Eigen::VectorXf& g) {
    o.build_hessian(p, R, bg, ba, robust, H, g);
  }
  static float Cost(const InertialOptimizer& o, const std::vector<Pose>& p, const Matrix3T& R, const Vector3T& bg,
                    const Vector3T& ba, const Eigen::VectorXf& x, float robust) {
    return o.calc_cost_with_update(p, R, bg, ba, x, robust);
  }
};
namespace {
Eigen::Matrix3d ExpReference(const Eigen::Vector3d& x) {
  const double a = x.norm();
  return a == 0. ? Eigen::Matrix3d::Identity() : Eigen::AngleAxisd(a, x / a).toRotationMatrix();
}
Eigen::Vector3d LogReference(const Eigen::Matrix3d& R) {
  const Eigen::AngleAxisd a(R);
  return a.angle() * a.axis();
}
}  // namespace
TEST(InitializationMath, ProductionCostGradientAndGaussNewtonMatchIndependentDoubleDifferences) {
  imu::ImuMeasurementStorage samples(100);
  for (int64_t t = 0; t <= 300000000; t += 5000000) {
    samples.push_back({t, Vector3T(.3f, -9.7f, .2f), Vector3T(.1f, -.07f, .05f)});
  }
  const imu::ImuCalibration cal;
  std::vector<Pose> poses(4);
  for (int i = 0; i < 4; ++i) {
    poses[i].w_from_imu.linear() = Eigen::AngleAxisf(.1f * i, Vector3T(1, 2, 3).normalized()).toRotationMatrix();
    poses[i].w_from_imu.translation() = Vector3T(.03f * i, .01f * i * i, -.02f * i);
    poses[i].velocity = Vector3T(.4f + .02f * i, -.2f, .1f);
    if (i < 3) {
      poses[i].preintegration =
          IMUPreintegration(cal, samples, Vector3T::Zero(), Vector3T::Zero(), 100000000LL * i, 100000000LL * (i + 1));
      poses[i].preintegration.cov_matrix_.setIdentity();
    }
  }
  const Matrix3T R = Eigen::AngleAxisf(.4f, Vector3T(2, -1, 3).normalized()).toRotationMatrix();
  const Vector3T bg(.003f, -.002f, .001f), ba(.02f, -.03f, .01f);
  constexpr double prior = 3.7;
  InertialOptimizer optimizer(prior);
  const int n = 8 + 3 * poses.size();
  const auto residual = [&](const Eigen::VectorXd& x) {
    Eigen::VectorXd out(9 * (poses.size() - 1));
    const Eigen::Vector3d w(x[0], 0., x[1]);
    const Eigen::Vector3d gravity = R.cast<double>() * ExpReference(w) * optimizer.get_default_gravity().cast<double>();
    for (size_t i = 0; i + 1 < poses.size(); ++i) {
      const auto& pi = poses[i].preintegration;
      const Eigen::Vector3d db = bg.cast<double>() + x.segment<3>(2) - pi.GetOriginalGyroBias().cast<double>();
      const Eigen::Vector3d da = ba.cast<double>() + x.segment<3>(5) - pi.GetOriginalAccBias().cast<double>();
      const Eigen::Matrix3d dR = pi.dR.cast<double>() * ExpReference(pi.JRg.cast<double>() * db);
      const Eigen::Vector3d dV = pi.dV.cast<double>() + pi.JVg.cast<double>() * db + pi.JVa.cast<double>() * da;
      const Eigen::Vector3d dP = pi.dP.cast<double>() + pi.JPg.cast<double>() * db + pi.JPa.cast<double>() * da;
      const Eigen::Matrix3d R1 = poses[i].w_from_imu.linear().cast<double>().transpose();
      const Eigen::Vector3d v1 = poses[i].velocity.cast<double>() + x.segment<3>(8 + 3 * i);
      const Eigen::Vector3d v2 = poses[i + 1].velocity.cast<double>() + x.segment<3>(11 + 3 * i);
      const double dt = pi.dT_s;
      out.segment<3>(9 * i) = LogReference(dR.transpose() * R1 * poses[i + 1].w_from_imu.linear().cast<double>());
      out.segment<3>(9 * i + 3) = R1 * (v2 - v1 - gravity * dt) - dV;
      out.segment<3>(9 * i + 6) =
          R1 * (poses[i + 1].w_from_imu.translation().cast<double>() -
                poses[i].w_from_imu.translation().cast<double>() - v1 * dt - .5 * gravity * dt * dt) -
          dP;
    }
    return out;
  };
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(n), r = residual(zero);
  Eigen::MatrixXd J(r.size(), n);
  for (int c = 0; c < n; ++c) {
    auto plus = zero, minus = zero;
    plus[c] = 1e-6;
    minus[c] = -1e-6;
    J.col(c) = (residual(plus) - residual(minus)) / 2e-6;
  }
  for (float robust : {1000.f, .2f}) {
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(n, n);
    Eigen::VectorXf gradient = Eigen::VectorXf::Zero(n);
    InertialOptimizerMathAccess::Build(optimizer, poses, R, bg, ba, robust, H, gradient);
    Eigen::MatrixXd referenceH = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd referenceG = Eigen::VectorXd::Zero(n);
    double referenceCost = 0.;
    for (int i = 0; i < 3; ++i) {
      const double s = r.segment<9>(9 * i).squaredNorm();
      const double norm = std::sqrt(s);
      const double weight = norm < robust ? .5 : .5 * robust / norm;
      referenceCost += norm < robust ? .5 * s : robust * norm - .5 * robust * robust;
      referenceH += weight * J.middleRows(9 * i, 9).transpose() * J.middleRows(9 * i, 9);
      referenceG += weight * J.middleRows(9 * i, 9).transpose() * r.segment<9>(9 * i);
    }
    referenceH.block<3, 3>(5, 5) += prior * Eigen::Matrix3d::Identity();
    referenceG.segment<3>(5) += prior * ba.cast<double>();
    referenceCost += prior * ba.squaredNorm();
    const Eigen::MatrixXd assembled = Eigen::MatrixXf(H.selfadjointView<Eigen::Lower>()).cast<double>();
    const double he = (assembled - referenceH).norm() / referenceH.norm();
    const double ge = (gradient.cast<double>() - referenceG).norm() / referenceG.norm();
    RecordProperty(robust > 1 ? "quadratic_hessian_relative" : "huber_hessian_relative", he);
    RecordProperty(robust > 1 ? "quadratic_gradient_relative" : "huber_gradient_relative", ge);
    EXPECT_LT(he, 3e-5);
    EXPECT_LT(ge, 3e-5);
    const auto z = Eigen::VectorXf::Zero(n).eval();
    const double cost = InertialOptimizerMathAccess::Cost(optimizer, poses, R, bg, ba, z, robust);
    EXPECT_NEAR(cost, referenceCost, 2e-6 * std::max(1., referenceCost));
    double worst = 0.;
    for (int c = 0; c < n; ++c) {
      auto plus = z, minus = z;
      plus[c] = .001f;
      minus[c] = -.001f;
      const double numeric = (InertialOptimizerMathAccess::Cost(optimizer, poses, R, bg, ba, plus, robust) -
                              InertialOptimizerMathAccess::Cost(optimizer, poses, R, bg, ba, minus, robust)) /
                             .002;
      worst = std::max(worst, std::abs(numeric - 2. * gradient[c]) / std::max(1., std::abs(2. * gradient[c])));
    }
    RecordProperty(robust > 1 ? "quadratic_cost_gradient_max_relative" : "huber_cost_gradient_max_relative", worst);
    EXPECT_LT(worst, .002);
    const Eigen::VectorXf scaling = H.diagonal();
    const float lambda = .4f;
    const Eigen::MatrixXf normal = H.selfadjointView<Eigen::Lower>();
    const Eigen::VectorXf step = (normal + (lambda * scaling).asDiagonal().toDenseMatrix()).ldlt().solve(-gradient);
    const double expected = (-2. * gradient.dot(step) - step.dot(normal * step)) / cost;
    EXPECT_NEAR(InitPredictedReduction(H, step, scaling, lambda, cost), expected, 2e-6);
    const double legacy = (step.dot(H * step) + 2. * lambda * step.dot(scaling.asDiagonal() * step)) / cost;
    EXPECT_GT(std::abs(legacy - expected), 1e-4);
  }
}
}  // namespace cuvslam::sba_imu

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
namespace cuvslam::sba_imu {
TEST(InitializationMath, ActualOptimizerUsesTheVerifiedSymmetricModelByDefault) {
  imu::ImuMeasurementStorage samples(1000);
  for (int64_t ns = 0; ns <= 1100000000; ns += 5000000) {
    const double t = ns * 1e-9;
    samples.push_back({ns, Vector3T(1.5f * std::sin(2 * t), -9.81f + .3f * std::cos(3 * t), std::cos(t)),
                       Vector3T(.2f, .25f * std::sin(2 * t), .3f)});
  }
  const imu::ImuCalibration cal;
  std::vector<Pose> original(11);
  original[0].velocity = Vector3T(.7f, .2f, .1f);
  const Vector3T gravity(0, GRAVITY_VALUE, 0);
  for (size_t i = 0; i + 1 < original.size(); ++i) {
    original[i].preintegration =
        IMUPreintegration(cal, samples, Vector3T::Zero(), Vector3T::Zero(), 100000000LL * i, 100000000LL * (i + 1));
    ASSERT_TRUE(original[i].predict_pose(gravity, original[i + 1]));
  }
  for (size_t i = 1; i < original.size(); ++i) {
    original[i].w_from_imu.translation() +=
        Vector3T(.001f * std::sin(i), .0015f * std::cos(2 * i), .001f * std::sin(3 * i));
  }
  const auto parse = [&](const std::string& text, const std::string& key) {
    const size_t at = text.find("\"" + key + "\":");
    if (at == std::string::npos) {
      throw std::runtime_error("Missing scalar in production LM audit: " + key);
    }
    return std::stod(text.substr(at + key.size() + 3));
  };
  const auto vector = [&](const std::string& text, const std::string& key) {
    const size_t at = text.find("\"" + key + "\":[");
    if (at == std::string::npos) {
      throw std::runtime_error("Missing array in production LM audit: " + key);
    }
    const size_t start = at + key.size() + 4, end = text.find(']', start);
    std::string numbers = text.substr(start, end - start);
    std::replace(numbers.begin(), numbers.end(), ',', ' ');
    std::istringstream in(numbers);
    std::vector<double> values;
    double x;
    while (in >> x) {
      values.push_back(x);
    }
    return values;
  };
  const char* previousDir = std::getenv("VIO_INIT_SNAPSHOT_DIR");
  const std::string savedDir = previousDir ? previousDir : "";
  const char* previousMode = std::getenv("VIO_INIT_SYMMETRIC_GAIN");
  const std::string savedMode = previousMode ? previousMode : "";
  unsetenv("VIO_INIT_SYMMETRIC_GAIN");
  for (int adaptive = 0; adaptive < 2; ++adaptive) {
    const std::string directory =
        "/tmp/cuvslam_production_lm_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(directory);
    setenv("VIO_INIT_SNAPSHOT_DIR", directory.c_str(), 1);
    InertialOptimizer optimizer(10.f);
    auto poses = original;
    Matrix3T R = Matrix3T::Identity();
    const bool ok = adaptive ? optimizer.OptimizeInertialAdaptive(
                                   poses, cal, R, 1000.f, std::vector<std::vector<camera::Observation>>(poses.size()),
                                   Isometry3T::Identity())
                             : optimizer.optimize_inertial(poses, cal, R, 1000.f);
    if (previousDir) {
      setenv("VIO_INIT_SNAPSHOT_DIR", savedDir.c_str(), 1);
    } else {
      unsetenv("VIO_INIT_SNAPSHOT_DIR");
    }
    EXPECT_TRUE(ok);
    int count = 0;
    double worst = 0.;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
      if (entry.path().extension() != ".jsonl") {
        continue;
      }
      std::ifstream stream(entry.path());
      std::string line;
      while (std::getline(stream, line)) {
        const int n = int(parse(line, "size"));
        const auto h = vector(line, "H"), d = vector(line, "step"), scale = vector(line, "scaling");
        ASSERT_EQ(h.size(), size_t(n * n));
        ASSERT_EQ(d.size(), size_t(n));
        ASSERT_EQ(scale.size(), size_t(n));
        const Eigen::MatrixXd H =
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(h.data(), n, n);
        const Eigen::MatrixXd symmetric = H.selfadjointView<Eigen::Lower>();
        const Eigen::Map<const Eigen::VectorXd> step(d.data(), n), scaling(scale.data(), n);
        const double before = parse(line, "before"), lambda = parse(line, "lambda");
        const double reference =
            (step.dot(symmetric * step) + 2. * lambda * step.dot(scaling.asDiagonal() * step)) / before;
        const double actual = parse(line, "predicted");
        worst = std::max(worst, std::abs(actual - reference) / std::max(1., std::abs(reference)));
        ++count;
      }
    }
    RecordProperty(adaptive ? "adaptive_production_lm_max_relative" : "standard_production_lm_max_relative", worst);
    RecordProperty(adaptive ? "adaptive_production_lm_audit" : "standard_production_lm_audit", directory);
    EXPECT_GT(count, 0) << "Actual optimizer never reached LM; wiring was not tested";
    EXPECT_LT(worst, 3e-6) << "Production call uses a different quadratic model";
  }
  if (previousMode) {
    setenv("VIO_INIT_SYMMETRIC_GAIN", savedMode.c_str(), 1);
  } else {
    unsetenv("VIO_INIT_SYMMETRIC_GAIN");
  }
}
}  // namespace cuvslam::sba_imu
