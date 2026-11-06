
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include <Eigen/Geometry>
#include "common/include_gtest.h"
#include "imu/imu_preintegration.h"
#include "imu/world_velocity.h"

namespace cuvslam::sba_imu {
namespace {
imu::ImuMeasurementStorage Samples(int64_t end_ns = 30000000, bool ramp = false) {
  imu::ImuMeasurementStorage s(100);
  for (int64_t t = 0; t <= end_ns; t += 5000000) {
    s.push_back({t, Vector3T(ramp ? t * 1e-9f : 2.f, 0, 0), Vector3T(0, 0, ramp ? 0.f : 1.f)});
  }
  return s;
}
IMUPreintegration Integrate(const imu::ImuMeasurementStorage& s, int64_t a, int64_t b) {
  return {imu::ImuCalibration{}, s, Vector3T::Zero(), Vector3T::Zero(), a, b};
}
TEST(ImuBoundaryTest, ExactCameraEndpoints) {
  auto s = Samples();
  for (const auto& interval :
       {std::pair<int64_t, int64_t>{10000000, 20000000}, {6000000, 8000000}, {1333333, 28333333}}) {
    auto p = Integrate(s, interval.first, interval.second);
    float dt = (interval.second - interval.first) * 1e-9f;
    EXPECT_NEAR(p.dT_s, dt, 1e-8f);
    EXPECT_NEAR(Eigen::AngleAxisf(p.dR).angle(), dt, 1e-6f);
  }
}
TEST(ImuBoundaryTest, StationaryAccelerationAnalytic) {
  auto s = Samples();
  for (auto& m : s) {
    m.angular_velocity.setZero();
  }
  auto p = Integrate(s, 6000000, 23000000);
  EXPECT_NEAR(p.dV.x(), 2.f * .017f, 1e-7f);
  EXPECT_NEAR(p.dP.x(), .017f * .017f, 1e-8f);
}
TEST(ImuBoundaryTest, InterpolatesBracketingMeasurements) {
  auto p = Integrate(Samples(30000000, true), 6000000, 8000000);
  EXPECT_NEAR(p.dV.x(), .5f * (.008f * .008f - .006f * .006f), 1e-10f);
}
TEST(ImuBoundaryTest, ReintegrationKeepsFractionalDurations) {
  auto p = Integrate(Samples(), 6000000, 23000000);
  p.SetNewBias(Vector3T(0, 0, .2f), Vector3T(.3f, 0, 0));
  p.Reintegrate(imu::ImuCalibration{});
  IMUPreintegration expected(imu::ImuCalibration{}, Samples(), Vector3T(0, 0, .2f), Vector3T(.3f, 0, 0), 6000000,
                             23000000);
  EXPECT_NEAR(p.dT_s, .017f, 1e-8f);
  EXPECT_LT((p.dP - expected.dP).norm(), 1e-8f);
  EXPECT_LT((p.dR - expected.dR).norm(), 1e-7f);
  EXPECT_LT((p.cov_matrix_ - expected.cov_matrix_).norm(), 1e-10f);
  auto before = p.dP;
  p.Reintegrate(imu::ImuCalibration{});
  EXPECT_LT((before - p.dP).norm(), 1e-8f);
}
TEST(ImuBoundaryTest, RejectsFutureOnlyAndStaleCoverage) {
  imu::ImuMeasurementStorage s(10);
  s.push_back({10000000, Vector3T::Zero(), Vector3T::Zero()});
  EXPECT_TRUE(Integrate(s, 6000000, 8000000).empty());
  EXPECT_TRUE(Integrate(s, 10000000, 30000000).empty());
  EXPECT_TRUE(Integrate(s, 10000000, 10000000).empty());
  EXPECT_TRUE(Integrate(s, 20000000, 10000000).empty());
}
TEST(ImuBoundaryTest, CausalTailCoversOnlyRequestedFraction) {
  auto p = Integrate(Samples(20000000), 16000000, 23000000);
  EXPECT_NEAR(p.dT_s, .007f, 1e-8f);
}
TEST(ImuBoundaryTest, RangeQueryNeverReturnsFutureSample) {
  auto s = Samples();
  int count = 0;
  s.iterate_from_to(6000000, 8000000, [&](const auto& m) { ++count; });
  EXPECT_EQ(count, 0);
}
TEST(ImuBoundaryTest, WhiteNoiseAndRandomWalkRespectDuration) {
  auto s = Samples();
  for (auto& m : s) {
    m.linear_acceleration.setZero();
    m.angular_velocity.setZero();
  }
  auto a = Integrate(s, 0, 10000000), b = Integrate(s, 1000000, 11000000);
  EXPECT_NEAR(a.cov_matrix_(0, 0), b.cov_matrix_(0, 0), 1e-12f);
  EXPECT_NEAR(a.acc_random_walk_accum_cov_matrix_(0, 0), b.acc_random_walk_accum_cov_matrix_(0, 0), 1e-12f);
}
TEST(ImuVelocityTest, TranslationHasCorrectSign) {
  Isometry3T a = Isometry3T::Identity(), b = a;
  b.translation().x() = .1f;
  EXPECT_LT((WorldVelocity(a, b, .1f) - Vector3T(1, 0, 0)).norm(), 1e-6f);
}
TEST(ImuVelocityTest, PureRotationAtNonzeroPositionHasZeroVelocity) {
  Isometry3T a = Isometry3T::Identity();
  a.translation() = Vector3T(1, .3f, .2f);
  auto b = a;
  b.linear() = Eigen::AngleAxisf(1.57f, Vector3T::UnitZ()).toRotationMatrix();
  EXPECT_LT(WorldVelocity(a, b, .1f).norm(), 1e-7f);
}
}  // namespace
}  // namespace cuvslam::sba_imu

#include "imu/soft_inertial_pnp.h"

namespace cuvslam::sba_imu {
namespace {
struct StaticRigFrame {
  imu::ImuCalibration calib{Isometry3T::Identity(), .000174875f, .000004f, .00207896f, .0002f, 200.f};
  Pose left, right;
  StereoPnPInput problem;
  StaticRigFrame() {
    imu::ImuMeasurementStorage samples(32);
    for (int n = 0; n <= 8; ++n) {
      samples.push_back({5000000LL*n, Vector3T::Zero(), Vector3T::Zero()});
    }
    left.preintegration = {calib, samples, Vector3T::Zero(), Vector3T::Zero(), 0, 33333333};
    left.info.setIdentity();
    left.info *= 1e4f;
    problem.shared_bias = true;
    problem.min_depth = .05f;
    problem.prior_acc = problem.prior_gyro = problem.translation_constraint = 0.f;
    problem.robustifier_scale_pose = -1.f;
    problem.gravity.setZero();
    problem.rig.num_cameras = 4;
    problem.max_iterations = 20;
    for (int cam = 0; cam < 4; ++cam) {
      auto& extrinsic = problem.rig.camera_from_rig[cam];
      extrinsic = Isometry3T::Identity();
      extrinsic.translation().x() = .03f * cam;
      for (int k = 0; k < 12; ++k) {
        Vector3T xyz((k%4-1.5f)*.12f,(k/4-1.f)*.15f,1.4f+.03f*k);
        Vector3T pc = extrinsic * xyz;
        problem.point_ids.push_back(static_cast<int>(problem.points.size()));
        problem.points.push_back(xyz);
        problem.camera_ids.push_back(cam);
        problem.observation_xys.push_back(pc.head<2>() / pc.z());
        problem.observation_infos.push_back(Matrix2T::Identity()*1e5f);
      }
    }
    left.predict_pose(problem.gravity, left.preintegration, right);
  }
};

TEST(JointFourFrameTest, StaticStatePreservesBiasInformation) {
  StaticRigFrame frame;
  Matrix9T info;
  frame.left.preintegration.InfoMatrix(info);
  ASSERT_GT(info.diagonal().minCoeff(), 0.f);
  ASSERT_TRUE(SoftInertialPnP(frame.calib, frame.problem, frame.left, frame.right, .001f));
  EXPECT_LT(frame.right.w_from_imu.translation().norm(), 1e-6f);
  EXPECT_LT(frame.right.velocity.norm(), 1e-6f);
  EXPECT_GT(frame.right.info.diagonal().segment<6>(9).minCoeff(), 9900.f);
}

TEST(JointFourFrameTest, CorrectsPerturbedPrediction) {
  StaticRigFrame frame;
  frame.right.w_from_imu.translation().x() += .03f;
  frame.right.velocity.x() += .25f;
  ASSERT_TRUE(SoftInertialPnP(frame.calib, frame.problem, frame.left, frame.right, .001f));
  EXPECT_LT(frame.right.w_from_imu.translation().norm(), .005f);
  EXPECT_LT(frame.right.velocity.norm(), .05f);
  EXPECT_TRUE(frame.right.info.allFinite());
  EXPECT_LT((frame.right.gyro_bias-frame.left.gyro_bias).norm(), 1e-7f);
  EXPECT_LT((frame.right.acc_bias-frame.left.acc_bias).norm(), 1e-7f);
}
}  // namespace
}  // namespace cuvslam::sba_imu
