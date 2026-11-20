#include <iomanip>
#include "pnp/cauchy_parameters.h"
#include <cunls/robustifier/cauchy_loss_function_batch.h>
#include "pnp/visual_support.h"
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

#include <cunls/common/device_vector.h>
#include <cunls/factor/pnp_factor_batch.h>
#include <cunls/state/se3_state_batch.h>
#include <limits>
#include <random>
#include "pnp/multicam_pnp.h"

#include "common/include_gtest.h"
#include "common/isometry.h"
#include "common/types.h"
#include "common/vector_2t.h"
#include "common/vector_3t.h"

#include "camera/camera.h"
#include "camera/observation.h"
#include "camera/rig.h"
#include "math/twist.h"

#include "pnp/multisensor_pose_estimator.h"

namespace test::pnp {

using namespace cuvslam;

namespace {

// Synthetic pinhole camera: 640×480, focal=320, principal=(320,240)
const Vector2T kResolution{640.f, 480.f};
const Vector2T kFocal{320.f, 320.f};
const Vector2T kPrincipal{320.f, 240.f};

camera::PinholeCameraModel MakePinhole() { return camera::PinholeCameraModel(kResolution, kFocal, kPrincipal); }

camera::Rig MakeRig(const camera::ICameraModel& cam) {
  camera::Rig rig;
  rig.num_cameras = 1;
  rig.camera_from_rig[0] = Isometry3T::Identity();
  rig.intrinsics[0] = &cam;
  return rig;
}

// cuVSLAM uses the OpenCV convention (x-right, y-down, z-forward), so visible landmarks
// have z > 0 in the camera frame. The cunls PnP factor enforces this with a z_threshold
// check; landmarks with non-positive z would be rejected (zero residual + zero Jacobian).
std::unordered_map<TrackId, Vector3T> GenerateLandmarks(const Isometry3T& world_from_cam, size_t count,
                                                        std::mt19937& rng) {
  std::unordered_map<TrackId, Vector3T> landmarks;
  std::uniform_real_distribution<float> xy_dist(-1.5f, 1.5f);
  std::uniform_real_distribution<float> z_dist(1.f, 5.f);

  for (size_t i = 0; i < count; i++) {
    Vector3T p_cam(xy_dist(rng), xy_dist(rng), z_dist(rng));
    landmarks[static_cast<TrackId>(i)] = world_from_cam * p_cam;
  }
  return landmarks;
}

std::vector<camera::Observation> ProjectLandmarks(const std::unordered_map<TrackId, Vector3T>& landmarks,
                                                  const Isometry3T& cam_from_world, CameraId cam_id = CameraId{0}) {
  std::vector<camera::Observation> observations;
  for (const auto& [id, lm_world] : landmarks) {
    Vector3T p_cam = cam_from_world * lm_world;
    // Skip points behind / on the camera plane (cuVSLAM uses positive-z-forward).
    if (p_cam.z() < 0.01f) {
      continue;
    }

    float inv_z = 1.f / p_cam.z();
    Vector2T xy(p_cam.x() * inv_z, p_cam.y() * inv_z);
    Matrix2T info = Matrix2T::Identity();
    observations.emplace_back(cam_id, id, xy, info);
  }
  return observations;
}

}  // namespace

// Recover a known pose from synthetic reprojection data (no depth).
TEST(MultisensorPoseEstimatorTest, ReprojectionConvergence) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  std::mt19937 rng(42);

  Isometry3T gt_cam_from_world = Isometry3T::Identity();
  gt_cam_from_world.translation() = Vector3T(0.f, 0.f, 0.f);

  auto landmarks = GenerateLandmarks(gt_cam_from_world.inverse(), 100, rng);
  auto observations = ProjectLandmarks(landmarks, gt_cam_from_world);
  ASSERT_GT(observations.size(), 10u);

  // Perturbed initial guess
  Vector6T perturbation;
  perturbation << 0.02f, -0.01f, 0.015f, 0.05f, -0.03f, 0.04f;
  Isometry3T delta;
  math::Exp(delta, perturbation);
  Isometry3T init_rig_from_world = gt_cam_from_world * delta;

  cuvslam::pnp::MultisensorSolverSettings settings;
  settings.max_iteration = 30;
  cuvslam::pnp::MultisensorPoseEstimator estimator(rig, settings);

  Isometry3T rig_from_world = init_rig_from_world;
  Matrix6T info;
  bool ok = estimator.solve(rig_from_world, info, observations, landmarks);
  ASSERT_TRUE(ok);

  Vector6T error;
  Isometry3T diff = gt_cam_from_world.inverse() * rig_from_world;
  math::Log(error, diff);
  float angular_err = error.head<3>().norm();
  float trans_err = error.tail<3>().norm();

  EXPECT_LT(angular_err, 0.01f) << "Angular error too large: " << angular_err;
  EXPECT_LT(trans_err, 0.02f) << "Translation error too large: " << trans_err;
}

// Verify solver runs without crashing on empty input.
TEST(MultisensorPoseEstimatorTest, EmptyObservations) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  cuvslam::pnp::MultisensorPoseEstimator estimator(rig);

  Isometry3T pose = Isometry3T::Identity();
  Matrix6T info;
  std::unordered_map<TrackId, Vector3T> landmarks;
  std::vector<camera::Observation> obs;

  bool ok = estimator.solve(pose, info, obs, landmarks);
  EXPECT_FALSE(ok);
}

// An inertial prior without preintegration contributes no factor, so with no observations the
// problem is empty. The solver has to report failure instead of handing cuNLS an empty problem.
TEST(MultisensorPoseEstimatorTest, InertialPriorWithoutResidualsFails) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);

  cuvslam::pnp::MultisensorPoseEstimator estimator(rig);

  Isometry3T pose = Isometry3T::Identity();
  Matrix6T info = Matrix6T::Identity();
  std::unordered_map<TrackId, Vector3T> landmarks;
  std::vector<camera::Observation> obs;
  cuvslam::pnp::InertialPriorInput imu_in;

  bool ok = true;
  EXPECT_NO_THROW(ok = estimator.solve(pose, info, obs, landmarks, {}, {}, {}, imu_in));
  EXPECT_FALSE(ok);
  EXPECT_TRUE(info.isIdentity());
}

// Multi-camera rig: verify that per-observation camera_from_rig extrinsics
// are correctly accounted for in the PnP factor.
TEST(MultisensorPoseEstimatorTest, MultiCameraReprojectionConvergence) {
  auto cam0 = MakePinhole();
  auto cam1 = MakePinhole();

  camera::Rig rig;
  rig.num_cameras = 2;
  rig.camera_from_rig[0] = Isometry3T::Identity();

  // Camera 1 is translated 0.5m to the right and rotated ~5 deg around Y.
  Vector6T extrinsic_twist;
  extrinsic_twist << 0.f, 0.09f, 0.f, 0.5f, 0.f, 0.f;
  Isometry3T cam1_from_rig;
  math::Exp(cam1_from_rig, extrinsic_twist);
  rig.camera_from_rig[1] = cam1_from_rig;

  rig.intrinsics[0] = &cam0;
  rig.intrinsics[1] = &cam1;

  std::mt19937 rng(123);

  Isometry3T gt_rig_from_world = Isometry3T::Identity();
  auto landmarks = GenerateLandmarks(gt_rig_from_world.inverse(), 150, rng);

  Isometry3T cam0_from_world = rig.camera_from_rig[0] * gt_rig_from_world;
  Isometry3T cam1_from_world = rig.camera_from_rig[1] * gt_rig_from_world;

  auto obs0 = ProjectLandmarks(landmarks, cam0_from_world, CameraId{0});
  auto obs1 = ProjectLandmarks(landmarks, cam1_from_world, CameraId{1});
  std::vector<camera::Observation> all_obs;
  all_obs.insert(all_obs.end(), obs0.begin(), obs0.end());
  all_obs.insert(all_obs.end(), obs1.begin(), obs1.end());
  ASSERT_GT(all_obs.size(), 20u);

  Vector6T perturbation;
  perturbation << 0.02f, -0.01f, 0.015f, 0.05f, -0.03f, 0.04f;
  Isometry3T delta;
  math::Exp(delta, perturbation);
  Isometry3T init_rig_from_world = gt_rig_from_world * delta;

  cuvslam::pnp::MultisensorSolverSettings settings;
  settings.max_iteration = 30;
  cuvslam::pnp::MultisensorPoseEstimator estimator(rig, settings);

  Isometry3T rig_from_world = init_rig_from_world;
  Matrix6T info;
  bool ok = estimator.solve(rig_from_world, info, all_obs, landmarks);
  ASSERT_TRUE(ok);

  Vector6T error;
  Isometry3T diff = gt_rig_from_world.inverse() * rig_from_world;
  math::Log(error, diff);
  float angular_err = error.head<3>().norm();
  float trans_err = error.tail<3>().norm();

  EXPECT_LT(angular_err, 0.01f) << "Angular error too large: " << angular_err;
  EXPECT_LT(trans_err, 0.02f) << "Translation error too large: " << trans_err;
}

// Appended inside namespace test::pnp; exercises real CUDA kernels.
TEST(MultisensorPoseEstimatorTest, BehindCameraCannotBeSuccessfulPnP) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);
  std::mt19937 rng(123);
  const auto landmarks = GenerateLandmarks(Isometry3T::Identity(), 50, rng);
  const auto observations = ProjectLandmarks(landmarks, Isometry3T::Identity());
  Isometry3T pose = Isometry3T::Identity();
  pose.linear() = Eigen::AngleAxisf(float(PI), Vector3T::UnitY()).toRotationMatrix();
  const auto original = pose;
  Matrix6T info = Matrix6T::Identity() * 37.f;
  cuvslam::pnp::MultisensorPoseEstimator estimator(rig);
  EXPECT_FALSE(estimator.solve(pose, info, observations, landmarks));
  EXPECT_TRUE((pose.matrix().array() == original.matrix().array()).all());
  EXPECT_TRUE((info.array() == (Matrix6T::Identity() * 37.f).array()).all());
}

TEST(MultisensorPoseEstimatorTest, InvalidInputDoesNotMutateOutputs) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);
  std::mt19937 rng(123);
  const auto landmarks = GenerateLandmarks(Isometry3T::Identity(), 50, rng);
  const auto observations = ProjectLandmarks(landmarks, Isometry3T::Identity());
  Isometry3T pose = Isometry3T::Identity();
  pose.translation().z() = std::numeric_limits<float>::infinity();
  Matrix6T info = Matrix6T::Identity() * 37.f;
  cuvslam::pnp::MultisensorPoseEstimator estimator(rig);
  EXPECT_FALSE(estimator.solve(pose, info, observations, landmarks));
  EXPECT_TRUE(std::isinf(pose.translation().z()));
  EXPECT_TRUE((info.array() == (Matrix6T::Identity() * 37.f).array()).all());
}

TEST(MultisensorPoseEstimatorTest, RealCudaPlusMatchesFourCameraJacobian) {
  Isometry3T pose = Isometry3T::Identity();
  pose.linear() = Eigen::AngleAxisf(.8f, Vector3T(1, -2, 3).normalized()).toRotationMatrix();
  pose.translation() = Vector3T(1.2f, -.7f, .8f);
  auto pack = [](const Isometry3T& m) {
    cunls::SE3Transform x;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) x[4 * i + j] = m.matrix()(i, j);
    return x;
  };
  std::vector<cunls::SE3Transform> ex;
  std::vector<cunls::Vector<3>> points;
  std::vector<cunls::Vector<2>> observations;
  std::vector<Vector2T> expected;
  for (int c = 0; c < 4; ++c) {
    Isometry3T e = Isometry3T::Identity();
    e.linear() = Eigen::AngleAxisf(.1f * c, Vector3T(1, 2, -1).normalized()).toRotationMatrix();
    e.translation() = Vector3T(.2f * c, -.08f * c, .04f * c);
    for (int j = 0; j < 5; ++j) {
      const Vector3T world = pose.inverse() * Vector3T(.2f * j - .4f, .1f * j - .2f, 4.f + .2f * j);
      const Vector3T pc = e * pose * world;
      ex.push_back(pack(e));
      points.push_back({world.x(), world.y(), world.z()});
      observations.push_back({0.f, 0.f});
      expected.push_back(pc.hnormalized());
    }
  }
  const auto packed = pack(pose);
  cunls::DeviceVector<float> state(std::vector<float>(packed.begin(), packed.end()));
  cunls::DeviceVector<cunls::SE3Transform> extrinsics(ex);
  cunls::DeviceVector<cunls::Vector<3>> landmarks(points);
  cunls::DeviceVector<cunls::Vector<2>> xy(observations);
  cunls::DeviceVector<float*> pointers(std::vector<float*>(points.size(), state.data()));
  cunls::DeviceVector<float> residual(2 * points.size()), jacobian(12 * points.size()), trial(16), delta(6);
  cunls::DeviceVector<float*> trial_ptrs(std::vector<float*>(points.size(), trial.data()));
  cunls::cuBLASHandle handle;
  cunls::SE3StateBatch states(handle, state.data(), 1);
  cunls::PnPFactorBatch factor(xy.data(), extrinsics.data(), landmarks.data(), points.size(), 1e-3f);
  ASSERT_TRUE(factor.Evaluate(residual.data(), jacobian.data(), pointers.data(), 0));
  std::vector<float> r(2 * points.size()), J(12 * points.size());
  ASSERT_EQ(cudaMemcpy(r.data(), residual.data(), r.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(J.data(), jacobian.data(), J.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < points.size(); ++i)
    for (int k = 0; k < 2; ++k) EXPECT_NEAR(r[2 * i + k], expected[i][k], 2e-6f);
  double worst = 0;
  for (int c = 0; c < 6; ++c) {
    std::vector<float> sides[2];
    for (int side = 0; side < 2; ++side) {
      float d[6]{};
      d[c] = (side == 0 ? 1.f : -1.f) * 2e-4f;
      ASSERT_EQ(cudaMemcpy(delta.data(), d, sizeof(d), cudaMemcpyHostToDevice), cudaSuccess);
      states.Plus(state.data(), delta.data(), trial.data(), 0);
      ASSERT_TRUE(factor.Evaluate(residual.data(), nullptr, trial_ptrs.data(), 0));
      sides[side].resize(r.size());
      ASSERT_EQ(cudaMemcpy(sides[side].data(), residual.data(), r.size() * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
    }
    for (size_t k = 0; k < r.size(); ++k)
      worst = std::max(worst, std::abs(double((sides[0][k] - sides[1][k]) / 4e-4f - J[6 * k + c])));
  }
  RecordProperty("cuda_plus_jacobian_max_abs_error", std::to_string(worst));
  EXPECT_LT(worst, .003);
}

TEST(MultisensorPoseEstimatorTest, CpuPnPNeedsMatchedVisibleObservations) {
  auto cam = MakePinhole();
  auto rig = MakeRig(cam);
  cuvslam::pnp::PNPSolver solver(rig);
  std::vector<camera::Observation> obs;
  std::unordered_map<TrackId, Vector3T> points;
  for (int i = 0; i < 20; ++i) obs.emplace_back(0, i, Vector2T(.1f, .1f), Matrix2T::Identity());
  points.emplace(999, Vector3T(0, 0, 3));
  Isometry3T pose = Isometry3T::Identity();
  Matrix6T info = Matrix6T::Identity() * 37.f;
  EXPECT_FALSE(solver.solve(pose, info, obs, points, cuvslam::pnp::PNPSettings{}));
  EXPECT_TRUE((info.array() == (Matrix6T::Identity() * 37.f).array()).all());
}

// Address protected base virtual slots; calls still dispatch to the actual LM override.
class MinimizerControlAccess : public cunls::GaussNewtonMinimizer {
public:
  static auto reject() { return &MinimizerControlAccess::RejectStep; }
  static auto accept() { return &MinimizerControlAccess::AcceptStep; }
};
TEST(MultisensorPoseEstimatorTest, NonfiniteLMStepQualityMustReject) {
  cunls::LevenbergMarquardtMinimizer lm;
  for (float q : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                  -std::numeric_limits<float>::infinity()}) {
    EXPECT_TRUE((lm.*MinimizerControlAccess::reject())(q));
    EXPECT_FALSE((lm.*MinimizerControlAccess::accept())(q));
  }
}
TEST(MultisensorPoseEstimatorTest, WhitenedCudaPlusMatchesFourCameraJacobian) {
  Isometry3T pose = Isometry3T::Identity();
  pose.linear() = Eigen::AngleAxisf(.8f, Vector3T(1, -2, 3).normalized()).toRotationMatrix();
  pose.translation() = Vector3T(1.2f, -.7f, .8f);
  auto pack = [](const Isometry3T& m) {
    cunls::SE3Transform x;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) x[4 * i + j] = m.matrix()(i, j);
    return x;
  };
  std::vector<cunls::SE3Transform> ex;
  std::vector<cunls::Vector<3>> points;
  std::vector<cunls::Vector<2>> observations;
  std::vector<Vector2T> expected;
  for (int c = 0; c < 4; ++c) {
    Isometry3T e = Isometry3T::Identity();
    e.linear() = Eigen::AngleAxisf(.1f * c, Vector3T(1, 2, -1).normalized()).toRotationMatrix();
    e.translation() = Vector3T(.2f * c, -.08f * c, .04f * c);
    for (int j = 0; j < 5; ++j) {
      const Vector3T world = pose.inverse() * Vector3T(.2f * j - .4f, .1f * j - .2f, 4.f + .2f * j);
      const Vector3T pc = e * pose * world;
      ex.push_back(pack(e));
      points.push_back({world.x(), world.y(), world.z()});
      observations.push_back({0.f, 0.f});
      expected.push_back(pc.hnormalized());
    }
  }
  const auto packed = pack(pose);
  cunls::DeviceVector<float> state(std::vector<float>(packed.begin(), packed.end()));
  cunls::DeviceVector<cunls::SE3Transform> extrinsics(ex);
  cunls::DeviceVector<cunls::Vector<3>> landmarks(points);
  cunls::DeviceVector<cunls::Vector<2>> xy(observations);
  cunls::DeviceVector<float*> pointers(std::vector<float*>(points.size(), state.data()));
  cunls::DeviceVector<float> residual(2 * points.size()), jacobian(12 * points.size()), trial(16), delta(6);
  cunls::DeviceVector<float*> trial_ptrs(std::vector<float*>(points.size(), trial.data()));
  cunls::cuBLASHandle handle;
  cunls::SE3StateBatch states(handle, state.data(), 1);
  cunls::PnPFactorBatch factor(xy.data(), extrinsics.data(), landmarks.data(), points.size(), 1e-3f);
  std::vector<float> roots(4*points.size());
  for(size_t i=0;i<points.size();++i){roots[4*i]=2.f;roots[4*i+1]=.3f;roots[4*i+2]=-.2f;roots[4*i+3]=1.1f;
    expected[i]=(Eigen::Matrix2f() << 2.f,.3f,-.2f,1.1f).finished()*expected[i];}
  cunls::DeviceVector<float> whitening(roots);factor.SetSqrtInformation(whitening.data());
  ASSERT_TRUE(factor.Evaluate(residual.data(), jacobian.data(), pointers.data(), 0));
  std::vector<float> r(2 * points.size()), J(12 * points.size());
  ASSERT_EQ(cudaMemcpy(r.data(), residual.data(), r.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(J.data(), jacobian.data(), J.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < points.size(); ++i)
    for (int k = 0; k < 2; ++k) EXPECT_NEAR(r[2 * i + k], expected[i][k], 2e-6f);
  double worst = 0;
  for (int c = 0; c < 6; ++c) {
    std::vector<float> sides[2];
    for (int side = 0; side < 2; ++side) {
      float d[6]{};
      d[c] = (side == 0 ? 1.f : -1.f) * 2e-4f;
      ASSERT_EQ(cudaMemcpy(delta.data(), d, sizeof(d), cudaMemcpyHostToDevice), cudaSuccess);
      states.Plus(state.data(), delta.data(), trial.data(), 0);
      ASSERT_TRUE(factor.Evaluate(residual.data(), nullptr, trial_ptrs.data(), 0));
      sides[side].resize(r.size());
      ASSERT_EQ(cudaMemcpy(sides[side].data(), residual.data(), r.size() * sizeof(float), cudaMemcpyDeviceToHost),
                cudaSuccess);
    }
    for (size_t k = 0; k < r.size(); ++k)
      worst = std::max(worst, std::abs(double((sides[0][k] - sides[1][k]) / 4e-4f - J[6 * k + c])));
  }
  RecordProperty("cuda_plus_jacobian_max_abs_error", std::to_string(worst));
  EXPECT_LT(worst, .006);
}


TEST(VisualSupportTest, UnrelatedOutliersDoNotDiluteObservableConsensus) {
 Eigen::MatrixXd j=Eigen::MatrixXd::Zero(20,6);j.topRows(6).setIdentity();j.bottomRows(6).setIdentity();
 EXPECT_TRUE(cuvslam::pnp::HasObservableVisualSupport(j,10,5));
 EXPECT_FALSE(cuvslam::pnp::HasObservableVisualSupport(j,10,11));
}
TEST(VisualSupportTest, DegenerateConsensusCannotSupportSixDofPose) {
 Eigen::MatrixXd j=Eigen::MatrixXd::Ones(200,6);
 EXPECT_FALSE(cuvslam::pnp::HasObservableVisualSupport(j,100,5));
 j(0,0)=std::numeric_limits<double>::quiet_NaN();
 EXPECT_FALSE(cuvslam::pnp::HasObservableVisualSupport(j,100,5));
}
}  // namespace test::pnp

#include <fstream>
#include <sstream>
#include <cstring>
TEST(FrozenBootstrapPnP, RepeatsSameInputWithoutChangingAcceptance) {
 using namespace cuvslam;
 const char* path=std::getenv("VIO_FROZEN_INPUT");
 if(!path) GTEST_SKIP()<<"Frozen input is supplied by the diagnostic harness";
 std::ifstream input(path);ASSERT_TRUE(input.good());long long timestamp;int accepted,ncam;input>>timestamp>>accepted>>ncam;
 auto read_pose=[&](){Isometry3T pose=Isometry3T::Identity();for(int r=0;r<4;++r)for(int c=0;c<4;++c)input>>pose.matrix()(r,c);return pose;};
 const auto initial=read_pose();const auto recorded=read_pose();
 camera::Rig rig;rig.num_cameras=ncam;
 for(int c=0;c<ncam;++c)rig.camera_from_rig[c]=read_pose();
 std::ifstream calib(std::string(path)+".intrinsics");ASSERT_TRUE(calib.good());
 std::vector<std::unique_ptr<camera::FisheyeCameraModel>> models;
 for(int c=0;c<ncam;++c){float w,h,fx,fy,cx,cy,k1,k2,k3,k4;calib>>w>>h>>fx>>fy>>cx>>cy>>k1>>k2>>k3>>k4;models.emplace_back(std::make_unique<camera::FisheyeCameraModel>(Vector2T(w,h),Vector2T(fx,fy),Vector2T(cx,cy),k1,k2,k3,k4));rig.intrinsics[c]=models.back().get();}
 std::unordered_map<TrackId,Vector3T> points;std::vector<camera::Observation> observations;
 std::string line;std::getline(input,line);
 while(std::getline(input,line)){std::istringstream row(line);TrackId id;int cam;float x,y,z,u,v,a,b,d,px,py;if(!(row>>id>>cam>>x>>y>>z>>u>>v>>a>>b>>d>>px>>py))continue;points[id]=Vector3T(x,y,z);Matrix2T info;info<<a,b,b,d;observations.emplace_back(CameraId(cam),id,Vector2T(u,v),info);}
 ASSERT_FALSE(observations.empty());pnp::MultisensorPoseEstimator solver(rig);std::optional<Isometry3T> first;
 for(int repeat=0;repeat<3;++repeat){auto pose=initial;Matrix6T info;const int mode=std::getenv("VIO_FROZEN_LOSS_MODE")?std::atoi(std::getenv("VIO_FROZEN_LOSS_MODE")):0;bool ok=solver.solve(pose,info,observations,points,{}, {}, {},std::nullopt,nullptr,mode);ASSERT_TRUE(solver.last_visual_trial());const auto& trial=*solver.last_visual_trial();
   std::ofstream output(std::string(path)+".mode"+std::to_string(mode)+".gpu"+std::to_string(repeat));output<<std::setprecision(17)<<int(ok)<<'\n';for(int r=0;r<4;++r){for(int c=0;c<4;++c)output<<trial.matrix()(r,c)<<' ';output<<'\n';}
   if(first){EXPECT_EQ(std::memcmp(first->matrix().data(),trial.matrix().data(),16*sizeof(float)),0);}else{first=trial;}
   if(!ok){EXPECT_EQ(std::memcmp(pose.matrix().data(),initial.matrix().data(),16*sizeof(float)),0)<<"Rejected trial modified caller pose";}
 }
}

TEST(VisualCauchy, GpuCostAndDerivativeAreConsistentForBothAdapters) {
 using namespace cuvslam;
 float* errors=nullptr;float3* values=nullptr;
 ASSERT_EQ(cudaMallocManaged(&errors,3*sizeof(float)),cudaSuccess);
 ASSERT_EQ(cudaMallocManaged(&values,3*sizeof(float3)),cudaSuccess);
 errors[0]=.1f;errors[1]=1.f;errors[2]=9.21f;
 for(int mode:{1,2}) {
  const auto p=pnp::VisualCauchyParameters(1.f,mode);
  EXPECT_NEAR(double(p.b)*p.c,1.,1e-6);
  cunls::CauchyLossFunctionBatch loss(p.b,p.c);ASSERT_TRUE(loss.Evaluate(errors,values,3,nullptr));ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  for(int i=0;i<3;++i) {
    const double x=errors[i],h=1e-5;
    const auto cost=[&](double s){return double(p.weight)*p.b*std::log1p(p.c*s);};
    const double numeric=(cost(x+h)-cost(x-h))/(2*h);
    EXPECT_NEAR(p.weight*values[i].y,numeric,1e-5);
    EXPECT_NEAR(p.weight*values[i].x,cost(x),1e-5);
  }
 }
 cudaFree(errors);cudaFree(values);
}
