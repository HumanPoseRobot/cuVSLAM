#include "pipelines/triangulator.h"
#include "pipelines/triangulation_validation.h"
#include "imu/lm_model_reduction.h"
#include "pipelines/init_visual_admission.h"
#include "camera/camera.h"
#include "pipelines/imu_initialization_validation.h"
#include "pipelines/gpu_window/frontend_guard_adapter.h"
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

#include "pipelines/tracker_state_machine.h"
#include "common/include_gtest.h"

namespace cuvslam::pipelines {
namespace {

constexpr int64_t kFrameDt = static_cast<int64_t>(150e6);  // 150 ms KF spacing

// Test settings sized for short, deterministic tests: 5 KFs minimum, 500 ms minimum span.
// 5 KFs at 150 ms spacing -> 600 ms span (exceeds the 500 ms min_time_period_ns).
StateMachineSettings small_settings() {
  StateMachineSettings s;
  s.min_num_kf_for_gravity = 5;
  s.min_time_period_ns = static_cast<int64_t>(500e6);
  s.max_time_period_ns = static_cast<int64_t>(5e9);
  s.max_integration_time_ns = static_cast<int64_t>(1e9);
  // Effectively disable periodic re-estimation by default; individual tests override as needed.
  s.gravity_update_period_ns = static_cast<int64_t>(5e12);
  return s;
}

class CallbackCounter {
public:
  CallbackCounter(bool result) : result_(result) {}
  bool operator()(size_t /*num_kfs*/) {
    ++calls_;
    return result_;
  }
  int calls() const { return calls_; }
  void set_result(bool r) { result_ = r; }

private:
  bool result_;
  int calls_ = 0;
};

TEST(StateMachineTest, UninitializedToOkOnSuccessfulInit) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);
  int64_t ts = 0;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s),
              StateMachine::State::Uninitialized);
    ts += kFrameDt;
  }
  EXPECT_EQ(cb.calls(), 0);
  EXPECT_EQ(sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s), StateMachine::State::Ok);
  EXPECT_EQ(cb.calls(), 1);
}

TEST(StateMachineTest, FailedCallbackKeepsTimelineAndStaysInitializing) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/false);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Initializing);
  EXPECT_EQ(cb.calls(), 1);

  sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
  ts += kFrameDt;
  EXPECT_EQ(sm.state(), StateMachine::State::Initializing);
  EXPECT_EQ(cb.calls(), 2);

  cb.set_result(true);
  sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);
  EXPECT_EQ(cb.calls(), 3);
}

TEST(StateMachineTest, FrameFailedDuringInitClearsTimeline) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 4; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);

  sm.on_event(StateMachine::Event::FrameFailed, ts, /*is_keyframe=*/false, s);
  ts += kFrameDt;

  for (int i = 0; i < 4; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(cb.calls(), 0);
  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);

  sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
  EXPECT_EQ(cb.calls(), 1);
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);
}

TEST(StateMachineTest, FrameFailedAfterOkKeepsTimelineAndState) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);

  sm.on_event(StateMachine::Event::FrameFailed, ts, /*is_keyframe=*/false, s);
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);
  EXPECT_EQ(cb.calls(), 1);
}

TEST(StateMachineTest, FrameDroppedTimeoutResetsToUninitialized) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);

  ts += static_cast<int64_t>(2e9);  // 2s gap > max_integration_time_ns
  sm.on_event(StateMachine::Event::FrameDropped, ts, /*is_keyframe=*/false, s);
  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);
}

TEST(StateMachineTest, ResetEventDropsToUninitialized) {
  StateMachine sm;
  const StateMachineSettings s = small_settings();
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);

  sm.on_event(StateMachine::Event::Reset, ts, /*is_keyframe=*/false, s);
  EXPECT_EQ(sm.state(), StateMachine::State::Uninitialized);
}

TEST(StateMachineTest, PeriodicGravityRefinementWhenOk) {
  StateMachine sm;
  StateMachineSettings s = small_settings();
  s.gravity_update_period_ns = static_cast<int64_t>(1e9);  // refine every 1s while Ok
  CallbackCounter cb(/*result=*/true);
  sm.set_gravity_init_fn([&](size_t n) { return cb(n); });

  // Drive into Ok with the initial 5 KFs (600 ms elapsed).
  int64_t ts = 0;
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  ASSERT_EQ(sm.state(), StateMachine::State::Ok);
  ASSERT_EQ(cb.calls(), 1);

  // Within the refresh period: callback must NOT fire again.
  for (int i = 0; i < 5; ++i) {
    sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s);
    ts += kFrameDt;
  }
  EXPECT_EQ(cb.calls(), 1);

  // Advance > refresh period from the last successful estimation -> callback fires once.
  ts += static_cast<int64_t>(2e9);
  // Use a small frame gap for max_integration_time_ns to allow a 2s jump only via a Reset would
  // normally trigger; bump the SM forward via a successful keyframe at the new timestamp. We need
  // max_integration_time_ns large enough to not reset on this artificial jump.
  StateMachineSettings s2 = s;
  s2.max_integration_time_ns = static_cast<int64_t>(5e9);
  sm.on_event(StateMachine::Event::FrameOk, ts, /*is_keyframe=*/true, s2);
  EXPECT_EQ(sm.state(), StateMachine::State::Ok);
  EXPECT_EQ(cb.calls(), 2);
}

}  // namespace
}  // namespace cuvslam::pipelines

TEST(FisheyePixelGeometry, UsesOriginalImageUnitsAtLargeAngles) {
 using namespace cuvslam;
 camera::FisheyeCameraModel cam(Vector2T(512,512),Vector2T(191,191),Vector2T(256,256),0,0,0,0);
 const Vector2T prediction(174,120),observation(173,120);Vector2T actual,measured;
 ASSERT_TRUE(pipelines::frontend_guard::actual_pixel_residual(cam,prediction,observation,actual,&measured));
 Vector2T a,b;ASSERT_TRUE(cam.denormalizePoint(prediction,a));ASSERT_TRUE(cam.denormalizePoint(observation,b));
 EXPECT_LT((actual-(a-b)).norm(),1e-5f);EXPECT_LT(actual.norm(),2.f);
 EXPECT_GT((prediction-observation).cwiseProduct(cam.getFocal()).norm(),100.f);
}
TEST(FisheyePixelGeometry, PinholeRemainsEquivalent) {
 using namespace cuvslam;
 camera::PinholeCameraModel cam(Vector2T(512,512),Vector2T(191,192),Vector2T(256,256));
 const Vector2T prediction(1.5f,.8f),observation(1.4f,.9f);Vector2T actual;
 ASSERT_TRUE(pipelines::frontend_guard::actual_pixel_residual(cam,prediction,observation,actual));
 EXPECT_LT((actual-(prediction-observation).cwiseProduct(cam.getFocal())).norm(),1e-4f);
}
TEST(FisheyePixelGeometry, NonfiniteProjectionCannotSupportRecovery) {
 using namespace cuvslam;
 camera::FisheyeCameraModel cam(Vector2T(512,512),Vector2T(191,191),Vector2T(256,256),0,0,0,0);
 Vector2T actual;
 EXPECT_FALSE(pipelines::frontend_guard::actual_pixel_residual(cam,Vector2T(std::numeric_limits<float>::quiet_NaN(),1),Vector2T(0,0),actual));
}

TEST(VisualGyroSeed, NonidentityBodyDeltaIsRightMultiplied) {
 using namespace cuvslam;
 const Matrix3T previous=Eigen::AngleAxisf(.6f,Vector3T::UnitY()).toRotationMatrix();
 const Matrix3T delta=Eigen::AngleAxisf(.3f,Vector3T::UnitX()).toRotationMatrix();Matrix3T out;
 ASSERT_TRUE(pipelines::VisualGyroRotationSeed(previous,delta,true,out));
 EXPECT_LT((out-previous*delta).norm(),1e-6f);EXPECT_GT((out-delta*previous).norm(),.1f);
}
TEST(VisualGyroSeed, UnhealthyOrInvalidRotationCannotMutateSeed) {
 using namespace cuvslam;Matrix3T out=Matrix3T::Identity()*3.f;const auto before=out;
 EXPECT_FALSE(pipelines::VisualGyroRotationSeed(Matrix3T::Identity(),Matrix3T::Identity(),false,out));EXPECT_TRUE(out==before);
 Matrix3T invalid=Matrix3T::Identity();invalid(0,0)=std::numeric_limits<float>::infinity();
 EXPECT_FALSE(pipelines::VisualGyroRotationSeed(Matrix3T::Identity(),invalid,true,out));EXPECT_TRUE(out==before);
}


TEST(InitVisualAdmission, UsesIndependentProjectionRatherThanAbsoluteSpeedCap) {
 using namespace cuvslam;
 camera::PinholeCameraModel cam(Vector2T(640,480),Vector2T(400,400),Vector2T(320,240));
 camera::Rig rig;rig.num_cameras=1;rig.camera_from_rig[0]=Isometry3T::Identity();rig.intrinsics[0]=&cam;
 std::unordered_map<TrackId,Vector3T> points;std::vector<camera::Observation> obs;
 const Isometry3T ref=Isometry3T::Identity();
 for(int i=0;i<50;++i){Vector3T p((i%7-3)*.2f,(i%11-5)*.13f,3.f+(i%3)*.4f);points[i+1]=p;obs.emplace_back(0,i+1,p.hnormalized(),Matrix2T::Identity()*160000.f/9.f);}
 EXPECT_TRUE(pipelines::CheckInitPrediction(rig,points,obs,ref,ref,25).accepted);
 auto inconsistent=ref;inconsistent.translation().x()=.5f;
 EXPECT_FALSE(pipelines::CheckInitPrediction(rig,points,obs,ref,inconsistent,25).accepted);
 // A large world displacement is valid when the actual measurements agree.
 for(auto& [id,p]:points)p.x()+=10.f;
 auto shifted=ref;shifted.translation().x()=10.f;
 EXPECT_TRUE(pipelines::CheckInitPrediction(rig,points,obs,shifted,shifted,25).accepted);
 EXPECT_TRUE(pipelines::BootstrapRetentionBudget(true,true,true,100,150,50));
 EXPECT_FALSE(pipelines::BootstrapRetentionBudget(true,true,true,100,151,50));
 EXPECT_FALSE(pipelines::BootstrapRetentionBudget(false,true,true,100,150,50));
}

TEST(TriangulatorCache, RetainsEveryVisibleTrackIndependentOfHashIterationOrder) {
  using namespace cuvslam;
  camera::Rig rig;
  rig.num_cameras = 2;
  rig.camera_from_rig[0] = Isometry3T::Identity();
  rig.camera_from_rig[1] = Isometry3T::Identity();
  rig.camera_from_rig[1].translation().x() = -.2f;
  pipelines::MulticamTriangulator triangulator(rig);
  std::vector<camera::Observation> observations;
  for (uint64_t i = 1; i <= 64; ++i) {
    const TrackId id = (uint64_t(2) << 56) | (i * 7919);
    const Vector3T p((i%7)*.12f, (i%11)*.08f, 2.f + (i%13)*.1f);
    observations.emplace_back(0, id, p.hnormalized(), Matrix2T::Identity());
    observations.emplace_back(1, id, (rig.camera_from_rig[1]*p).hnormalized(), Matrix2T::Identity());
  }
  const auto first = triangulator.triangulate(Isometry3T::Identity(), observations);
  ASSERT_EQ(first.size(), 64u);
  for (int pass = 0; pass < 4; ++pass) {
    std::reverse(observations.begin(), observations.end());
    EXPECT_TRUE(triangulator.triangulate(Isometry3T::Identity(), observations).empty());
  }
  const auto removed_id = observations.front().id;
  auto retained = observations;
  retained.erase(std::remove_if(retained.begin(), retained.end(), [&](const auto& o) {
    return o.id == removed_id;
  }), retained.end());
  EXPECT_TRUE(triangulator.triangulate(Isometry3T::Identity(), retained).empty());
  const auto recreated = triangulator.triangulate(Isometry3T::Identity(), observations);
  ASSERT_EQ(recreated.size(), 1u);
  EXPECT_EQ(recreated.front().id, removed_id);
}
TEST(TriangulationGeometry, RejectsEpipolarCorrectionThatDoesNotExplainOriginalPixels) {
  using namespace cuvslam;
  camera::PinholeCameraModel camera(Vector2T(640,480),Vector2T(400,400),Vector2T(320,240));
  camera::Rig rig; rig.num_cameras=2;
  rig.intrinsics[0]=rig.intrinsics[1]=&camera;
  rig.camera_from_rig[0]=rig.camera_from_rig[1]=Isometry3T::Identity();
  rig.camera_from_rig[1].translation().x()=-.2f;
  const Vector3T point(.2f,.1f,3.f);
  std::vector<camera::Observation> o;
  for (int c=0;c<2;++c) {
    o.emplace_back(c,1,(rig.camera_from_rig[c]*point).hnormalized(),Matrix2T::Identity()*160000.f/9.f);
  }
  std::vector<std::reference_wrapper<const camera::Observation>> refs{std::cref(o[0]),std::cref(o[1])};
  EXPECT_TRUE(pipelines::TriangulationObservationsConsistent(rig,Isometry3T::Identity(),point,refs));
  o[1].xy.y()+=.1f;
  EXPECT_FALSE(pipelines::TriangulationObservationsConsistent(rig,Isometry3T::Identity(),point,refs));
  o[1].xy.y()-=.1f;
  EXPECT_FALSE(pipelines::TriangulationObservationsConsistent(rig,Isometry3T::Identity(),-point,refs));
}
TEST(InitializationLM, QuadraticModelUsesTheSameLowerTriangleAsTheSolve) {
  using namespace cuvslam;
  Eigen::MatrixXf h(3,3); h << 6,0,0, 2,7,0, -1,3,8;
  Eigen::VectorXf x(3),scale(3);x << .1f,-.2f,.3f;scale << 6,7,8;
  const Eigen::MatrixXf full=h.selfadjointView<Eigen::Lower>();
  const float expected=(x.dot(full*x)+.8f*x.dot(scale.asDiagonal()*x))/2.f;
  EXPECT_NEAR(sba_imu::InitPredictedReduction(h,x,scale,.4f,2.f),expected,1e-7f);
  EXPECT_GT(std::abs(expected-(x.dot(h*x)+.8f*x.dot(scale.asDiagonal()*x))/2.f),1e-3f);
  h(0,1)=1000.f;h(0,2)=-1000.f;
  EXPECT_NEAR(sba_imu::InitPredictedReduction(h,x,scale,.4f,2.f),expected,1e-7f);
}
