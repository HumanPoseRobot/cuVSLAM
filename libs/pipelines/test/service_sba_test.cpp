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

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "common/include_gtest.h"
#include "map/map.h"
#include "pipelines/service_sba.h"

namespace test {
using namespace cuvslam;
using cuvslam::map::KeyFrame;
using cuvslam::map::Landmark;
using cuvslam::map::LandmarkPtr;
using cuvslam::map::UnifiedMap;
using cuvslam::pipelines::run_imu_sba;
using cuvslam::pipelines::run_sba;

constexpr size_t kNumKeyframes = 4;
constexpr size_t kLandmarksPerKeyframe = 600;

// Records the problem it is given and reports success, so the caller copies the solution back to the map.
struct RecordingSbaBundler {
  bool solve(sba::BundleAdjustmentProblem& problem) {
    points = problem.points;
    point_ids = problem.point_ids;
    return true;
  }
  std::vector<Vector3T> points;
  std::vector<int32_t> point_ids;
};

struct RecordingImuBundler {
  int iterations = -1;
  float robust_scale = -1;
  bool solve(sba_imu::ImuBAProblem& problem) {
    iterations = problem.max_iterations;
    robust_scale = problem.robustifier_scale;
    points = problem.points;
    point_ids = problem.point_ids;
    return true;
  }
  std::vector<Vector3T> points;
  std::vector<int32_t> point_ids;
};

struct SubMapFixture {
  UnifiedMap::SubMap submap;
  std::vector<LandmarkPtr> landmarks;
};

// Each keyframe observes its own landmarks once; has_pose(i) decides whether landmark i is triangulated.
template <class HasPose>
SubMapFixture MakeSubMap(HasPose has_pose) {
  SubMapFixture f;
  f.submap.landmark_and_obs.resize(kNumKeyframes);
  for (size_t k = 0; k < kNumKeyframes; k++) {
    auto kf = std::make_shared<KeyFrame>(map::State{}, static_cast<int64_t>(k));
    auto preint = k + 1 < kNumKeyframes ? std::make_shared<sba_imu::IMUPreintegration>() : nullptr;
    f.submap.consecutive_keyframes.push_back({kf, preint});
    for (size_t j = 0; j < kLandmarksPerKeyframe; j++) {
      const auto id = static_cast<TrackId>(f.landmarks.size());
      auto landmark = has_pose(id) ? std::make_shared<Landmark>(id, Vector3T(0, 0, 1)) : std::make_shared<Landmark>(id);
      UnifiedMap::LandmarkAndObserv lo{landmark, {}};
      lo.observations.try_push_back(camera::Observation(0, id, Vector2T::Zero(), Matrix2T::Identity()));
      f.submap.landmark_and_obs[k].try_push_back(lo);
      f.landmarks.push_back(landmark);
    }
  }
  return f;
}

template <class Bundler>
void ExpectConsistentProblem(const Bundler& bundler) {
  for (int32_t point_id : bundler.point_ids) {
    ASSERT_GE(point_id, 0);
    ASSERT_LT(static_cast<size_t>(point_id), bundler.points.size());
  }
}

// Triangulates landmarks from another thread, the way the tracking thread does (UnifiedMap::add_keyframe) while the
// SBA worker builds its problem from a submap that shares them. Until destroyed, it keeps clearing every pose and
// then setting them again in reverse order, so it is still running whenever the SBA passes are, and landmarks keep
// gaining a pose after the first pass skipped them.
class ConcurrentTriangulator {
public:
  explicit ConcurrentTriangulator(const std::vector<LandmarkPtr>& landmarks)
      : thread_([this, &landmarks] {
          running_.store(true, std::memory_order_release);
          while (!stop_.load(std::memory_order_acquire)) {
            for (const auto& landmark : landmarks) {
              landmark->reset();
            }
            for (auto it = landmarks.rbegin(); it != landmarks.rend(); ++it) {
              (*it)->set_pose(Vector3T(0, 0, 1));
            }
          }
        }) {
    while (!running_.load(std::memory_order_acquire)) {
    }
  }

  ~ConcurrentTriangulator() {
    stop_.store(true, std::memory_order_release);
    thread_.join();
  }

private:
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

TEST(ServiceSbaTest, SbaSkipsLandmarksWithoutPose) {
  auto f = MakeSubMap([](TrackId id) { return id % 2 == 0; });
  RecordingSbaBundler bundler;
  run_sba(f.submap, camera::Rig{}, sba::Settings{}, bundler);

  EXPECT_EQ(bundler.points.size(), f.landmarks.size() / 2);
  ExpectConsistentProblem(bundler);
  for (const auto& landmark : f.landmarks) {
    EXPECT_EQ(landmark->get_pose().has_value(), landmark->id() % 2 == 0) << "landmark " << landmark->id();
  }
}

TEST(ServiceSbaTest, ImuSbaSkipsLandmarksWithoutPose) {
  auto f = MakeSubMap([](TrackId id) { return id % 2 == 0; });
  RecordingImuBundler bundler;
  run_imu_sba(f.submap, Vector3T(0, 0, -9.81f), camera::Rig{}, imu::ImuCalibration{}, sba::Settings{}, bundler);

  EXPECT_EQ(bundler.points.size(), f.landmarks.size() / 2);
  ExpectConsistentProblem(bundler);
  for (const auto& landmark : f.landmarks) {
    EXPECT_EQ(landmark->get_pose().has_value(), landmark->id() % 2 == 0) << "landmark " << landmark->id();
  }
}

// Each trial races problem construction against triangulation; a landmark that gains its pose between the two
// passes over the submap must not be looked up in the first pass's index. The schedule is up to the OS, so a single
// trial can miss the window between the passes; many trials make missing it in all of them negligible.
constexpr int kRaceTrials = 1000;

TEST(ServiceSbaTest, SbaToleratesConcurrentTriangulation) {
  for (int trial = 0; trial < kRaceTrials; trial++) {
    auto f = MakeSubMap([](TrackId) { return false; });
    RecordingSbaBundler bundler;
    {
      ConcurrentTriangulator triangulator(f.landmarks);
      ASSERT_NO_THROW(run_sba(f.submap, camera::Rig{}, sba::Settings{}, bundler)) << "trial " << trial;
    }
    ExpectConsistentProblem(bundler);
  }
}

TEST(ServiceSbaTest, ImuSbaToleratesConcurrentTriangulation) {
  for (int trial = 0; trial < kRaceTrials; trial++) {
    auto f = MakeSubMap([](TrackId) { return false; });
    RecordingImuBundler bundler;
    {
      ConcurrentTriangulator triangulator(f.landmarks);
      ASSERT_NO_THROW(
          run_imu_sba(f.submap, Vector3T(0, 0, -9.81f), camera::Rig{}, imu::ImuCalibration{}, sba::Settings{}, bundler))
          << "trial " << trial;
    }
    ExpectConsistentProblem(bundler);
  }
}

TEST(ServiceSbaTest, ImuBackendReceivesExplicitSettings) {
  auto f = MakeSubMap([](TrackId) { return true; });
  RecordingImuBundler bundler;
  sba::Settings settings;
  settings.num_inertial_sba_iterations = 13;
  settings.inertial_robustifier_scale = .7f;
  run_imu_sba(f.submap, Vector3T(0, 0, -9.81f), camera::Rig{}, imu::ImuCalibration{}, settings, bundler);
  EXPECT_EQ(bundler.iterations, 13);
  EXPECT_FLOAT_EQ(bundler.robust_scale, .7f);
}
class RecordingSettingsService : public pipelines::SbaServiceBase {
public:
  explicit RecordingSettingsService(UnifiedMap& map) : SbaServiceBase(map, false) {}
  void service_task() override { observed = current_; }
  sba::Settings observed;
};
TEST(ServiceSbaTest, TriggerSnapshotsLatestSettings) {
  UnifiedMap map(100);
  RecordingSettingsService service(map);
  sba::Settings settings;
  settings.num_inertial_sba_frames = 20;
  service.trigger(settings);
  EXPECT_EQ(service.observed.num_inertial_sba_frames, 20);
  settings.num_inertial_sba_frames = 12;
  service.trigger(settings);
  EXPECT_EQ(service.observed.num_inertial_sba_frames, 12);
}
TEST(ServiceSbaTest, PreservesFourObservationsForNewAndExistingTrack) {
  UnifiedMap map(8);
  std::vector<camera::Observation> observations;
  for (int cam = 0; cam < 4; ++cam)
    observations.emplace_back(cam, 42, Vector2T::Zero(), Matrix2T::Identity());
  for (int t = 0; t < 2; ++t) {
    map.add_keyframe(t, map::State{}, sba_imu::IMUPreintegration{}, observations, {});
    const auto snapshot = map.get_recent_submap(1);
    const auto& saved = snapshot.landmark_and_obs.back().begin()->observations;
    ASSERT_EQ(saved.size(), 4u);
    for (int cam = 0; cam < 4; ++cam) EXPECT_EQ(saved.begin()[cam].cam_id, cam);
  }
}

struct UpdateCandidate {
  float offset;
  bool solve(sba_imu::ImuBAProblem& p) {
    p.rig_poses.back().w_from_imu.translation().x() = offset;
    for (auto& point : p.points) point.z() = 2.f;
    for (auto& pose : p.rig_poses) pose.velocity.y() = .5f;
    p.rig_poses.front().preintegration.dT_s = .125f;
    return true;
  }
};
TEST(ServiceSbaTest, ChecksMapUpdateBeforeWritingAnyState) {
  for (const float correction : {.3f, .002f}) {
    auto fixture = MakeSubMap([](TrackId) { return true; });
    UpdateCandidate solver{correction};
    const Matrix3T covariance = 1e-5f * Matrix3T::Identity();
    run_imu_sba(fixture.submap, Vector3T(0,0,-9.81f), camera::Rig{}, imu::ImuCalibration{},
                sba::Settings{}, solver, covariance);
    const bool accepted = correction < .01f;
    const auto& keys = fixture.submap.consecutive_keyframes;
    EXPECT_FLOAT_EQ(keys.back().keyframe->get_pose().inverse().translation().x(), accepted ? correction : 0.f);
    EXPECT_FLOAT_EQ(keys.front().keyframe->get_velocity().y(), accepted ? .5f : 0.f);
    EXPECT_FLOAT_EQ(keys.front().preintegration->dT_s, accepted ? .125f : 0.f);
    for (const auto& lm : fixture.landmarks)
      EXPECT_FLOAT_EQ(lm->get_pose()->z(), accepted ? 2.f : 1.f);
  }
}
}  // namespace test
