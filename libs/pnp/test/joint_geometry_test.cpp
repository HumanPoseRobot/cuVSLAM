#include "pnp/joint_geometry.h"
#include <gtest/gtest.h>
#include "camera/camera.h"
TEST(JointGeometry, StereoHistoryRefinesPoseWithoutWritingLandmarks) {
  using namespace cuvslam;
  camera::FisheyeCameraModel lens(Vector2T(1200, 1200), Vector2T(400, 400), Vector2T(600, 600), 0, 0, 0, 0);
  camera::Rig rig;
  rig.num_cameras = 2;
  for (int c = 0; c < 2; ++c) {
    rig.intrinsics[c] = &lens;
    rig.camera_from_rig[c] = Isometry3T::Identity();
  }
  rig.camera_from_rig[1].translation().x() = -.2f;
  map::UnifiedMap history(4);
  std::unordered_map<TrackId, Vector3T> points, truth;
  std::vector<pipelines::Landmark> created;
  for (int k = 0; k < 30; ++k) {
    const Vector3T p(.14f * ((k % 6) - 2), .13f * ((k / 6) - 2), 2.5f + .07f * k);
    truth[k + 1] = p;
    points[k + 1] = p + Vector3T(.0002f, -.0001f, .003f);
    created.push_back({TrackId(k + 1), points[k + 1]});
  }
  const auto observe = [&](const Isometry3T& T) {
    std::vector<camera::Observation> out;
    for (const auto& [id, p] : truth) {
      for (int c = 0; c < 2; ++c) {
        out.emplace_back(CameraId(c), id, (rig.camera_from_rig[c] * T * p).hnormalized(),
                         Matrix2T::Identity() * (400.f * 400.f / 9.f));
      }
    }
    return out;
  };
  for (int k = 0; k < 2; ++k) {
    map::State state;
    state.rig_from_w = Isometry3T::Identity();
    state.rig_from_w.translation().x() = -.1f * k;
    history.add_keyframe(1000000000LL * k, state, {}, observe(state.rig_from_w),
                         k == 0 ? created : std::vector<pipelines::Landmark>{});
  }
  Isometry3T target = Isometry3T::Identity();
  target.translation() = Vector3T(-.25f, .01f, .02f);
  target.linear() = Eigen::AngleAxisf(.03f, Vector3T::UnitY()).toRotationMatrix();
  auto initial = target;
  initial.translation() += Vector3T(.002f, .001f, .005f);
  const auto obs = observe(target);
  Isometry3T solutions[2];
  const char* previous = std::getenv("VIO_GEOMETRY_CHECK_JACOBIANS");
  const std::string saved = previous ? previous : "";
  setenv("VIO_GEOMETRY_CHECK_JACOBIANS", "1", 1);
  for (int ray = 0; ray < 2; ++ray) {
    pnp::JointGeometryReport report;
    solutions[ray] = initial;
    Matrix6T information = Matrix6T::Zero();
    ASSERT_TRUE(
        pnp::RefineJointGeometry(history, rig, points, obs, 5, 3000000000LL, solutions[ray], &information, &report));
    EXPECT_TRUE(information.allFinite());
    EXPECT_EQ(report.historical_poses, 2u);
    EXPECT_EQ(report.navigation_covariance.rows(), 12);
    EXPECT_GT(report.point_pose_covariance.norm(), 1e-8);
    Eigen::SelfAdjointEigenSolver<Matrix6T> eig(information);
    ASSERT_EQ(eig.info(), Eigen::Success);
    EXPECT_GT(eig.eigenvalues().minCoeff(), 0.f);
    const auto committed = solutions[ray];
    const auto committed_info = information;
    ASSERT_FALSE(pnp::RefineJointGeometry(history, rig, points, {}, 5, 3000000001LL, solutions[ray], &information));
    EXPECT_EQ((solutions[ray].matrix() - committed.matrix()).squaredNorm(), 0.f);
    EXPECT_EQ((information - committed_info).squaredNorm(), 0.f);
    EXPECT_LT((solutions[ray].matrix() - target.matrix()).norm(), .2f * (initial.matrix() - target.matrix()).norm());
    const auto map_points = history.get_recent_landmarks();
    for (const auto& [id, p] : points) {
      ASSERT_EQ((map_points.at(id) - p).squaredNorm(), 0.f);
    }
  }
  if (previous) {
    setenv("VIO_GEOMETRY_CHECK_JACOBIANS", saved.c_str(), 1);
  } else {
    unsetenv("VIO_GEOMETRY_CHECK_JACOBIANS");
  }
  EXPECT_LT((solutions[0].matrix() - solutions[1].matrix()).norm(), 1e-4f);
}

TEST(JointGeometry, NoHistoryDoesNotCreateFixedMapEvidence) {
  using namespace cuvslam;
  camera::FisheyeCameraModel lens(Vector2T(1200, 1200), Vector2T(400, 400), Vector2T(600, 600), 0, 0, 0, 0);
  camera::Rig rig;
  rig.num_cameras = 2;
  for (int c = 0; c < 2; ++c) {
    rig.intrinsics[c] = &lens;
    rig.camera_from_rig[c] = Isometry3T::Identity();
  }
  rig.camera_from_rig[1].translation().x() = -.2f;
  map::UnifiedMap history(4);
  std::unordered_map<TrackId, Vector3T> points, truth;
  std::vector<pipelines::Landmark> created;
  for (int k = 0; k < 30; ++k) {
    const Vector3T p(.14f * ((k % 6) - 2), .13f * ((k / 6) - 2), 2.5f + .07f * k);
    truth[k + 1] = p;
    points[k + 1] = p + Vector3T(.0002f, -.0001f, .003f);
    created.push_back({TrackId(k + 1), points[k + 1]});
  }
  const auto observe = [&](const Isometry3T& T) {
    std::vector<camera::Observation> out;
    for (const auto& [id, p] : truth) {
      for (int c = 0; c < 2; ++c) {
        out.emplace_back(CameraId(c), id, (rig.camera_from_rig[c] * T * p).hnormalized(),
                         Matrix2T::Identity() * (400.f * 400.f / 9.f));
      }
    }
    return out;
  };
  for (int k = 0; k < 2; ++k) {
    map::State state;
    state.rig_from_w = Isometry3T::Identity();
    state.rig_from_w.translation().x() = -.1f * k;
    history.add_keyframe(1000000000LL * k, state, {}, observe(state.rig_from_w),
                         k == 0 ? created : std::vector<pipelines::Landmark>{});
  }
  Isometry3T target = Isometry3T::Identity();
  target.translation() = Vector3T(-.25f, .01f, .02f);
  target.linear() = Eigen::AngleAxisf(.03f, Vector3T::UnitY()).toRotationMatrix();
  auto initial = target;
  initial.translation() += Vector3T(.002f, .001f, .005f);
  auto obs = observe(target);
  std::unordered_map<TrackId, Vector3T> unanchored;
  for (const auto& [id, point] : points) {
    unanchored.emplace(TrackId(id + 1000), point);
  }
  points = std::move(unanchored);
  for (auto& o : obs) {
    o.id = TrackId(o.id + 1000);
  }
  const auto map_before = history.get_recent_landmarks();
  auto candidate = initial;
  Matrix6T information = Matrix6T::Identity() * 7.f;
  const auto original_information = information;
  pnp::JointGeometryReport report;
  EXPECT_FALSE(pnp::RefineJointGeometry(history, rig, points, obs, 5, 3000000000LL, candidate, &information, &report));
  EXPECT_EQ((candidate.matrix() - initial.matrix()).squaredNorm(), 0.f);
  EXPECT_EQ((information - original_information).squaredNorm(), 0.f);
  EXPECT_EQ(report.missing_history, 30u);
  const auto map_after = history.get_recent_landmarks();
  for (const auto& [id, point] : map_before) {
    EXPECT_EQ((map_after.at(id) - point).squaredNorm(), 0.f);
  }
}

TEST(JointGeometry, AlreadyOptimalPoseStillReceivesJointInformation) {
  using namespace cuvslam;
  camera::FisheyeCameraModel lens(Vector2T(1200, 1200), Vector2T(400, 400), Vector2T(600, 600), 0, 0, 0, 0);
  camera::Rig rig;
  rig.num_cameras = 2;
  for (int c = 0; c < 2; ++c) {
    rig.intrinsics[c] = &lens;
    rig.camera_from_rig[c] = Isometry3T::Identity();
  }
  rig.camera_from_rig[1].translation().x() = -.2f;
  map::UnifiedMap history(4);
  std::unordered_map<TrackId, Vector3T> points, truth;
  std::vector<pipelines::Landmark> created;
  for (int k = 0; k < 30; ++k) {
    const Vector3T p(.14f * ((k % 6) - 2), .13f * ((k / 6) - 2), 2.5f + .07f * k);
    truth[k + 1] = p;
    points[k + 1] = p;
    created.push_back({TrackId(k + 1), points[k + 1]});
  }
  const auto observe = [&](const Isometry3T& T) {
    std::vector<camera::Observation> out;
    for (const auto& [id, p] : truth) {
      for (int c = 0; c < 2; ++c) {
        out.emplace_back(CameraId(c), id, (rig.camera_from_rig[c] * T * p).hnormalized(),
                         Matrix2T::Identity() * (400.f * 400.f / 9.f));
      }
    }
    return out;
  };
  for (int k = 0; k < 2; ++k) {
    map::State state;
    state.rig_from_w = Isometry3T::Identity();
    state.rig_from_w.translation().x() = -.1f * k;
    history.add_keyframe(1000000000LL * k, state, {}, observe(state.rig_from_w),
                         k == 0 ? created : std::vector<pipelines::Landmark>{});
  }
  Isometry3T target = Isometry3T::Identity();
  target.translation() = Vector3T(-.25f, .01f, .02f);
  target.linear() = Eigen::AngleAxisf(.03f, Vector3T::UnitY()).toRotationMatrix();
  auto initial = target;
  initial.translation() += Vector3T(.002f, .001f, .005f);
  const auto obs = observe(target);
  auto candidate = target;
  Matrix6T information = Matrix6T::Zero();
  ASSERT_TRUE(pnp::RefineJointGeometry(history, rig, points, obs, 5, 3000000000LL, candidate, &information));
  EXPECT_EQ((candidate.matrix() - target.matrix()).squaredNorm(), 0.f);
  EXPECT_GT(information.norm(), 0.f);
}

#include <memory>
#include <set>

#include <json/json.h>
#include <fstream>
TEST(JointGeometry, FrozenRejectedMeanRetainsAcceptedPoseAndJointInformation) {
  using namespace cuvslam;
  const char* path = std::getenv("VIO_JOINT_GEOMETRY_FROZEN_JSON");
  if (!path) {
    GTEST_SKIP() << "Set a frozen joint geometry input";
  }
  Json::Value data;
  std::ifstream file(path);
  ASSERT_TRUE(file.good());
  file >> data;
  std::vector<std::unique_ptr<camera::FisheyeCameraModel>> lenses;
  camera::Rig rig;
  rig.num_cameras = data["cameras"].size();
  for (int c = 0; c < rig.num_cameras; ++c) {
    const auto& spec = data["cameras"][c];
    const auto& k = spec["intrinsics"];
    const auto& d = spec["distortion_coeffs"];
    lenses.emplace_back(new camera::FisheyeCameraModel(
        Vector2T(spec["resolution"][0].asFloat(), spec["resolution"][1].asFloat()),
        Vector2T(k[0].asFloat(), k[1].asFloat()), Vector2T(k[2].asFloat(), k[3].asFloat()), d[0].asFloat(),
        d[1].asFloat(), d[2].asFloat(), d[3].asFloat()));
    rig.intrinsics[c] = lenses.back().get();
    rig.camera_from_rig[c] = Isometry3T::Identity();
    for (int r = 0; r < 4; ++r) {
      for (int j = 0; j < 4; ++j) {
        rig.camera_from_rig[c].matrix()(r, j) = data["camera_from_rig"][c][4 * r + j].asFloat();
      }
    }
  }
  const auto observation = [](const Json::Value& value) {
    Matrix2T information;
    for (int r = 0; r < 2; ++r) {
      for (int c = 0; c < 2; ++c) {
        information(r, c) = value["info"][2 * r + c].asFloat();
      }
    }
    return camera::Observation(CameraId(value["cam"].asInt()), TrackId(value["id"].asUInt64()),
                               Vector2T(value["xy"][0].asFloat(), value["xy"][1].asFloat()), information);
  };
  const size_t history_count = data["times"].size();
  map::UnifiedMap history(history_count + 2);
  std::set<TrackId> created_ids;
  for (size_t n = 0; n < history_count; ++n) {
    map::State state;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        state.rig_from_w.matrix()(r, c) = data["poses"][Json::ArrayIndex(n)][4 * r + c].asFloat();
      }
    }
    std::vector<camera::Observation> observations;
    std::vector<pipelines::Landmark> created;
    for (const auto& group : data["groups"]) {
      const TrackId id(group["id"].asUInt64());
      bool seen = false;
      for (const auto& item : group["observations"]) {
        if (item["node"].asUInt64() == n) {
          observations.push_back(observation(item["value"]));
          seen = true;
        }
      }
      if (seen && created_ids.insert(id).second) {
        created.push_back(
            {id, Vector3T(group["point"][0].asFloat(), group["point"][1].asFloat(), group["point"][2].asFloat())});
      }
    }
    history.add_keyframe(data["times"][Json::ArrayIndex(n)].asInt64(), state, {}, observations, created);
  }
  Isometry3T initial = Isometry3T::Identity();
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      initial.matrix()(r, c) = data["poses"][Json::ArrayIndex(history_count)][4 * r + c].asFloat();
    }
  }
  std::unordered_map<TrackId, Vector3T> points;
  for (const auto& group : data["landmarks"]) {
    points.emplace(TrackId(group["id"].asUInt64()),
                   Vector3T(group["point"][0].asFloat(), group["point"][1].asFloat(), group["point"][2].asFloat()));
  }
  std::vector<camera::Observation> observations;
  for (const auto& value : data["observations"]) {
    observations.push_back(observation(value));
  }
  const char* previous = std::getenv("VIO_GEOMETRY_DISABLE_COV_ONLY");
  const std::string saved = previous ? previous : "";
  setenv("VIO_GEOMETRY_DISABLE_COV_ONLY", "1", 1);
  auto control = initial;
  Matrix6T control_info = Matrix6T::Identity() * 7.f;
  EXPECT_FALSE(pnp::RefineJointGeometry(history, rig, points, observations, data["required"].asUInt64(),
                                        data["time_ns"].asInt64(), control, &control_info));
  EXPECT_EQ((control.matrix() - initial.matrix()).squaredNorm(), 0.f);
  unsetenv("VIO_GEOMETRY_DISABLE_COV_ONLY");
  auto candidate = initial;
  Matrix6T information = Matrix6T::Identity() * 7.f;
  pnp::JointGeometryReport report;
  const bool accepted = pnp::RefineJointGeometry(history, rig, points, observations, data["required"].asUInt64(),
                                                 data["time_ns"].asInt64(), candidate, &information, &report);
  if (previous) {
    setenv("VIO_GEOMETRY_DISABLE_COV_ONLY", saved.c_str(), 1);
  } else {
    unsetenv("VIO_GEOMETRY_DISABLE_COV_ONLY");
  }
  ASSERT_TRUE(accepted);
  EXPECT_TRUE(report.covariance_only);
  EXPECT_EQ((candidate.matrix() - initial.matrix()).squaredNorm(), 0.f);
  EXPECT_TRUE(information.allFinite());
  Eigen::SelfAdjointEigenSolver<Matrix6T> eigen(information);
  ASSERT_EQ(eigen.info(), Eigen::Success);
  EXPECT_GT(eigen.eigenvalues().minCoeff(), 0.f);
  EXPECT_GT((information - control_info).norm(), 1.f);
  EXPECT_TRUE(report.profile_gradient.allFinite());
}
