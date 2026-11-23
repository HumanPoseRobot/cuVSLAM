#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include "pnp/geometry_profile.h"
TEST(GeometryCacheAudit, ExistingKeyframePoseCorrectionMustInvalidatePrior) {
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
    const auto obs = observe(state.rig_from_w);
    if (k == 0) {
      state.rig_from_w.linear() = Eigen::AngleAxisf(.005f, Vector3T::UnitX()).toRotationMatrix();
    }
    history.add_keyframe(1000000000LL * k, state, {}, obs, k == 0 ? created : std::vector<pipelines::Landmark>{});
  }
  Isometry3T target = Isometry3T::Identity();
  target.translation() = Vector3T(-.25f, .01f, .02f);
  target.linear() = Eigen::AngleAxisf(.03f, Vector3T::UnitY()).toRotationMatrix();
  auto initial = target;
  initial.translation() += Vector3T(.002f, .001f, .005f);
  const auto observations = observe(target);
  pnp::GeometryProfile cached;
  auto warm = initial;
  Matrix6T warm_info;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000000LL, warm, &warm_info));
  const auto graph = history.get_recent_submap(4);
  const auto stamp = graph.consecutive_keyframes.back().keyframe->time_ns();
  graph.consecutive_keyframes.front().keyframe->set_pose(Isometry3T::Identity());
  ASSERT_EQ(history.get_recent_submap(4).consecutive_keyframes.back().keyframe->time_ns(), stamp);
  auto stale = initial, fresh = initial;
  Matrix6T stale_info, fresh_info;
  pnp::GeometryProfile rebuilt;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000001LL, stale, &stale_info));
  ASSERT_TRUE(rebuilt.refine(history, rig, points, observations, 5, false, 3000000001LL, fresh, &fresh_info));
  const double pose_difference = (stale.matrix() - fresh.matrix()).norm();
  const double center_difference = (stale.inverse().translation() - fresh.inverse().translation()).norm();
  const double info_relative = (stale_info - fresh_info).norm() / fresh_info.norm();
  cached.reset();
  auto reset_pose = initial;
  Matrix6T reset_info;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000001LL, reset_pose, &reset_info));
  const double reset_difference = (reset_pose.matrix() - fresh.matrix()).norm();
  const double reset_information_difference = (reset_info - fresh_info).norm();
  EXPECT_EQ(reset_difference, 0.);
  EXPECT_EQ(reset_information_difference, 0.);
  const double angle_deg =
      Eigen::AngleAxisf(stale.linear() * fresh.linear().transpose()).angle() * 180. / 3.141592653589793;
  std::ofstream out;
  if (const char* directory = std::getenv("VIO_GEOMETRY_TEST_RESULTS_DIR")) {
    out.open(std::string(directory) + "/CACHE_FIXED.json");
    ASSERT_TRUE(out.is_open());
  }
  out << std::setprecision(17)
      << "{\"same_latest_keyframe_timestamp\":true,\"pose_matrix_difference\":" << pose_difference
      << ",\"camera_center_difference_m\":" << center_difference
      << ",\"information_relative_difference\":" << info_relative << ",\"rotation_difference_deg\":" << angle_deg
      << ",\"explicit_reset_pose_difference\":" << reset_difference
      << ",\"explicit_reset_information_difference\":" << reset_information_difference
      << ",\"cached_error_to_truth\":" << (stale.matrix() - target.matrix()).norm()
      << ",\"fresh_error_to_truth\":" << (fresh.matrix() - target.matrix()).norm() << "}\n";
  EXPECT_LT(pose_difference, 1e-6) << "Same current map/input/initial pose must not depend on a stale prior cache";
  EXPECT_LT(info_relative, 1e-5);
}

#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include "pnp/geometry_profile.h"
TEST(GeometryCacheAudit, ExistingLandmarkCorrectionMustInvalidatePrior) {
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
    const auto obs = observe(state.rig_from_w);
    if (k == 0) {
      state.rig_from_w.linear() = Eigen::AngleAxisf(.005f, Vector3T::UnitX()).toRotationMatrix();
    }
    history.add_keyframe(1000000000LL * k, state, {}, obs, k == 0 ? created : std::vector<pipelines::Landmark>{});
  }
  Isometry3T target = Isometry3T::Identity();
  target.translation() = Vector3T(-.25f, .01f, .02f);
  target.linear() = Eigen::AngleAxisf(.03f, Vector3T::UnitY()).toRotationMatrix();
  auto initial = target;
  initial.translation() += Vector3T(.002f, .001f, .005f);
  const auto observations = observe(target);
  pnp::GeometryProfile cached;
  auto warm = initial;
  Matrix6T warm_info;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000000LL, warm, &warm_info));
  const auto graph = history.get_recent_submap(4);
  const auto stamp = graph.consecutive_keyframes.back().keyframe->time_ns();
  for (const auto& entry : graph.landmark_and_obs.front()) {
    auto point = entry.landmark->get_pose();
    ASSERT_TRUE(point.has_value());
    *point += Vector3T(.001f, -.001f, .002f);
    entry.landmark->set_pose(*point);
    points[entry.landmark->id()] = *point;
  }
  ASSERT_EQ(history.get_recent_submap(4).consecutive_keyframes.back().keyframe->time_ns(), stamp);
  auto stale = initial, fresh = initial;
  Matrix6T stale_info, fresh_info;
  pnp::GeometryProfile rebuilt;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000001LL, stale, &stale_info));
  ASSERT_TRUE(rebuilt.refine(history, rig, points, observations, 5, false, 3000000001LL, fresh, &fresh_info));
  const double pose_difference = (stale.matrix() - fresh.matrix()).norm();
  const double center_difference = (stale.inverse().translation() - fresh.inverse().translation()).norm();
  const double info_relative = (stale_info - fresh_info).norm() / fresh_info.norm();
  cached.reset();
  auto reset_pose = initial;
  Matrix6T reset_info;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000001LL, reset_pose, &reset_info));
  const double reset_difference = (reset_pose.matrix() - fresh.matrix()).norm();
  const double reset_information_difference = (reset_info - fresh_info).norm();
  EXPECT_EQ(reset_difference, 0.);
  EXPECT_EQ(reset_information_difference, 0.);
  const double angle_deg =
      Eigen::AngleAxisf(stale.linear() * fresh.linear().transpose()).angle() * 180. / 3.141592653589793;
  std::ofstream out;
  if (const char* directory = std::getenv("VIO_GEOMETRY_TEST_RESULTS_DIR")) {
    out.open(std::string(directory) + "/CACHE_LANDMARK_FIXED.json");
    ASSERT_TRUE(out.is_open());
  }
  out << std::setprecision(17)
      << "{\"same_latest_keyframe_timestamp\":true,\"pose_matrix_difference\":" << pose_difference
      << ",\"camera_center_difference_m\":" << center_difference
      << ",\"information_relative_difference\":" << info_relative << ",\"rotation_difference_deg\":" << angle_deg
      << ",\"explicit_reset_pose_difference\":" << reset_difference
      << ",\"explicit_reset_information_difference\":" << reset_information_difference
      << ",\"cached_error_to_truth\":" << (stale.matrix() - target.matrix()).norm()
      << ",\"fresh_error_to_truth\":" << (fresh.matrix() - target.matrix()).norm() << "}\n";
  EXPECT_LT(pose_difference, 1e-6) << "Same current map/input/initial pose must not depend on a stale prior cache";
  EXPECT_LT(info_relative, 1e-5);
}

#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include "pnp/geometry_profile.h"
TEST(GeometryCacheAudit, ExistingCalibrationCorrectionMustInvalidatePrior) {
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
    const auto obs = observe(state.rig_from_w);
    if (k == 0) {
      state.rig_from_w.linear() = Eigen::AngleAxisf(.005f, Vector3T::UnitX()).toRotationMatrix();
    }
    history.add_keyframe(1000000000LL * k, state, {}, obs, k == 0 ? created : std::vector<pipelines::Landmark>{});
  }
  Isometry3T target = Isometry3T::Identity();
  target.translation() = Vector3T(-.25f, .01f, .02f);
  target.linear() = Eigen::AngleAxisf(.03f, Vector3T::UnitY()).toRotationMatrix();
  auto initial = target;
  initial.translation() += Vector3T(.002f, .001f, .005f);
  const auto observations = observe(target);
  pnp::GeometryProfile cached;
  auto warm = initial;
  Matrix6T warm_info;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000000LL, warm, &warm_info));
  const auto graph = history.get_recent_submap(4);
  const auto stamp = graph.consecutive_keyframes.back().keyframe->time_ns();
  rig.camera_from_rig[1].translation().x() -= .002f;
  ASSERT_EQ(history.get_recent_submap(4).consecutive_keyframes.back().keyframe->time_ns(), stamp);
  auto stale = initial, fresh = initial;
  Matrix6T stale_info, fresh_info;
  pnp::GeometryProfile rebuilt;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000001LL, stale, &stale_info));
  ASSERT_TRUE(rebuilt.refine(history, rig, points, observations, 5, false, 3000000001LL, fresh, &fresh_info));
  const double pose_difference = (stale.matrix() - fresh.matrix()).norm();
  const double center_difference = (stale.inverse().translation() - fresh.inverse().translation()).norm();
  const double info_relative = (stale_info - fresh_info).norm() / fresh_info.norm();
  cached.reset();
  auto reset_pose = initial;
  Matrix6T reset_info;
  ASSERT_TRUE(cached.refine(history, rig, points, observations, 5, false, 3000000001LL, reset_pose, &reset_info));
  const double reset_difference = (reset_pose.matrix() - fresh.matrix()).norm();
  const double reset_information_difference = (reset_info - fresh_info).norm();
  EXPECT_EQ(reset_difference, 0.);
  EXPECT_EQ(reset_information_difference, 0.);
  const double angle_deg =
      Eigen::AngleAxisf(stale.linear() * fresh.linear().transpose()).angle() * 180. / 3.141592653589793;
  std::ofstream out;
  if (const char* directory = std::getenv("VIO_GEOMETRY_TEST_RESULTS_DIR")) {
    out.open(std::string(directory) + "/CACHE_RIG_FIXED.json");
    ASSERT_TRUE(out.is_open());
  }
  out << std::setprecision(17)
      << "{\"same_latest_keyframe_timestamp\":true,\"pose_matrix_difference\":" << pose_difference
      << ",\"camera_center_difference_m\":" << center_difference
      << ",\"information_relative_difference\":" << info_relative << ",\"rotation_difference_deg\":" << angle_deg
      << ",\"explicit_reset_pose_difference\":" << reset_difference
      << ",\"explicit_reset_information_difference\":" << reset_information_difference
      << ",\"cached_error_to_truth\":" << (stale.matrix() - target.matrix()).norm()
      << ",\"fresh_error_to_truth\":" << (fresh.matrix() - target.matrix()).norm() << "}\n";
  EXPECT_LT(pose_difference, 1e-6) << "Same current map/input/initial pose must not depend on a stale prior cache";
  EXPECT_LT(info_relative, 1e-5);
}

#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include "pnp/geometry_profile.h"
TEST(GeometryCacheAudit, PriorMeanControlSameMapInputAndInitial) {
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
    const auto obs = observe(state.rig_from_w);
    if (k == 0) {
      state.rig_from_w.linear() = Eigen::AngleAxisf(.005f, Vector3T::UnitX()).toRotationMatrix();
    }
    history.add_keyframe(1000000000LL * k, state, {}, obs, k == 0 ? created : std::vector<pipelines::Landmark>{});
  }
  Isometry3T target = Isometry3T::Identity();
  target.translation() = Vector3T(-.25f, .01f, .02f);
  target.linear() = Eigen::AngleAxisf(.03f, Vector3T::UnitY()).toRotationMatrix();
  auto initial = target;
  initial.translation() += Vector3T(.002f, .001f, .005f);
  const auto observations = observe(target);

  pnp::GeometryProfile profile;
  auto conditional = initial, centered = initial;
  Matrix6T conditional_info, centered_info;
  ASSERT_TRUE(profile.refineWithPriorMean(history, rig, points, observations, 5, false, 3000000000LL, conditional,
                                          &conditional_info, pnp::GeometryProfile::PriorMean::Conditional));
  ASSERT_TRUE(profile.refineWithPriorMean(history, rig, points, observations, 5, false, 3000000000LL, centered,
                                          &centered_info, pnp::GeometryProfile::PriorMean::PreserveMap));
  EXPECT_GT((conditional.matrix() - centered.matrix()).norm(), 1e-6f);
  const auto after = history.get_recent_landmarks();
  for (const auto& [id, point] : points) {
    EXPECT_EQ((after.at(id) - point).squaredNorm(), 0.f);
  }
  auto centered_repeat = initial;
  Matrix6T centered_repeat_info;
  ASSERT_TRUE(profile.refineWithPriorMean(history, rig, points, observations, 5, false, 3000000000LL, centered_repeat,
                                          &centered_repeat_info, pnp::GeometryProfile::PriorMean::PreserveMap));
  EXPECT_EQ((centered_repeat.matrix() - centered.matrix()).squaredNorm(), 0.f);
  EXPECT_EQ((centered_repeat_info - centered_info).squaredNorm(), 0.f);
  std::ofstream out;
  if (const char* directory = std::getenv("VIO_GEOMETRY_TEST_RESULTS_DIR")) {
    out.open(std::string(directory) + "/MEAN_CONTROL.json");
    ASSERT_TRUE(out.is_open());
  }
  out << std::setprecision(17) << "{\"conditional_to_centered_center_distance_m\":"
      << (conditional.inverse().translation() - centered.inverse().translation()).norm()
      << ",\"conditional_dz_world_m\":" << (conditional.inverse().translation() - initial.inverse().translation()).z()
      << ",\"centered_dz_world_m\":" << (centered.inverse().translation() - initial.inverse().translation()).z()
      << ",\"repeat_exact\":true,\"public_landmarks_unchanged\":true}\n";
}

TEST(GeometryMeanMath, CenteredPriorEqualsCorrelatedCovarianceMarginalization) {
  using namespace cuvslam::pnp::geometry_profile_math;
  Eigen::Matrix<double, 10, 6> A;
  Eigen::Matrix<double, 10, 3> B;
  Eigen::Matrix<double, 10, 1> residual;
  for (int row = 0; row < 10; ++row) {
    residual[row] = .1 * std::sin(.8 * row + .3);
    for (int c = 0; c < 6; ++c) {
      A(row, c) = std::sin((row + 1) * (c + 1) * .31);
    }
    for (int c = 0; c < 3; ++c) {
      B(row, c) = std::cos((row + 1) * (c + 1) * .41);
    }
  }
  Eigen::Matrix3d C = Eigen::Vector3d(.02, .1, .3).asDiagonal();
  Eigen::Matrix<double, 10, 10> S = Eigen::Matrix<double, 10, 10>::Identity() + B * C * B.transpose();
  const Eigen::Matrix<double, 6, 6> H = A.transpose() * S.ldlt().solve(A);
  const Eigen::Matrix<double, 6, 1> gradient = A.transpose() * S.ldlt().solve(residual);
  Rows b;
  b.pose = Eigen::MatrixXd::Zero(13, 6);
  b.pose.topRows(10) = A;
  b.point = Eigen::MatrixXd::Zero(13, 3);
  b.point.topRows(10) = B;
  b.point.bottomRows(3) = C.inverse().llt().matrixU();
  b.residual = Eigen::VectorXd::Zero(13);
  b.residual.head(10) = residual;
  const auto actual = Solve({b}, 0.);
  ASSERT_TRUE(actual.valid);
  EXPECT_LT((actual.pose + H.ldlt().solve(gradient)).norm(), 1e-11);
  EXPECT_GT(std::abs(S(0, 5)), 1e-6);  // Shared point creates cross-camera correlation.
  const Eigen::Vector3d base(1., 2., 3.), origin(.1, .3, .2);
  const Eigen::Matrix3d R = b.point.bottomRows(3);
  const Eigen::Vector3d offset = -R * (base - origin);
  EXPECT_LT((R * (base - origin) + offset).norm(), 1e-14);
}
