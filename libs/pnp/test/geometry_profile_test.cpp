#include "pnp/geometry_profile.h"
#include <gtest/gtest.h>
#include <Eigen/QR>
#include <random>

TEST(GeometryProfile, PointEliminationMatchesFullJointDampedQR) {
  using namespace cuvslam::pnp::geometry_profile_math;
  std::mt19937 random(76231);
  std::normal_distribution<double> normal;
  for (double lambda : {1e-6, 1e-3, 1., 100.}) {
    std::vector<Rows> blocks(3);
    int rows = 0;
    for (int k = 0; k < 3; ++k) {
      auto& b = blocks[k];
      const int count = 9 + k;
      b.pose.resize(count, 6);
      b.point.resize(count, k < 2 ? 3 : 0);
      b.residual.resize(count);
      for (int i = 0; i < b.pose.size(); ++i) {
        b.pose.data()[i] = normal(random);
      }
      for (int i = 0; i < b.point.size(); ++i) {
        b.point.data()[i] = normal(random);
      }
      for (int i = 0; i < count; ++i) {
        b.residual[i] = normal(random);
      }
      rows += count;
    }
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(rows, 12);
    Eigen::VectorXd r(rows);
    int at = 0;
    for (int k = 0; k < 3; ++k) {
      const auto& b = blocks[k];
      const int count = b.pose.rows();
      J.block(at, 0, count, 6) = b.pose;
      if (k < 2) {
        J.block(at, 6 + 3 * k, count, 3) = b.point;
      }
      r.segment(at, count) = b.residual;
      at += count;
    }
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows + 12, 12);
    A.topRows(rows) = J;
    A.bottomRows(12).diagonal() = (lambda * J.array().square().colwise().sum().max(1e-12)).sqrt().matrix();
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(rows + 12);
    rhs.head(rows) = -r;
    const Eigen::VectorXd reference = A.colPivHouseholderQr().solve(rhs);
    const auto actual = Solve(blocks, lambda);
    ASSERT_TRUE(actual.valid);
    EXPECT_LT((actual.pose - reference.head<6>()).norm(), 1e-11);
    for (int k = 0; k < 2; ++k) {
      EXPECT_LT((actual.points[k] - reference.segment<3>(6 + 3 * k)).norm(), 1e-11);
    }
    const Eigen::VectorXd dr = J * reference;
    const double predicted = -r.dot(dr) - .5 * dr.squaredNorm();
    EXPECT_NEAR(actual.predicted, predicted, 1e-10);
  }
}

#include "camera/camera.h"
#include "pnp/pixel_support.h"
TEST(PixelSupport, NativeFisheyeUsesActualPixelDistanceAtImagePeriphery) {
  using namespace cuvslam;
  camera::FisheyeCameraModel lens(Vector2T(1200, 1200), Vector2T(400, 400), Vector2T(600, 600), 0, 0, 0, 0);
  camera::Rig rig;
  rig.num_cameras = 1;
  rig.intrinsics[0] = &lens;
  rig.camera_from_rig[0] = Isometry3T::Identity();
  for (float angle : {0.f, .5f, 1.04719755f}) {
    const Vector3T p(std::tan(angle), 0, 1);
    const Vector2T measured(std::tan(angle + .015f), 0);
    camera::Observation o(0, 1, measured, Matrix2T::Identity());
    Vector2T a, b;
    ASSERT_TRUE(lens.denormalizePoint(p.hnormalized(), a));
    ASSERT_TRUE(lens.denormalizePoint(measured, b));
    EXPECT_NEAR((a - b).norm(), 6.f, .001f);
    EXPECT_TRUE(pnp::PixelObservationSupported(rig, o, p, .05f));
    if (angle > 1.f) {
      EXPECT_GT((p.hnormalized() - measured).cwiseProduct(lens.getFocal()).norm(), 8.f);
    }
    o.xy = Vector2T(std::tan(angle + .025f), 0);
    EXPECT_FALSE(pnp::PixelObservationSupported(rig, o, p, .05f));
    EXPECT_FALSE(pnp::PixelObservationSupported(rig, o, -p, .05f));
  }
}

TEST(GeometryProfile, StereoHistoryRefinesPoseWithoutWritingLandmarks) {
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
    pnp::GeometryProfile profile;
    solutions[ray] = initial;
    Matrix6T information = Matrix6T::Zero();
    ASSERT_TRUE(profile.refine(history, rig, points, obs, 5, ray, 3000000000LL, solutions[ray], &information));
    EXPECT_TRUE(information.allFinite());
    Eigen::SelfAdjointEigenSolver<Matrix6T> eig(information);
    ASSERT_EQ(eig.info(), Eigen::Success);
    EXPECT_GT(eig.eigenvalues().minCoeff(), 0.f);
    const auto committed = solutions[ray];
    const auto committed_info = information;
    ASSERT_FALSE(profile.refine(history, rig, points, {}, 5, ray, 3000000001LL, solutions[ray], &information));
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
