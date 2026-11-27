#include "joint_posterior.h"
#include "joint_initial_graph.h"
#include "joint_noise.h"
#include "joint_online.h"

#include <gtest/gtest.h>
#include <unistd.h>
#include <filesystem>
#include "camera/camera.h"
#include "camera/observation.h"

using namespace gpu_window::posterior;
namespace gaussian = cuvslam::pnp::joint_geometry;
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Graph Fixture(int count, int point_count = 10) {
  Graph graph;
  graph.gravity = Eigen::Vector3d(.12, -.23, -9.8).normalized() * 9.81;
  graph.huber = 20.;
  for (int i = 0; i < count; ++i) {
    graph.times.push_back(1000000000LL + i * 100000000LL);
    gpu_window::State state;
    const double w[3] = {.18 * std::sin(.7 * i), .12 * std::cos(.5 * i), .07 * i};
    gpu_vi::exp3(w, state.R);
    state.p[0] = .035 * i;
    state.p[1] = .04 * std::sin(.8 * i);
    state.p[2] = .03 * std::cos(.5 * i);
    state.v[0] = .35;
    state.v[1] = .32 * std::cos(.8 * i);
    state.v[2] = -.15 * std::sin(.5 * i);
    for (int k = 0; k < 3; ++k) {
      state.bg[k] = .001 * (k - 1);
      state.ba[k] = .01 * (k - 1);
    }
    graph.nodes.push_back(state);
  }
  graph.fixed_pose_time = graph.times.back();
  for (int point = 0; point < point_count; ++point) {
    graph.ids.push_back(point + 1);
    graph.points.emplace_back(.2 * (point % 5 - 2), .15 * (point / 5 - 1), 2.5 + .05 * point);
  }
  for (int node = 0; node < count; ++node) {
    const auto& state = graph.nodes[node];
    const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> rotation(state.R);
    const Eigen::Map<const Eigen::Vector3d> position(state.p);
    for (int point = 0; point < int(graph.points.size()); ++point) {
      for (int camera = 0; camera < 2; ++camera) {
        gpu_window::Visual visual;
        visual.node = node;
        visual.point = point;
        const double w[3] = {.025 * camera, -.03 * camera, .015 * camera};
        gpu_vi::exp3(w, visual.obs.Rci);
        visual.obs.tci[0] = -.15 * camera;
        visual.obs.tci[1] = .012 * camera;
        const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> camera_rotation(visual.obs.Rci);
        const Eigen::Vector3d pc = camera_rotation * rotation.transpose() * (graph.points[point] - position) +
                                   Eigen::Map<const Eigen::Vector3d>(visual.obs.tci);
        visual.obs.xy[0] = pc.x() / pc.z() + .0001 * std::sin(point + node);
        visual.obs.xy[1] = pc.y() / pc.z() - .0001 * std::cos(point + node);
        visual.obs.sqrt_info[0] = visual.obs.sqrt_info[3] = 100.;
        visual.obs.min_depth = .05;
        graph.visuals.push_back(visual);
        graph.visual_cameras.push_back(camera);
      }
    }
    if (node) {
      gpu_window::Edge edge;
      edge.left = node - 1;
      edge.right = node;
      auto& data = edge.data;
      const auto& left = graph.nodes[node - 1];
      const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> lr(left.R);
      const double dt = .1;
      CopyRows(data.dR, (lr.transpose() * rotation).eval());
      const Eigen::Vector3d dv = lr.transpose() * (Eigen::Map<const Eigen::Vector3d>(state.v) -
                                                   Eigen::Map<const Eigen::Vector3d>(left.v) - dt * graph.gravity);
      const Eigen::Vector3d dp =
          lr.transpose() * (position - Eigen::Map<const Eigen::Vector3d>(left.p) -
                            dt * Eigen::Map<const Eigen::Vector3d>(left.v) - .5 * dt * dt * graph.gravity);
      CopyRows(data.dV, dv);
      CopyRows(data.dP, dp);
      CopyRows(data.JRg, Eigen::Matrix3d(-dt * Eigen::Matrix3d::Identity()));
      CopyRows(data.JVa, Eigen::Matrix3d(-dt * Eigen::Matrix3d::Identity()));
      CopyRows(data.JPa, Eigen::Matrix3d(-.5 * dt * dt * Eigen::Matrix3d::Identity()));
      std::copy_n(left.bg, 3, data.bg0);
      std::copy_n(left.ba, 3, data.ba0);
      for (int k = 0; k < 9; ++k) {
        data.imu_sqrt[9 * k + k] = k < 3 ? 1000. : 200.;
      }
      for (int k = 0; k < 6; ++k) {
        edge.rw[6 * k + k] = k < 3 ? 500. : 50.;
      }
      data.dt = dt;
      graph.edges.push_back(edge);
    }
  }
  // Genuine nonzero residuals: a posterior must carry the resulting RHS.
  graph.points[0].x() += .002;
  graph.nodes[0].v[2] += .003;
  graph.gravity = RetractGravity(graph.gravity, Eigen::Vector2d(.004, -.003));
  return graph;
}

Graph FromPrior(const Prior& prior) {
  Graph graph;
  graph.prior = prior;
  graph.times = prior.times;
  graph.ids = prior.ids;
  graph.nodes = prior.nodes;
  graph.points = prior.points;
  graph.gravity = prior.gravity;
  graph.fixed_pose_time = prior.fixed_pose_time;
  graph.huber = 20.;
  return graph;
}

std::vector<int> KeepColumns(const Graph& graph, const std::vector<int64_t>& times, const std::vector<int64_t>& ids) {
  std::vector<int> kept;
  for (auto time : times) {
    const int node = std::find(graph.times.begin(), graph.times.end(), time) - graph.times.begin();
    for (int k = 0; k < 15; ++k) {
      kept.push_back(15 * node + k);
    }
  }
  for (auto id : ids) {
    const int point = std::find(graph.ids.begin(), graph.ids.end(), id) - graph.ids.begin();
    for (int k = 0; k < 3; ++k) {
      kept.push_back(15 * graph.nodes.size() + 3 * point + k);
    }
  }
  kept.push_back(15 * graph.nodes.size() + 3 * graph.points.size());
  kept.push_back(15 * graph.nodes.size() + 3 * graph.points.size() + 1);
  return kept;
}
}  // namespace

TEST(JointPosterior, GravityChartAndJacobian) {
  for (const Eigen::Vector3d& anchor : {Eigen::Vector3d(0., 0., -9.81), Eigen::Vector3d(.4, -.7, 9.7)}) {
    const Eigen::Vector2d step(.12, -.09);
    const auto value = RetractGravity(anchor, step);
    EXPECT_NEAR(value.norm(), anchor.norm(), 1e-13);
    EXPECT_LT((GravityDifference(value, anchor) - step).norm(), 1e-13);
    Eigen::Matrix2d numerical;
    for (int k = 0; k < 2; ++k) {
      const Eigen::Vector2d d = 1e-6 * Eigen::Vector2d::Unit(k);
      numerical.col(k) =
          (GravityDifference(RetractGravity(value, d), anchor) - GravityDifference(RetractGravity(value, -d), anchor)) /
          2e-6;
    }
    EXPECT_LT((numerical - GravityDifferenceJacobian(value, anchor)).norm(), 1e-8);
  }
}

TEST(JointPosterior, RawVisualImuAndGravityGradientMatchesRetraction) {
  const auto graph = Fixture(3);
  const auto linear = Linearize(graph, false);
  const auto factor = Assemble(graph, linear);
  const Vector gradient = factor.rows.transpose() * factor.residual;
  double worst = 0.;
  for (int k = 0; k < gradient.size(); ++k) {
    Vector step = Vector::Zero(gradient.size());
    step[k] = 1e-6;
    const double actual =
        (Linearize(Retract(graph, step), false).cost - Linearize(Retract(graph, -step), false).cost) / 2e-6;
    worst = std::max(worst, std::abs(actual - gradient[k]) / std::max({1., std::abs(actual), std::abs(gradient[k])}));
  }
  EXPECT_LT(worst, 2e-6);
}

TEST(JointPosterior, InitializationTransferMatchesBatchIncludingNonzeroRhsAndConstant) {
  const auto graph = Fixture(8);  // Deliberately exceeds the old six-slot limit.
  const auto linear = Linearize(graph, false);
  const auto raw = Assemble(graph, linear);
  const std::vector<int64_t> times{graph.times.back()}, ids{1, 3, 6, 9};
  const auto prior =
      Marginalize(graph, linear, times, ids, graph.times[graph.times.size() - 2], graph.times[graph.times.size() - 2]);
  const auto expected = gaussian::eliminate(raw, KeepColumns(graph, times, ids));
  const auto stored = FromPrior(prior);
  const auto actual = Assemble(stored, Linearize(stored, false));
  const Matrix information = expected.rows.transpose() * expected.rows;
  const Vector gradient = expected.rows.transpose() * expected.residual;
  EXPECT_GT(gradient.norm(), 1e-5);
  EXPECT_LT((actual.rows.transpose() * actual.rows - information).norm() / information.norm(), 1e-9);
  EXPECT_LT((actual.rows.transpose() * actual.residual - gradient).norm() / gradient.norm(), 1e-9);
  EXPECT_NEAR(actual.residual.squaredNorm(), expected.residual.squaredNorm(), 1e-9);
  for (int trial = 0; trial < 3; ++trial) {
    Vector delta(actual.rows.cols());
    for (int k = 0; k < delta.size(); ++k) {
      delta[k] = 1e-3 * std::sin(.31 * (k + trial));
    }
    EXPECT_NEAR((actual.rows * delta + actual.residual).squaredNorm(),
                (expected.rows * delta + expected.residual).squaredNorm(), 1e-8);
  }
  // The actual retained prior contains gravity/nav/point cross-information.
  EXPECT_GT(information.topRightCorner(15, 2).norm(), 1e-3);
  EXPECT_GT(information.block(6, 15, 9, 12).norm(), 1e-3);
}

TEST(JointPosterior, RetainedGravityPropagationMatchesBatchOnNextFrame) {
  auto batch = Fixture(9);
  // Freeze the root at the last initialization frame, not the new image.
  batch.fixed_pose_time = batch.times[7];
  auto history = batch;
  history.nodes.pop_back();
  history.times.pop_back();
  history.edges.pop_back();
  while (!history.visuals.empty() && history.visuals.back().node == 8) {
    history.visuals.pop_back();
    history.visual_cameras.pop_back();
  }
  const auto prior = Marginalize(history, Linearize(history, false), {history.times.back()}, history.ids,
                                 history.times[6], history.times[6]);
  auto live = FromPrior(prior);
  live.nodes.push_back(batch.nodes.back());
  live.times.push_back(batch.times.back());
  auto edge = batch.edges.back();
  edge.left = 0;
  edge.right = 1;
  live.edges.push_back(edge);
  for (size_t k = 0; k < batch.visuals.size(); ++k) {
    auto v = batch.visuals[k];
    if (v.node == 8) {
      v.node = 1;
      live.visuals.push_back(v);
      live.visual_cameras.push_back(batch.visual_cameras[k]);
    }
  }
  const auto old = Assemble(batch, Linearize(batch, false));
  const auto now = Assemble(live, Linearize(live, false));
  const std::vector<int64_t> kept_time{batch.times.back()};
  const auto a = gaussian::eliminate(old, KeepColumns(batch, kept_time, {}));
  const auto b = gaussian::eliminate(now, KeepColumns(live, kept_time, {}));
  const Matrix h = a.rows.transpose() * a.rows;
  EXPECT_LT((b.rows.transpose() * b.rows - h).norm() / h.norm(), 1e-9);
  EXPECT_LT((b.rows.transpose() * b.residual - a.rows.transpose() * a.residual).norm(), 1e-7);
  // Silently fixing gravity changes this posterior even at the same mean.
  auto fixed = now;
  fixed.rows.rightCols<2>().setZero();
  std::vector<int> current;
  for (int k = 15; k < 30; ++k) {
    current.push_back(k);
  }
  const auto full = gaussian::marginal(now, current);
  const auto wrong = gaussian::marginal(fixed, current);
  ASSERT_TRUE(full.observable);
  ASSERT_TRUE(wrong.observable);
  EXPECT_GT((full.covariance - wrong.covariance).norm(), 1e-8);
}

TEST(JointPosterior, StoredPriorGradientRemainsCorrectAfterChangingStateAndGravity) {
  const auto graph = Fixture(8);
  auto prior =
      Marginalize(graph, Linearize(graph, false), {graph.times.back()}, {1, 2, 3, 4}, graph.times[6], graph.times[6]);
  auto live = FromPrior(prior);
  Vector step = Vector::Zero(15 + 12 + 2);
  for (int k = 6; k < step.size(); ++k) {
    step[k] = 1e-4 * std::sin(.27 * k);
  }
  live = Retract(live, step);
  const auto factor = Assemble(live, Linearize(live, false));
  const Vector gradient = factor.rows.transpose() * factor.residual;
  for (int k = 6; k < gradient.size(); ++k) {
    Vector delta = Vector::Zero(gradient.size());
    delta[k] = 1e-6;
    const double actual =
        (Linearize(Retract(live, delta), false).cost - Linearize(Retract(live, -delta), false).cost) / 2e-6;
    EXPECT_NEAR(actual, gradient[k], 2e-6 * std::max(1., std::abs(actual)));
  }
  live.visuals.push_back(graph.visuals.front());
  live.visual_cameras.push_back(graph.visual_cameras.front());
  live.visuals.back().node = 0;
  EXPECT_THROW(Linearize(live, false), std::invalid_argument);
}

TEST(JointPosterior, IndependentPointOptimizationPreservesGaugeAndLowersActualCost) {
  auto graph = Fixture(8);
  const auto root = graph.nodes.back();
  graph.iterations = 8;
  const auto report = Optimize(graph);
  ASSERT_TRUE(report.success);
  EXPECT_LT(report.after, report.before);
  for (int k = 0; k < 9; ++k) {
    EXPECT_DOUBLE_EQ(graph.nodes.back().R[k], root.R[k]);
  }
  for (int k = 0; k < 3; ++k) {
    EXPECT_DOUBLE_EQ(graph.nodes.back().p[k], root.p[k]);
  }
  EXPECT_LT(report.max_linear_residual, 1e-8);
}

TEST(JointPosteriorNoise, PixelInformationAlreadyContainsThreePixelVariance) {
  const auto pixel = cuvslam::camera::GetDefaultObservationInfoUV();
  EXPECT_NEAR(pixel(0, 0), 1. / 9., 1e-8);
  const auto whiten = InformationWhitening<2>(pixel.cast<double>());
  EXPECT_NEAR((whiten * Eigen::Vector2d(3., 0.)).squaredNorm(), 1., 1e-7);
  EXPECT_NEAR((InformationWhitening<2>(9. * pixel.cast<double>()) * Eigen::Vector2d(3., 0.)).squaredNorm(), 9., 1e-6);
}

TEST(JointPosteriorNoise, ImuWhiteningUsesPropagatedCalibrationWithoutFrequencyMultiplierTwice) {
  using namespace cuvslam;
  for (float frequency : {100.f, 200.f}) {
    imu::ImuCalibration calibration(Isometry3T::Identity(), .0001f, .00001f, .002f, .003f, frequency);
    imu::ImuMeasurementStorage measurements(1000);
    for (int i = 0; i <= int(frequency); ++i) {
      measurements.push_back(imu::ImuMeasurement{1000000000LL + int64_t(i) * int64_t(1e9 / frequency),
                                                 Vector3T(0, 0, 9.81f), Vector3T::Zero()});
    }
    sba_imu::IMUPreintegration integration(calibration, measurements, Vector3T::Zero(), Vector3T::Zero(), 1000000000LL,
                                           2000000000LL);
    const auto edge = CalibratedEdge(integration, calibration, 0, 1);
    const Eigen::Map<const Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> w(edge.data.imu_sqrt);
    const Eigen::Matrix<double, 9, 9> normalized = w * StoredSymmetric(integration.cov_matrix_) * w.transpose();
    EXPECT_LT((normalized - Eigen::Matrix<double, 9, 9>::Identity()).norm(), 1e-10);
    EXPECT_NEAR(integration.cov_matrix_(0, 0), .0001 * .0001, 1e-13);
    const Eigen::Map<const Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> rw(edge.rw);
    Eigen::Matrix<double, 6, 6> covariance = Eigen::Matrix<double, 6, 6>::Zero();
    covariance.topLeftCorner<3, 3>() = StoredSymmetric(integration.gyro_random_walk_accum_cov_matrix_);
    covariance.bottomRightCorner<3, 3>() = StoredSymmetric(integration.acc_random_walk_accum_cov_matrix_);
    EXPECT_LT((rw * covariance * rw.transpose() - Eigen::Matrix<double, 6, 6>::Identity()).norm(), 1e-12);
    const Eigen::Matrix3d expected = calibration.gyro_random_walk().cast<double>() * frequency;
    const double roundoff = 8. * integration.size() * std::numeric_limits<float>::epsilon();
    EXPECT_LT((covariance.topLeftCorner<3, 3>() - expected).norm() / expected.norm(), roundoff);
  }
}

TEST(JointPosteriorNoise, InvalidCovarianceIsNotRepairedIntoAConfidentPrior) {
  Eigen::Matrix2d bad = Eigen::Matrix2d::Identity();
  bad(0, 1) = 1.;
  EXPECT_THROW(CovarianceWhitening<2>(bad), std::invalid_argument);
  bad = -Eigen::Matrix2d::Identity();
  EXPECT_THROW(CovarianceWhitening<2>(bad), std::runtime_error);
}

TEST(JointPosterior, RobustResidualConstantEnergySurvivesTransfer) {
  auto graph = Fixture(8);
  graph.huber = .5;
  graph.visuals.front().obs.xy[0] += .03;
  const auto linear = Linearize(graph, false);
  EXPECT_GT(linear.constant, 0.);
  const auto factor = Assemble(graph, linear);
  EXPECT_NEAR(.5 * factor.residual.squaredNorm(), linear.cost, 1e-10);
  const auto prior = Marginalize(graph, linear, {graph.times.back()}, {1, 3, 6}, graph.times[6], graph.times[6]);
  const auto raw = gaussian::eliminate(factor, KeepColumns(graph, prior.times, prior.ids));
  const auto live = FromPrior(prior);
  const auto stored = Assemble(live, Linearize(live, false));
  EXPECT_NEAR(stored.residual.squaredNorm(), raw.residual.squaredNorm(), 1e-9);
}

TEST(JointPosterior, DuplicateRawFactorsCannotIncreaseConfidence) {
  auto graph = Fixture(4);
  graph.visuals.push_back(graph.visuals.front());
  graph.visual_cameras.push_back(graph.visual_cameras.front());
  EXPECT_THROW(Linearize(graph, false), std::invalid_argument);
  graph = Fixture(4);
  graph.edges.push_back(graph.edges.front());
  EXPECT_THROW(Linearize(graph, false), std::invalid_argument);
}

TEST(JointPosterior, NewBirthEvidenceAtRetainedFrameIsNotMistakenForConsumedEvidence) {
  const auto batch = Fixture(8);
  auto history = batch;
  const auto missing = std::find_if(history.visuals.begin(), history.visuals.end(), [](const auto& visual) {
    return visual.node == 7 && visual.point == 0 && visual.obs.tci[0] < 0.;
  });
  ASSERT_NE(missing, history.visuals.end());
  const int index = missing - history.visuals.begin();
  const auto visual = *missing;
  const int camera = history.visual_cameras[index];
  history.visuals.erase(missing);
  history.visual_cameras.erase(history.visual_cameras.begin() + index);
  const auto prior = Marginalize(history, Linearize(history, false), {history.times.back()}, history.ids,
                                 history.times[6], history.times[6]);
  auto live = FromPrior(prior);
  live.visuals.push_back(visual);
  live.visuals.back().node = 0;
  live.visual_cameras.push_back(camera);
  ASSERT_NO_THROW(Linearize(live, false));
  const auto actual = Assemble(live, Linearize(live, false));
  const auto expected =
      gaussian::eliminate(Assemble(batch, Linearize(batch, false)), KeepColumns(batch, prior.times, prior.ids));
  EXPECT_LT((actual.rows.transpose() * actual.rows - expected.rows.transpose() * expected.rows).norm() /
                (expected.rows.transpose() * expected.rows).norm(),
            1e-9);
}

TEST(JointPosterior, FrozenSnapshotRoundTripPreservesFactorAndOwnership) {
  const auto initial = Fixture(8);
  const auto prior = Marginalize(initial, Linearize(initial, false), {initial.times.back()}, initial.ids,
                                 initial.times[6], initial.times[6]);
  const auto graph = FromPrior(prior);
  const auto path =
      std::filesystem::temp_directory_path() / ("cuvslam_joint_snapshot_" + std::to_string(getpid()) + ".bin");
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::filesystem::remove(path); }
  } cleanup{path};
  SaveGraph(graph, path.string());
  EXPECT_THROW(SaveGraph(graph, path.string()), std::runtime_error);
  const auto restored = LoadGraph(path.string());
  EXPECT_EQ(restored.times, graph.times);
  EXPECT_EQ(restored.ids, graph.ids);
  EXPECT_EQ(restored.prior.consumed_visual, graph.prior.consumed_visual);
  const auto a = Assemble(graph, Linearize(graph, false));
  const auto b = Assemble(restored, Linearize(restored, false));
  EXPECT_TRUE((a.rows.array() == b.rows.array()).all());
  EXPECT_TRUE((a.residual.array() == b.residual.array()).all());
}

TEST(JointPosteriorOnline, InitializationRetainsGaugeAndRejectionDoesNotOverwriteAcceptedState) {
  using namespace cuvslam;
  auto graph = Fixture(8);
  const auto root = graph.nodes.back();
  camera::Rig rig;
  imu::ImuCalibration calibration;
  CurrentFrame current{graph.times.back(), StatePose(root), {}, {}};
  Online backend;
  Initialization initialized;
  ASSERT_TRUE(backend.initialize(graph, current, rig, calibration, initialized));
  EXPECT_EQ(initialized.root_covariance.rows(), 9);
  EXPECT_GT(initialized.root_covariance.norm(), 0.);
  EXPECT_EQ(initialized.live.prior.times.size(), 1u);
  EXPECT_EQ(initialized.live.prior.factor().rows.cols(), 15 + 3 * int(initialized.live.ids.size()) + 2);
  EXPECT_EQ(Eigen::Map<const Eigen::Vector3d>(root.p),
            Eigen::Map<const Eigen::Vector3d>(initialized.complete.nodes.back().p));
  backend.adopt(std::move(initialized), current);
  sba_imu::Pose before;
  ASSERT_TRUE(backend.pose_at(current.time_ns, before));
  const auto covariance = backend.position_covariance(current.time_ns);
  ASSERT_TRUE(covariance);
  map::UnifiedMap map(20);
  map.set_gravity(graph.gravity.cast<float>());
  const map::State state{before.w_from_imu.inverse(), before.velocity, before.acc_bias, before.gyro_bias};
  map.add_keyframe(current.time_ns, state, sba_imu::IMUPreintegration(), {}, {});
  imu::ImuMeasurementStorage storage(100);
  pipelines::InertialPnPSettings settings;
  sba_imu::Pose rejected, predicted;
  bool valid_prediction = false;
  // An invalid clock is a structural failure, independent of scene thresholds.
  EXPECT_FALSE(backend.solve(map, rig, calibration, storage, {}, {}, settings, current.time_ns, false, rejected,
                             predicted, valid_prediction));
  EXPECT_FALSE(valid_prediction);
  backend.discard();
  sba_imu::Pose after;
  ASSERT_TRUE(backend.pose_at(current.time_ns, after));
  EXPECT_EQ(before.w_from_imu.matrix(), after.w_from_imu.matrix());
  EXPECT_EQ(before.velocity, after.velocity);
  EXPECT_EQ(before.gyro_bias, after.gyro_bias);
  EXPECT_EQ(before.acc_bias, after.acc_bias);
  EXPECT_EQ(*covariance, *backend.position_covariance(current.time_ns));
  const auto unchanged = map.get_consecutive_keyframes().back().keyframe->get_state();
  EXPECT_EQ(unchanged.rig_from_w.matrix(), state.rig_from_w.matrix());
  EXPECT_EQ(unchanged.velocity, state.velocity);
  backend.reset();
  EXPECT_FALSE(backend.ready());
}

TEST(JointPosteriorOnline, RealImuPredictionAndEmptyVisualRejectionLeaveAcceptedPosteriorUntouched) {
  using namespace cuvslam;
  const auto graph = Fixture(8);
  camera::Rig rig;
  imu::ImuCalibration calibration;
  CurrentFrame current{graph.times.back(), StatePose(graph.nodes.back()), {}, {}};
  Online backend;
  Initialization initialized;
  ASSERT_TRUE(backend.initialize(graph, current, rig, calibration, initialized));
  backend.adopt(std::move(initialized), current);
  sba_imu::Pose before;
  ASSERT_TRUE(backend.pose_at(current.time_ns, before));
  const auto covariance = backend.position_covariance(current.time_ns);
  map::UnifiedMap map(20);
  map.set_gravity(graph.gravity.cast<float>());
  const map::State state{before.w_from_imu.inverse(), before.velocity, before.acc_bias, before.gyro_bias};
  map.add_keyframe(current.time_ns, state, sba_imu::IMUPreintegration(), {}, {});
  imu::ImuMeasurementStorage storage(100);
  for (int i = -1; i <= 11; ++i) {
    storage.push_back(
        imu::ImuMeasurement{current.time_ns + i * 5000000LL, Vector3T(0.f, 0.f, 9.81f), Vector3T::Zero()});
  }
  pipelines::InertialPnPSettings settings;
  sba_imu::Pose candidate, prediction;
  bool prediction_valid = false;
  EXPECT_FALSE(backend.solve(map, rig, calibration, storage, {}, {}, settings, current.time_ns + 50000000LL, false,
                             candidate, prediction, prediction_valid));
  ASSERT_TRUE(prediction_valid);
  EXPECT_TRUE(prediction.w_from_imu.matrix().allFinite());
  backend.discard();
  sba_imu::Pose after;
  ASSERT_TRUE(backend.pose_at(current.time_ns, after));
  EXPECT_EQ(before.w_from_imu.matrix(), after.w_from_imu.matrix());
  EXPECT_EQ(before.velocity, after.velocity);
  EXPECT_EQ(before.acc_bias, after.acc_bias);
  EXPECT_EQ(before.gyro_bias, after.gyro_bias);
  EXPECT_EQ(*covariance, *backend.position_covariance(current.time_ns));
  EXPECT_EQ(map.get_consecutive_keyframes().back().keyframe->get_state().rig_from_w.matrix(),
            state.rig_from_w.matrix());
}

TEST(JointPosteriorOnline, ChangedPublishedGravityRejectsPendingCommitWithoutWritingStateOrCovariance) {
  using namespace cuvslam;
  const auto graph = Fixture(8, 20);
  camera::PinholeCameraModel camera(Vector2T(1024.f, 1024.f), Vector2T(400.f, 400.f), Vector2T(512.f, 512.f));
  camera::Rig rig;
  rig.num_cameras = 2;
  for (int c = 0; c < 2; ++c) {
    rig.intrinsics[c] = &camera;
    rig.camera_from_rig[c] = Isometry3T::Identity();
    const auto& visual = graph.visuals[c];
    rig.camera_from_rig[c].linear() =
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(visual.obs.Rci).cast<float>();
    rig.camera_from_rig[c].translation() = Eigen::Map<const Eigen::Vector3d>(visual.obs.tci).cast<float>();
  }
  imu::ImuCalibration calibration(Isometry3T::Identity(), .0001f, .00001f, .002f, .003f, 200.f);
  CurrentFrame current{graph.times.back(), StatePose(graph.nodes.back()), {}, {}};
  Online backend;
  Initialization initialized;
  ASSERT_TRUE(backend.initialize(graph, current, rig, calibration, initialized));
  const auto points = initialized.live.points;
  const auto ids = initialized.live.ids;
  const Vector3T gravity = initialized.live.gravity.cast<float>();
  backend.adopt(std::move(initialized), current);
  sba_imu::Pose before;
  ASSERT_TRUE(backend.pose_at(current.time_ns, before));
  const auto covariance = backend.position_covariance(current.time_ns);
  map::UnifiedMap map(20);
  map.set_gravity(gravity);
  const map::State state{before.w_from_imu.inverse(), before.velocity, before.acc_bias, before.gyro_bias};
  map.add_keyframe(current.time_ns, state, sba_imu::IMUPreintegration(), {}, {});
  imu::ImuMeasurementStorage storage(100);
  for (int i = -1; i <= 11; ++i) {
    storage.push_back(
        imu::ImuMeasurement{current.time_ns + i * 5000000LL, Vector3T(0.f, 0.f, 9.81f), Vector3T::Zero()});
  }
  const auto next = current.time_ns + 50000000LL;
  sba_imu::IMUPreintegration pi(calibration, storage, before.gyro_bias, before.acc_bias, current.time_ns, next);
  sba_imu::Pose predicted;
  ASSERT_TRUE(before.predict_pose(gravity, pi, predicted));
  std::vector<camera::Observation> observations;
  for (size_t i = 0; i < points.size(); ++i) {
    for (int c = 0; c < 2; ++c) {
      const Vector3T pc = rig.camera_from_rig[c] * (predicted.w_from_imu.inverse() * points[i].cast<float>());
      observations.emplace_back(c, TrackId(ids[i]), pc.hnormalized(), Matrix2T::Identity() * 10000.f);
    }
  }
  pipelines::InertialPnPSettings settings;
  sba_imu::Pose candidate, prediction;
  bool prediction_valid = false;
  ASSERT_TRUE(backend.solve(map, rig, calibration, storage, {}, observations, settings, next, false, candidate,
                            prediction, prediction_valid));
  ASSERT_TRUE(prediction_valid);
  // An independent writer changing a published mean invalidates the transaction.
  const Vector3T changed_gravity = gravity + Vector3T(.1f, 0.f, 0.f);
  map.set_gravity(changed_gravity);
  EXPECT_FALSE(backend.commit(map, calibration, next));
  sba_imu::Pose after;
  ASSERT_TRUE(backend.pose_at(current.time_ns, after));
  EXPECT_FALSE(backend.pose_at(next, after));
  EXPECT_EQ(before.w_from_imu.matrix(), after.w_from_imu.matrix());
  EXPECT_EQ(before.velocity, after.velocity);
  EXPECT_EQ(*covariance, *backend.position_covariance(current.time_ns));
  EXPECT_EQ(map.get_consecutive_keyframes().back().keyframe->get_state().rig_from_w.matrix(),
            state.rig_from_w.matrix());
  EXPECT_EQ(*map.get_gravity(), changed_gravity);
}

TEST(JointPosterior, BlockAssemblyPreservesFullCorrelatedNormalSystem) {
  const auto history = Fixture(6, 40);
  const auto prior = Marginalize(history, Linearize(history, false), {history.times.back()}, history.ids,
                                 history.times[4], history.times[4]);
  for (auto graph : {history, FromPrior(prior)}) {
    const int dim = 15 * graph.nodes.size() + 3 * graph.points.size() + 2;
    Vector change = Vector::Zero(dim);
    for (int k = 0; k < dim; ++k) {
      change[k] = 1e-4 * std::sin(k * .2);
    }
    graph = Retract(graph, change);
    const auto linear = Linearize(graph, false);
    const auto full = Assemble(graph, linear);
    Matrix h;
    Vector g;
    NormalEquations(graph, linear, h, g);
    const Matrix reference = full.rows.transpose() * full.rows;
    const Vector rhs = full.rows.transpose() * full.residual;
    EXPECT_LT((h - reference).norm() / reference.norm(), 1e-12);
    EXPECT_LT((g - rhs).norm() / std::max(1., rhs.norm()), 1e-12);
    EXPECT_LT((h - h.transpose()).norm() / h.norm(), 1e-13);
  }
}

TEST(JointPosterior, GuardedQrMatchesPivotedProjectionAndRejectsRankDeficiency) {
  namespace dense = cuvslam::pnp::joint_geometry::dense;
  if (!dense::blas().handle) {
    GTEST_SKIP() << "Optional LAPACK acceleration unavailable";
  }
  Matrix nuisance = Matrix::Zero(40, 10), kept(40, 5);
  Vector residual(40);
  for (int i = 0; i < 40; ++i) {
    residual[i] = std::sin(.37 * i);
    for (int j = 0; j < 8; ++j) {
      nuisance(i, j) = std::sin(.13 * (i + 1) * (j + 1)) + (i == j ? 2. : 0.);
    }
    for (int j = 0; j < 5; ++j) {
      kept(i, j) = std::cos(.23 * (i + 2) * (j + 1));
    }
  }
  for (int j = 0; j < 8; ++j) {
    nuisance.col(j).normalize();
  }
  Matrix expected, actual;
  Vector er, ar;
  gpu_window::project_out(nuisance, kept, residual, expected, er);
  ASSERT_TRUE(dense::project_full_rank(nuisance, kept, residual, actual, ar));
  EXPECT_LT((actual.transpose() * actual - expected.transpose() * expected).norm(), 1e-10);
  EXPECT_LT((actual.transpose() * ar - expected.transpose() * er).norm(), 1e-10);
  EXPECT_NEAR(ar.squaredNorm(), er.squaredNorm(), 1e-10);
  nuisance.col(7) = nuisance.col(6);
  EXPECT_FALSE(dense::project_full_rank(nuisance, kept, residual, actual, ar));
  nuisance.col(7) += 1e-12 * kept.col(0);
  EXPECT_FALSE(dense::project_full_rank(nuisance, kept, residual, actual, ar));
  nuisance.setZero();
  ASSERT_TRUE(dense::project_full_rank(nuisance, kept, residual, actual, ar));
  EXPECT_EQ(actual, kept);
  EXPECT_EQ(ar, residual);
}

namespace gpu_window::posterior {
void CachedNormalEquationsForTest(const Graph&, const Linearization&, Eigen::MatrixXd&, Eigen::VectorXd&);
}
TEST(JointPosterior, CachedPriorPreservesReanchoredMeanCorrelationAndNonzeroRhs) {
  const auto history = Fixture(8, 40);
  const auto prior = Marginalize(history, Linearize(history, false), {history.times.back()}, history.ids,
                                 history.times[6], history.times[6]);
  auto graph = FromPrior(prior);
  Vector change = Vector::Zero(15 * graph.nodes.size() + 3 * graph.points.size() + 2);
  for (int i = 0; i < change.size(); ++i) {
    change[i] = 1e-3 * std::sin(.3 * i);
  }
  graph = Retract(graph, change);
  const auto linear = Linearize(graph, false);
  Matrix h, reference;
  Vector g, rhs;
  NormalEquations(graph, linear, reference, rhs);
  CachedNormalEquationsForTest(graph, linear, h, g);
  EXPECT_GT(rhs.norm(), 1e-4);
  EXPECT_LT((h - reference).norm() / reference.norm(), 1e-12);
  EXPECT_LT((g - rhs).norm() / rhs.norm(), 1e-11);
  Factor broken = graph.prior.factor();
  broken.residual.resize(1);
  graph.prior.SetFactor(std::move(broken));
  EXPECT_THROW(CachedNormalEquationsForTest(graph, linear, h, g), std::invalid_argument);
}

#include <future>
#include "pnp/joint_gpu_dense.h"
TEST(JointPosteriorGpu, DoublePrecisionLinearAlgebraAndRankProtectionMatchCpu) {
  namespace dense = cuvslam::pnp::joint_geometry::dense;
  if (!std::getenv("VIO_JOINT_GPU_QR")) {
    GTEST_SKIP() << "GPU experiment disabled";
  }
  Matrix a(40, 12);
  Vector r(40);
  for (int i = 0; i < 40; ++i) {
    r[i] = std::sin(.17 * i);
    for (int j = 0; j < 12; ++j) {
      a(i, j) = std::sin(.13 * (i + 1) * (j + 1)) + (i == j ? 2. : 0.);
    }
  }
  Matrix gram;
  ASSERT_TRUE(dense::GpuGram(a, gram));
  const Matrix expected = a.transpose() * a;
  EXPECT_LT((gram - expected).norm() / expected.norm(), 1e-12);
  Matrix spd = expected + Matrix::Identity(12, 12);
  Vector rhs = Vector::Ones(12), x;
  ASSERT_TRUE(dense::GpuSolveSpd(spd, rhs, x));
  EXPECT_LT((spd * x - rhs).norm(), 1e-10);
  Matrix w;
  Vector wr;
  ASSERT_TRUE(dense::GpuCompress(a, r, w, wr));
  EXPECT_LT((w.transpose() * w - expected).norm() / expected.norm(), 1e-12);
  EXPECT_LT((w.transpose() * wr - a.transpose() * r).norm(), 1e-10);
  Matrix nuisance = a.leftCols(6), kept = a.rightCols(6), q, reference;
  Vector qr, rr;
  for (int j = 0; j < 6; ++j) {
    nuisance.col(j).normalize();
  }
  ASSERT_TRUE(dense::GpuProjectFullRank(nuisance, kept, r, q, qr));
  gpu_window::project_out(nuisance, kept, r, reference, rr);
  EXPECT_LT((q.transpose() * q - reference.transpose() * reference).norm(), 1e-10);
  EXPECT_LT((q.transpose() * qr - reference.transpose() * rr).norm(), 1e-10);
  nuisance.col(5) = nuisance.col(4);
  EXPECT_FALSE(dense::GpuProjectFullRank(nuisance, kept, r, q, qr));
  spd.setIdentity();
  spd(0, 0) = -1.;
  EXPECT_FALSE(dense::GpuSolveSpd(spd, rhs, x));
}
TEST(JointPosteriorGpu, BuffersArePrivateAcrossThreadsAndOldContentsAreOverwritten) {
  namespace dense = cuvslam::pnp::joint_geometry::dense;
  if (!std::getenv("VIO_JOINT_GPU_QR")) {
    GTEST_SKIP() << "GPU experiment disabled";
  }
  auto worker = [](int seed) {
    double worst = 0.;
    for (int n : {17, 31, 11}) {
      Matrix a(n, n);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          a(i, j) = std::sin(.13 * (i + 1) * (j + 1) + seed);
        }
      }
      Matrix h;
      if (!dense::GpuGram(a, h)) {
        return std::make_pair(false, 1.);
      }
      const Matrix ref = a.transpose() * a;
      worst = std::max(worst, (h - ref).norm() / ref.norm());
    }
    return std::make_pair(true, worst);
  };
  auto first = std::async(std::launch::async, worker, 1), second = std::async(std::launch::async, worker, 2);
  for (auto result : {first.get(), second.get()}) {
    EXPECT_TRUE(result.first);
    EXPECT_LT(result.second, 1e-12);
  }
}

#include <cstring>
namespace gpu_window::posterior {
Factor AssembleReferenceForTest(const Graph&, const Linearization&);
void AnalyticVisualForTest(const Visual&, const State&, const Eigen::Vector3d&, Eigen::Vector2d&,
                           Eigen::Matrix<double, 2, 18>&);
}  // namespace gpu_window::posterior
TEST(JointPosterior, ImmutablePriorSharesOnlyReadOnlyCoefficients) {
  auto history = Fixture(8, 20);
  const auto prior = Marginalize(history, Linearize(history, false), {history.times.back()}, history.ids,
                                 history.times[6], history.times[6]);
  auto first = FromPrior(prior), second = first;
  EXPECT_EQ(first.prior.factor().rows.data(), second.prior.factor().rows.data());
  const Vector original = first.prior.factor().residual;
  Factor changed = second.prior.factor();
  changed.residual.array() += .1;
  second.prior.SetFactor(std::move(changed));
  EXPECT_NE(first.prior.factor().rows.data(), second.prior.factor().rows.data());
  EXPECT_EQ(first.prior.factor().residual, original);
  second.prior.nodes.front().p[0] += .1;
  EXPECT_NE(first.prior.nodes.front().p[0], second.prior.nodes.front().p[0]);
}
TEST(JointPosterior, DirectAssemblyPreservesExactRowsResidualAndEvidence) {
  const auto history = Fixture(6, 20);
  const auto prior = Marginalize(history, Linearize(history, false), {history.times.back()}, history.ids,
                                 history.times[4], history.times[4]);
  for (const auto& graph : {history, FromPrior(prior)}) {
    const auto linear = Linearize(graph, false);
    const auto expected = AssembleReferenceForTest(graph, linear), actual = Assemble(graph, linear);
    ASSERT_EQ(expected.rows.rows(), actual.rows.rows());
    ASSERT_EQ(expected.rows.cols(), actual.rows.cols());
    EXPECT_EQ(std::memcmp(expected.rows.data(), actual.rows.data(), expected.rows.size() * sizeof(double)), 0);
    EXPECT_EQ(std::memcmp(expected.residual.data(), actual.residual.data(), expected.residual.size() * sizeof(double)),
              0);
    EXPECT_EQ(expected.evidence, actual.evidence);
  }
}
TEST(JointPosterior, AnalyticVisualMatchesAdAndFiniteDifferencesIncludingInheritedRotationAndCameraCenter) {
  auto graph = Fixture(2);
  auto visual = graph.visuals[27];
  auto node = graph.nodes[visual.node];
  for (int k = 0; k < 9; ++k) {
    node.R[k] = double(float(node.R[k]));
  }
  node.R[0] += 2e-7;
  visual.obs.tci[0] = .4;
  visual.obs.tci[1] = -.3;
  visual.obs.tci[2] = .1;
  visual.obs.sqrt_info[1] = 4.;
  const auto point = graph.points[visual.point];
  Eigen::Vector2d residual, reference;
  Eigen::Matrix<double, 2, 18> analytic, ad, numerical;
  AnalyticVisualForTest(visual, node, point, residual, analytic);
  for (int column = 0; column < 18; ++column) {
    gpu_vi::Jet x[18], y[2];
    for (int k = 0; k < 18; ++k) {
      x[k] = {0., k == column ? 1. : 0.};
    }
    gpu_window::visual_res(visual, node, point.data(), x, y);
    for (int row = 0; row < 2; ++row) {
      reference[row] = y[row].v;
      ad(row, column) = y[row].d;
    }
    double plus[18] = {}, minus[18] = {}, yp[2], ym[2];
    plus[column] = 1e-6;
    minus[column] = -1e-6;
    gpu_window::visual_res(visual, node, point.data(), plus, yp);
    gpu_window::visual_res(visual, node, point.data(), minus, ym);
    for (int row = 0; row < 2; ++row) {
      numerical(row, column) = (yp[row] - ym[row]) / 2e-6;
    }
  }
  EXPECT_LT((residual - reference).norm(), 1e-12);
  EXPECT_LT((analytic - ad).norm() / ad.norm(), 1e-12);
  EXPECT_LT((analytic - numerical).cwiseAbs().maxCoeff(), 1e-6);
  EXPECT_EQ(analytic.middleCols(6, 9), (Eigen::Matrix<double, 2, 9>::Zero()));
}
TEST(JointPosterior, ReusingOnlyTheSameSolveFinalLinearizationPreservesPosterior) {
  auto graph = Fixture(8, 20);
  Linearization final;
  const auto report = Optimize(graph, &final);
  ASSERT_TRUE(report.success);
  const auto reference = Assemble(graph, Linearize(graph, false)), actual = Assemble(graph, final);
  EXPECT_EQ(reference.rows, actual.rows);
  EXPECT_EQ(reference.residual, actual.residual);
  Matrix first, second;
  Vector g1, g2;
  ASSERT_TRUE(CurrentMarginal(graph, graph.times.back(), first, g1));
  ASSERT_TRUE(CurrentMarginal(graph, graph.times.back(), second, g2, &final));
  EXPECT_LT((first - second).norm() / first.norm(), 1e-10);
  EXPECT_LT((g1 - g2).norm(), 1e-8);
}

TEST(JointPosteriorDiagnostic, FixedGravityConstrainsBothRowsAndRetractionAfterTransfer) {
  const char* flag = "VIO_POSTERIOR_DIAG_FIXED_GRAVITY";
  ASSERT_EQ(std::getenv(flag), nullptr);
  auto initial = Fixture(2);
  auto stored = initial;
  stored.prior = Marginalize(initial, Linearize(initial, false), initial.times, initial.ids, -1, -1);
  stored.visuals.clear();
  stored.visual_cameras.clear();
  stored.edges.clear();
  setenv(flag, "1", 1);
  const auto conditioned = Assemble(stored, Linearize(stored, false));
  EXPECT_EQ(conditioned.rows.rightCols<2>().norm(), 0.);
  Eigen::MatrixXd hessian;
  Eigen::VectorXd gradient;
  NormalEquations(stored, Linearize(stored, false), hessian, gradient);
  EXPECT_EQ(hessian.rightCols<2>().norm(), 0.);
  EXPECT_EQ(gradient.tail<2>().norm(), 0.);
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(15 * initial.nodes.size() + 3 * initial.points.size() + 2);
  delta.tail<2>() << .1, -.2;
  EXPECT_TRUE(Retract(stored, delta).gravity == stored.gravity);
  // Initial gravity remains estimable before the transfer.
  EXPECT_GT((Retract(initial, delta).gravity - initial.gravity).norm(), .01);
  unsetenv(flag);
}

TEST(JointPosteriorDiagnostic, LegacyVisualInformationKeepsCorrelationsAndUnits) {
  const char* flag = "VIO_POSTERIOR_DIAG_RC2_VISUAL";
  ASSERT_EQ(std::getenv(flag), nullptr);
  Eigen::Matrix2d information;
  information << 8., 2., 2., 3.;
  const auto original = VisualWhitening(information);
  setenv(flag, "1", 1);
  const auto legacy = VisualWhitening(information);
  unsetenv(flag);
  EXPECT_LT((original.transpose() * original - information).norm(), 1e-13);
  EXPECT_LT((legacy.transpose() * legacy - 9. * information).norm(), 1e-12);
}

TEST(JointPosteriorDiagnostic, ConfiguredPixelSigmaPreservesFullInformation) {
  const char* key = "VIO_POSTERIOR_VISUAL_SIGMA_PX";
  ASSERT_EQ(std::getenv(key), nullptr);
  Eigen::Matrix2d information;
  information << 8., 2., 2., 3.;
  setenv(key, "1", 1);
  const auto one = VisualWhitening(information);
  setenv(key, "2", 1);
  const auto two = VisualWhitening(information);
  EXPECT_LT((one.transpose() * one - 9. * information).norm(), 1e-12);
  EXPECT_LT((two.transpose() * two - 2.25 * information).norm(), 1e-12);
  unsetenv(key);
  setenv("VIO_POSTERIOR_DIAG_RC2_VISUAL", "1", 1);
  const auto legacy = VisualWhitening(information);
  EXPECT_TRUE(one == legacy);
  setenv(key, "1", 1);
  EXPECT_THROW(VisualWhitening(information), std::invalid_argument);
  unsetenv("VIO_POSTERIOR_DIAG_RC2_VISUAL");
  for (const char* bad : {"0", "-1", "nan", "inf", "1junk", "1e-320", "1e-200", "1e200"}) {
    setenv(key, bad, 1);
    EXPECT_THROW(VisualWhitening(information), std::invalid_argument);
  }
  unsetenv(key);
}

TEST(JointPosteriorDiagnostic, InvalidNoiseFailsBeforeTracking) {
  setenv("VIO_POSTERIOR_VISUAL_SIGMA_PX", "0", 1);
  EXPECT_THROW(Online{}, std::invalid_argument);
  unsetenv("VIO_POSTERIOR_VISUAL_SIGMA_PX");
}
