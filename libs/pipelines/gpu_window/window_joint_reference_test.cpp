#include "pipelines/gpu_window/window_joint_reference.h"
#include <gtest/gtest.h>
#include "pipelines/gpu_window/window_backend.h"
using namespace gpu_window;
namespace {
Problem Fixture(int count) {
  Problem p;
  p.count = count;
  p.iterations = 10;
  p.huber = 2.;
  for (int n = 0; n < Slots; ++n) {
    p.anchors[n] = p.nodes[n];
    for (int k = 0; k < 15; ++k) {
      if (n == 0 || n >= count) {
        p.W[(15 * n + k) * Dim + 15 * n + k] = 10.;
      }
    }
  }
  for (int i = 0; i < 12; ++i) {
    p.points.push_back({.13 * (i % 4 - 1), .15 * (i / 4 - 1), 2. + .12 * i});
  }
  for (int n = 0; n < count; ++n) {
    p.nodes[n].p[0] = .01 * n;
    for (int i = 0; i < int(p.points.size()); ++i) {
      for (int c = 0; c < 2; ++c) {
        Visual v;
        v.node = n;
        v.point = i;
        v.obs.Rci[0] = v.obs.Rci[4] = v.obs.Rci[8] = 1.;
        v.obs.tci[0] = -.2 * c;
        v.obs.xy[0] = (p.points[i][0] - .01 * n - .2 * c) / p.points[i][2];
        v.obs.xy[1] = p.points[i][1] / p.points[i][2];
        v.obs.sqrt_info[0] = v.obs.sqrt_info[3] = 300.;
        v.obs.min_depth = .05;
        p.visuals.push_back(v);
      }
    }
  }
  for (int n = 0; n < count - 1; ++n) {
    Edge e;
    e.left = n;
    e.right = n + 1;
    auto& d = e.data;
    d.dt = .1;
    d.gravity[2] = -9.81;
    d.dR[0] = d.dR[4] = d.dR[8] = 1.;
    d.dV[2] = .981;
    d.dP[2] = .04905;
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
      const auto& s = p.nodes[n + endpoint];
      std::copy(s.R, s.R + 9, d.R[endpoint]);
      std::copy(s.p, s.p + 3, d.p[endpoint]);
      std::copy(s.v, s.v + 3, d.v[endpoint]);
    }
    for (int k = 0; k < 9; ++k) {
      d.imu_sqrt[9 * k + k] = 20.;
    }
    for (int k = 0; k < 3; ++k) {
      d.JRg[3 * k + k] = -.1;
      d.JVa[3 * k + k] = -.1;
      d.JPa[3 * k + k] = -.005;
    }
    for (int k = 0; k < 6; ++k) {
      e.rw[6 * k + k] = 30.;
    }
    p.edges.push_back(e);
  }
  return p;
}
std::vector<int64_t> Ids(const Problem& p) {
  std::vector<int64_t> result;
  for (size_t i = 0; i < p.points.size(); ++i) {
    result.push_back(i + 1);
  }
  return result;
}
std::vector<int64_t> Times(const Problem& p) {
  std::vector<int64_t> result;
  for (int i = 0; i < p.count; ++i) {
    result.push_back(i * 100000000LL);
  }
  return result;
}
joint_reference::Prior Retained(const Problem& p) {
  std::vector<double> x(15 * p.count + 3 * p.points.size(), 0.);
  auto e = fp64::EvaluateHost(p, x, true);
  auto boundary = joint_reference::linearize(p, e, x, {}, Times(p), Ids(p), true);
  std::vector<int> kept;
  for (int k = 15; k < 30 + 3 * int(p.points.size()); ++k) {
    kept.push_back(k);
  }
  joint_reference::Prior prior;
  prior.factor = cuvslam::pnp::joint_geometry::compress(cuvslam::pnp::joint_geometry::eliminate(boundary.factor, kept));
  prior.times = {100000000LL};
  prior.states = {p.nodes[1]};
  prior.factor = joint_reference::Reanchor(prior.factor, prior.states);
  prior.ids = Ids(p);
  for (const auto& point : p.points) {
    prior.points.emplace_back(point[0], point[1], point[2]);
  }
  prior.absorbed_through_ns = 0;
  return prior;
}
Problem Later(const Problem& batch) {
  Problem p = batch;
  p.count = 2;
  p.nodes[0] = batch.nodes[1];
  p.nodes[1] = batch.nodes[2];
  p.visuals.clear();
  p.edges.clear();
  p.W.fill(0.);
  p.offset.fill(0.);
  for (int n = 2; n < Slots; ++n) {
    for (int k = 0; k < 15; ++k) {
      p.W[(15 * n + k) * Dim + 15 * n + k] = 1.;
    }
  }
  for (auto v : batch.visuals) {
    if (v.node) {
      --v.node;
      p.visuals.push_back(v);
    }
  }
  auto edge = batch.edges[1];
  edge.left = 0;
  edge.right = 1;
  p.edges.push_back(edge);
  return p;
}
}  // namespace
TEST(JointWindowReference, BoundaryPosteriorMatchesCompleteVIOFactorBatch) {
  const auto first = Fixture(2);
  const auto batch = Fixture(3);
  const auto prior = Retained(first);
  const auto later = Later(batch);
  std::vector<double> x(15 * batch.count + 3 * batch.points.size(), 0.);
  auto full = joint_reference::linearize(batch, fp64::EvaluateHost(batch, x, true), x, {}, Times(batch), Ids(batch));
  std::vector<double> xi(15 * later.count + 3 * later.points.size(), 0.);
  auto incremental = joint_reference::linearize(later, fp64::EvaluateHost(later, xi, true), xi, prior,
                                                {100000000LL, 200000000LL}, Ids(later));
  std::vector<int> full_current, inc_current;
  for (int k = 0; k < 15; ++k) {
    full_current.push_back(30 + k);
    inc_current.push_back(15 + k);
  }
  const auto a = cuvslam::pnp::joint_geometry::marginal(full.factor, full_current);
  const auto b = cuvslam::pnp::joint_geometry::marginal(incremental.factor, inc_current);
  ASSERT_TRUE(a.observable);
  ASSERT_TRUE(b.observable);
  EXPECT_LT((a.information - b.information).norm() / a.information.norm(), 1e-9);
  EXPECT_LT((a.covariance - b.covariance).norm() / a.covariance.norm(), 1e-8);
  EXPECT_LT((a.gradient - b.gradient).norm(), 1e-8);
  const Eigen::MatrixXd H = prior.factor.rows.transpose() * prior.factor.rows;
  EXPECT_GT(H.topRightCorner(15, H.cols() - 15).norm(), 1e-6);
  EXPECT_GT(H.block(15, 18, 3, 3).norm(), 1e-6);  // Distinct points retain common-pose correlation.
}
TEST(JointWindowReference, ActualPriorManifoldGradientMatchesFiniteDifferences) {
  const auto first = Fixture(2);
  const auto prior = Retained(first);
  const auto p = Later(Fixture(3));
  std::vector<double> x(15 * p.count + 3 * p.points.size(), 0.);
  x[0] = .01;
  x[1] = -.015;
  x[3] = .003;
  x[18] = -.002;
  x[30] = .001;
  const auto eval = [&](const std::vector<double>& value) {
    return joint_reference::linearize(p, fp64::EvaluateHost(p, value, true), value, prior, {100000000LL, 200000000LL},
                                      Ids(p));
  };
  const auto current = eval(x);
  const Eigen::VectorXd gradient = current.factor.rows.transpose() * current.factor.residual;
  double worst = 0.;
  const int navigation = 30;
  const int dim = navigation + 3 * p.points.size();
  for (int k = 0; k < dim; ++k) {
    auto plus = x, minus = x;
    const int col = k;
    plus[col] += 1e-6;
    minus[col] -= 1e-6;
    const double numeric = (eval(plus).cost - eval(minus).cost) / 2e-6;
    worst = std::max(worst, std::abs(numeric - gradient[k]) / std::max({1., std::abs(numeric), std::abs(gradient[k])}));
  }
  EXPECT_LT(worst, 1e-5);
}
TEST(JointWindowReference, ConsumedTimestampCannotBeAddedAgain) {
  const auto p = Fixture(2);
  const auto prior = Retained(p);
  std::vector<double> x(15 * p.count + 3 * p.points.size(), 0.);
  EXPECT_THROW(joint_reference::linearize(p, fp64::EvaluateHost(p, x, true), x, prior, Times(p), Ids(p)),
               std::invalid_argument);
}
TEST(JointWindowReference, NonlinearFullJointSolveReducesActualObjective) {
  auto p = Later(Fixture(3));
  auto prior = Retained(Fixture(2));
  p.nodes[1].p[0] += .005;
  const auto result = joint_reference::optimize(p, prior, {100000000LL, 200000000LL}, Ids(p));
  EXPECT_LT(result.after, result.before);
  EXPECT_TRUE(std::isfinite(result.after));
  const auto x = joint_reference::unpack(p, result);
  const auto checked =
      joint_reference::linearize(p, fp64::EvaluateHost(p, x, true), x, prior, {100000000LL, 200000000LL}, Ids(p));
  EXPECT_NEAR(checked.cost, result.after, 1e-5 * std::max(1.f, result.before));
}

TEST(JointWindowReference, ReanchorPreservesJacobianForStoredFloatRotations) {
  State state;
  Eigen::Matrix3f R = Eigen::AngleAxisf(.043f, Eigen::Vector3f(1, 2, 3).normalized()).toRotationMatrix();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      state.R[3 * r + c] = R(r, c);
    }
  }
  joint_reference::Factor raw;
  raw.rows = Eigen::MatrixXd::Identity(15, 15);
  raw.residual = Eigen::VectorXd::Zero(15);
  const auto corrected = joint_reference::Reanchor(raw, {state});
  Eigen::Matrix<double, 15, 15> D;
  for (int c = 0; c < 15; ++c) {
    gpu_vi::Jet z[15], r[15];
    for (int k = 0; k < 15; ++k) {
      z[k] = {0., k == c ? 1. : 0.};
    }
    difference(state, state, z, r);
    for (int k = 0; k < 15; ++k) {
      D(k, c) = r[k].d;
    }
  }
  EXPECT_LT((corrected.rows * D - raw.rows).norm(), 1e-12);
}

#include "camera/camera.h"
TEST(JointWindowReference, HistoricalSeedMatchesRawVisualFactorsAndOwnsRootObservations) {
  using namespace cuvslam;
  camera::FisheyeCameraModel lens(Vector2T(1200, 1200), Vector2T(300, 300), Vector2T(600, 600), 0, 0, 0, 0);
  camera::Rig rig;
  rig.num_cameras = 2;
  for (int c = 0; c < 2; ++c) {
    rig.intrinsics[c] = &lens;
    rig.camera_from_rig[c] = Isometry3T::Identity();
  }
  rig.camera_from_rig[1].translation().x() = -.2f;
  imu::ImuCalibration calibration(Isometry3T::Identity(), .0001f, .00001f, .002f, .003f, 200.f);
  map::UnifiedMap history(4);
  auto p = Fixture(2);
  p.W.fill(0.);
  p.offset.fill(0.);
  p.edges.clear();
  p.visuals.clear();
  std::vector<pipelines::Landmark> created;
  std::map<TrackId, Vector3T> points;
  for (size_t i = 0; i < p.points.size(); ++i) {
    Vector3T point(p.points[i][0], p.points[i][1], p.points[i][2]);
    points.emplace(TrackId(i + 1), point);
    created.push_back({TrackId(i + 1), point});
    for (int k = 0; k < 3; ++k) {
      p.points[i][k] = point[k];
    }
  }
  for (int node = 0; node < 2; ++node) {
    map::State state;
    state.rig_from_w = Isometry3T::Identity();
    state.rig_from_w.translation().x() = -.01f * node;
    const auto wi = state.rig_from_w.inverse();
    for (int k = 0; k < 3; ++k) {
      p.nodes[node].p[k] = wi.translation()[k];
      for (int c = 0; c < 3; ++c) {
        p.nodes[node].R[3 * k + c] = wi.linear()(k, c);
      }
    }
    std::vector<camera::Observation> observations;
    for (const auto& [id, point] : points) {
      for (int cam = 0; cam < 2; ++cam) {
        const Vector2T xy = (rig.camera_from_rig[cam] * state.rig_from_w * point).hnormalized();
        observations.emplace_back(CameraId(cam), id, xy, Matrix2T::Identity() * 10000.f);
        Visual v;
        v.node = node;
        v.point = static_cast<int>(id) - 1;
        v.obs.Rci[0] = v.obs.Rci[4] = v.obs.Rci[8] = 1.;
        v.obs.tci[0] = rig.camera_from_rig[cam].translation().x();
        v.obs.xy[0] = xy.x();
        v.obs.xy[1] = xy.y();
        v.obs.sqrt_info[0] = v.obs.sqrt_info[3] = 300.;
        v.obs.min_depth = .05;
        p.visuals.push_back(v);
      }
    }
    history.add_keyframe(node * 100000000LL, state, {}, observations,
                         node == 0 ? created : std::vector<pipelines::Landmark>{});
  }
  joint_reference::Prior prior;
  ASSERT_TRUE(
      joint_reference::SeedHistoricalGeometry(history, rig, calibration, 100000000LL, p.nodes[1], points, prior));
  EXPECT_EQ(prior.visual_absorbed_through_ns, 100000000LL);
  EXPECT_EQ(prior.absorbed_through_ns, -1);
  std::vector<double> x(30 + 3 * p.points.size(), 0.);
  const auto e = fp64::EvaluateHost(p, x, true);
  joint_reference::Factor raw;
  raw.rows = Eigen::MatrixXd::Zero(2 * p.visuals.size(), 6 + 3 * p.points.size());
  raw.residual = Eigen::Map<const Eigen::VectorXd>(e.visual.data(), e.visual.size());
  for (size_t i = 0; i < p.visuals.size(); ++i) {
    const auto& v = p.visuals[i];
    const Eigen::Map<const Eigen::Matrix<double, 2, 18, Eigen::RowMajor>> J(e.visual_j.data() + 36 * i);
    if (v.node) {
      raw.rows.block(2 * i, 0, 2, 6) = J.leftCols(6);
    }
    raw.rows.block(2 * i, 6 + 3 * v.point, 2, 3) = J.rightCols(3);
  }
  std::vector<int> kept;
  for (int k = 0; k < 6; ++k) {
    kept.push_back(k);
  }
  for (int k = 15; k < prior.factor.rows.cols(); ++k) {
    kept.push_back(k);
  }
  const auto seeded = cuvslam::pnp::joint_geometry::eliminate(prior.factor, kept);
  const Eigen::MatrixXd expected = raw.rows.transpose() * raw.rows;
  EXPECT_LT((seeded.rows.transpose() * seeded.rows - expected).norm() / expected.norm(), 1e-8);
  EXPECT_LT((seeded.rows.transpose() * seeded.residual - raw.rows.transpose() * raw.residual).norm(), 1e-7);
  const auto marginal = cuvslam::pnp::joint_geometry::marginal(seeded, {0, 1, 2, 3, 4, 5});
  ASSERT_TRUE(marginal.observable);
  auto future = Later(Fixture(3));
  future.nodes[0] = p.nodes[1];
  future.points = p.points;
  // Root observations are owned by prior; only new visual observations remain.
  future.visuals.erase(
      std::remove_if(future.visuals.begin(), future.visuals.end(), [](const auto& v) { return v.node == 0; }),
      future.visuals.end());
  std::vector<double> next_x(30 + 3 * future.points.size(), 0.);
  EXPECT_NO_THROW(joint_reference::linearize(future, fp64::EvaluateHost(future, next_x, true), next_x, prior,
                                             {100000000LL, 200000000LL}, Ids(future)));
}

TEST(JointWindowReference, RealOrdinaryFrameTransactionsRetainAndMarginalize) {
  using namespace cuvslam;
  setenv("VIO_WINDOW_JOINT_GEOMETRY", "1", 1);
  setenv("CUVSLAM_WINDOW_SOLVER", "schur_fp64", 1);
  setenv("CUVSLAM_WINDOW_SIZE", "3", 1);
  struct Restore {
    ~Restore() {
      unsetenv("VIO_WINDOW_JOINT_GEOMETRY");
      unsetenv("CUVSLAM_WINDOW_SOLVER");
      unsetenv("CUVSLAM_WINDOW_SIZE");
    }
  } restore;
  camera::PinholeCameraModel lens(Vector2T(640, 480), Vector2T(320, 320), Vector2T(320, 240));
  camera::Rig rig;
  rig.num_cameras = 2;
  for (int c = 0; c < 2; ++c) {
    rig.intrinsics[c] = &lens;
    rig.camera_from_rig[c] = Isometry3T::Identity();
    rig.camera_from_rig[c].translation().x() = -.2f * c;
  }
  imu::ImuCalibration cal(Isometry3T::Identity(), .0001f, .00001f, .002f, .003f, 200.f);
  imu::ImuMeasurementStorage samples(200);
  for (int i = 0; i <= 100; ++i) {
    samples.push_back(imu::ImuMeasurement{1000000000LL + i * 5000000LL, Vector3T(0, 0, 9.81f), Vector3T::Zero()});
  }
  std::unordered_map<TrackId, Vector3T> points;
  std::vector<pipelines::Landmark> created;
  std::vector<camera::Observation> observations;
  for (int k = 0; k < 24; ++k) {
    Vector3T point(.1f * (k % 6 - 2), .1f * (k / 6 - 1), 2.f + .03f * k);
    points.emplace(k + 1, point);
    created.push_back({TrackId(k + 1), point});
    for (int c = 0; c < 2; ++c) {
      observations.emplace_back(CameraId(c), TrackId(k + 1), (rig.camera_from_rig[c] * point).hnormalized(),
                                Matrix2T::Identity() * 10000.f);
    }
  }
  map::UnifiedMap history(20);
  map::State state{Isometry3T::Identity(), Vector3T::Zero(), Vector3T::Zero(), Vector3T::Zero()};
  history.add_keyframe(1000000000LL, state, {}, observations, created);
  history.add_keyframe(1050000000LL, state, {}, observations, {});
  sba_imu::WindowBackend backend;
  sba_imu::Pose previous;
  previous.info = Matrix15T::Identity() * 100;
  int64_t previous_time = 1050000000LL;
  pipelines::InertialPnPSettings settings;
  settings.min_observations = 12;
  settings.max_iteration = 4;
  settings.robustifier_scale = 2;
  const Vector3T gravity(0, 0, -9.81f);
  for (int step = 0; step < 5; ++step) {
    const int64_t time = previous_time + 50000000LL;
    auto candidate = previous;
    ASSERT_TRUE(backend.solve({history, cal, rig, samples, points, observations, gravity, settings, previous, previous,
                               previous_time, time, [](const auto&, const auto&) { return true; }, false},
                              candidate));
    ASSERT_TRUE(backend.commit(history, cal, time));
    EXPECT_EQ(history.size(), 2u);  // Ordinary states stay in the lag, not the public KF map.
    EXPECT_LT(candidate.w_from_imu.translation().norm(), 1e-5);
    EXPECT_TRUE(candidate.info.allFinite());
    previous = candidate;
    previous_time = time;
  }
}
