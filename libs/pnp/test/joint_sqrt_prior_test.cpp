#include "pnp/joint_sqrt_prior.h"
#include <gtest/gtest.h>
using namespace cuvslam::pnp::joint_geometry;

TEST(JointGeometryGaussian, PosePointCorrelationMatchesFullGaussianMarginal) {
  Eigen::Matrix4d L = Eigen::Matrix4d::Identity();
  L(1, 0) = .3;
  L(2, 0) = .6;
  L(2, 1) = .2;
  L(3, 0) = -.4;
  L(3, 1) = .5;
  L(3, 2) = .2;
  const Eigen::Matrix4d C = L * L.transpose();
  Factor prior;
  prior.rows = Eigen::MatrixXd::Zero(4, 6);
  prior.rows.leftCols(4) = C.inverse().llt().matrixU();
  prior.residual = Eigen::Vector4d(.1, .2, -.1, .3);
  prior.evidence = {1};
  Factor observations;
  observations.rows = Eigen::MatrixXd::Zero(2, 6);
  observations.rows.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity();
  observations.rows.rightCols(2) = -Eigen::Matrix2d::Identity();
  observations.residual = Eigen::Vector2d(.2, -.3);
  observations.evidence = {2};
  auto graph = combine({prior, observations}, 6);
  const Eigen::MatrixXd H = graph.rows.transpose() * graph.rows;
  const Eigen::MatrixXd covariance = H.inverse();
  const auto result = marginal(graph, {4, 5});
  ASSERT_TRUE(result.observable);
  EXPECT_LT((result.covariance - covariance.bottomRightCorner<2, 2>()).norm(), 1e-11);
  EXPECT_GT((covariance.block<2, 2>(0, 2).norm()), .1);
  // Historical poses have finite covariance. Conditioning on exact history
  // removes their map correlation and produces a different uncertainty.
  Factor fixed = graph;
  fixed.rows.leftCols(2).setZero();
  const auto conditional = marginal(fixed, {4, 5});
  ASSERT_TRUE(conditional.observable);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> diff(result.covariance - conditional.covariance);
  EXPECT_GT(diff.eigenvalues().minCoeff(), 1e-6);
}
TEST(JointGeometryGaussian, MissingHistoryDoesNotMeanFixedMapPoint) {
  Factor fresh;
  fresh.rows.resize(2, 4);
  fresh.rows << -1, 0, 1, 0, 0, -1, 0, 1;
  fresh.residual = Eigen::Vector2d(.1, .2);
  fresh.evidence = {1};
  const auto result = marginal(fresh, {0, 1});
  EXPECT_FALSE(result.observable);
  EXPECT_EQ(result.covariance.size(), 0);  // Unknown, not a zero covariance.
  EXPECT_LT(result.information.norm(), 1e-20);
}
TEST(JointGeometryGaussian, BoundaryRetainsPosePointCrossInformationAndMatchesBatch) {
  // Variables: old pose, retained pose, landmark, future pose.
  Factor history;
  history.rows = Eigen::MatrixXd::Zero(4, 3);
  history.rows << 1, 0, 0, -1, 0, 1, 0, -1, 1, -1, 1, 0;
  history.residual = Eigen::Vector4d(.1, .2, -.2, .3);
  history.evidence = {1, 2, 3, 4};
  const auto retained = compress(eliminate(history, {1, 2}));
  EXPECT_GT(std::abs((retained.rows.transpose() * retained.rows)(0, 1)), .1);
  Factor lifted = retained;
  lifted.rows.conservativeResize(retained.rows.rows(), 3);
  lifted.rows.rightCols(1).setZero();
  Factor next;
  next.rows.resize(2, 3);
  next.rows << -1, 0, 1, 0, 1, -1;
  next.residual = Eigen::Vector2d(.2, .4);
  next.evidence = {5, 6};
  const auto incremental = combine({lifted, next}, 3);
  const auto inc = marginal(incremental, {2});
  ASSERT_TRUE(inc.observable);
  Factor fullhistory = history;
  fullhistory.rows.conservativeResize(4, 4);
  fullhistory.rows.rightCols(1).setZero();
  Factor fullnext;
  fullnext.rows = Eigen::MatrixXd::Zero(2, 4);
  fullnext.rows.rightCols(3) = next.rows;
  fullnext.residual = next.residual;
  fullnext.evidence = next.evidence;
  const auto batch = marginal(combine({fullhistory, fullnext}, 4), {3});
  ASSERT_TRUE(batch.observable);
  EXPECT_LT((inc.information - batch.information).norm(), 1e-12);
  EXPECT_LT((inc.covariance - batch.covariance).norm(), 1e-12);
  EXPECT_LT((inc.gradient - batch.gradient).norm(), 1e-12);
}
TEST(JointGeometryGaussian, EvidenceCannotBeReaddedAfterMarginalization) {
  Factor row;
  row.rows = Eigen::MatrixXd::Identity(2, 2);
  row.residual = Eigen::Vector2d::Zero();
  row.evidence = {11};
  const auto prior = compress(row);
  EXPECT_THROW(combine({prior, row}, 2), std::invalid_argument);
}

TEST(JointGeometryGaussian, LaboratoryLapackPreservesRankGradientAndInformation) {
  if (!dense::blas().handle) {
    GTEST_SKIP() << "Explicit laboratory BLAS library not selected";
  }
  Eigen::MatrixXd nuisance(120, 25), kept(120, 9);
  Eigen::VectorXd r(120);
  for (int i = 0; i < 120; ++i) {
    r[i] = std::sin(.193 * (i + 1));
    for (int j = 0; j < 25; ++j) nuisance(i, j) = std::sin(.017 * (i + 1) * (j + 2));
    for (int j = 0; j < 9; ++j) kept(i, j) = std::cos(.031 * (i + 1) * (j + 2));
  }
  nuisance.col(23) = nuisance.col(3);
  nuisance.col(24).setZero();
  Eigen::MatrixXd reference, actual;
  Eigen::VectorXd rr, ar;
  gpu_window::project_out(nuisance, kept, r, reference, rr);
  dense::project(nuisance, kept, r, actual, ar);
  ASSERT_EQ(reference.rows(), actual.rows());
  // This sinusoidal nuisance matrix is ill conditioned. Compare BOTH QR
  // implementations with an independent SVD projector and a roundoff bound
  // derived from its measured condition, rather than an absolute magic bound.
  Eigen::MatrixXd scaled = nuisance;
  for (int j = 0; j < scaled.cols(); ++j) {
    const double norm = scaled.col(j).norm();
    if (norm > 1e-30) scaled.col(j) /= norm;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(scaled, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto singular = svd.singularValues();
  const int rank = (singular.array() > 1e-10 * singular[0]).count();
  ASSERT_EQ(actual.rows(), 120 - rank);
  const Eigen::MatrixXd q = svd.matrixU().leftCols(rank);
  const Eigen::MatrixXd oracle = kept - q * (q.transpose() * kept);
  const double roundoff = 64. * std::numeric_limits<double>::epsilon() * scaled.rows() * singular[0] /
                          singular[rank - 1] * std::max(1., kept.squaredNorm());
  const Eigen::MatrixXd information = oracle.transpose() * oracle;
  EXPECT_LT((reference.transpose() * reference - information).norm(), roundoff);
  EXPECT_LT((actual.transpose() * actual - information).norm(), roundoff);
  // Repeat with well-conditioned orthonormal columns and true null columns.
  scaled.leftCols(rank) = svd.matrixU().leftCols(rank);
  scaled.col(23) = scaled.col(3);
  scaled.col(24).setZero();
  Eigen::MatrixXd a0, a1;
  Eigen::VectorXd b0, b1;
  gpu_window::project_out(scaled, kept, r, a0, b0);
  dense::project(scaled, kept, r, a1, b1);
  EXPECT_LT((a0.transpose() * a0 - a1.transpose() * a1).norm(), 1e-10);
  EXPECT_LT((a0.transpose() * b0 - a1.transpose() * b1).norm(), 1e-10);
  EXPECT_LT((reference.transpose() * rr - actual.transpose() * ar).norm(), 1e-10);
  EXPECT_LT(std::abs(rr.squaredNorm() - ar.squaredNorm()), 1e-10);
  Eigen::MatrixXd w0, w1;
  Eigen::VectorXd c0, c1;
  gpu_window::compress_prior(kept, r, w0, c0);
  dense::compress(kept, r, w1, c1);
  EXPECT_LT((w0.transpose() * w0 - w1.transpose() * w1).norm(), 1e-10);
  EXPECT_LT((w0.transpose() * c0 - w1.transpose() * c1).norm(), 1e-10);
  EXPECT_LT((dense::gram(kept) - kept.transpose() * kept).norm(), 1e-10);
  EXPECT_LT((dense::multiply(kept.transpose(), nuisance) - kept.transpose() * nuisance).norm(), 1e-10);
  Eigen::MatrixXd h = kept.transpose() * kept + Eigen::MatrixXd::Identity(9, 9);
  Eigen::VectorXd rhs = kept.transpose() * r, solution;
  ASSERT_TRUE(dense::solve_spd(h, rhs, solution));
  EXPECT_LT((solution - h.ldlt().solve(rhs)).norm(), 1e-11);
}
