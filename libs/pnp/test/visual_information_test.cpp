#include "pnp/visual_information.h"
#include <gtest/gtest.h>
using namespace cuvslam::pnp;
TEST(VisualInformation, InversionAndLeverArmMatchIndependentBodyFiniteDifferences) {
  Eigen::Isometry3d W = Eigen::Isometry3d::Identity(), E = W;
  W.linear() = Eigen::AngleAxisd(.8, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  W.translation() << 9, -6, 4;
  E.linear() = Eigen::AngleAxisd(-.4, Eigen::Vector3d(2, -1, 3).normalized()).toRotationMatrix();
  E.translation() << .23, -.16, .07;
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero(), numeric = H;
  for (int k = 0; k < 50; ++k) {
    Eigen::Vector3d P = W * Eigen::Vector3d(.03 * k - .7, .04 * k - .9, 3. + .02 * k);
    auto f = [&](const Eigen::Isometry3d& X) -> Eigen::Vector2d {
      const Eigen::Vector3d p = E * X.inverse() * P;
      return p.hnormalized();
    };
    const Eigen::Vector3d pc = E * W.inverse() * P;
    Eigen::Matrix<double, 2, 3> J;
    J << 1. / pc.z(), 0., -pc.x() / (pc.z() * pc.z()), 0., 1. / pc.z(), -pc.y() / (pc.z() * pc.z());
    Eigen::Matrix<double, 3, 6> K;
    K.leftCols<3>() = -InformationSkew(P);
    K.rightCols<3>().setIdentity();
    const Eigen::Matrix<double, 2, 6> A = J * (E * W.inverse()).linear() * K;
    H += A.transpose() * A;
    Eigen::Matrix<double, 2, 6> N;
    for (int c = 0; c < 6; ++c) {
      auto plus = W, minus = W;
      const double h = 1e-6;
      if (c < 3) {
        Eigen::Vector3d axis = Eigen::Vector3d::Unit(c);
        plus.linear() = W.linear() * Eigen::AngleAxisd(h, axis).toRotationMatrix();
        minus.linear() = W.linear() * Eigen::AngleAxisd(-h, axis).toRotationMatrix();
      } else {
        plus.translation() += h * W.linear().col(c - 3);
        minus.translation() -= h * W.linear().col(c - 3);
      }
      N.col(c) = (f(plus) - f(minus)) / (2 * h);
    }
    numeric += N.transpose() * N;
  }
  auto actual = WorldTwistToBodyInformation(H, W);
  EXPECT_LT((actual - numeric).norm() / numeric.norm(), 1e-8);
  EXPECT_GT((actual.topRightCorner<3, 3>().norm()), .1);
}
TEST(VisualInformation, NonfiniteInformationIsNotHidden) {
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Identity();
  H(0, 0) = NAN;
  EXPECT_FALSE(WorldTwistToBodyInformation(H, Eigen::Isometry3d::Identity()).allFinite());
}

TEST(VisualInformation, IndependentMeasurementsAddInformationWithoutSolverAveraging) {
  Eigen::Matrix<double, 2, 6> J;
  J << 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1;
  const Eigen::Matrix<double, 6, 6> one = VisualInformationWeight(0., 3.) * J.transpose() * J;
  Eigen::Matrix<double, 6, 6> many = Eigen::Matrix<double, 6, 6>::Zero();
  for (int k = 0; k < 17; ++k) {
    many += VisualInformationWeight(0., 3.) * J.transpose() * J;
  }
  EXPECT_LT((many - 17. * one).norm(), 1e-12);
  EXPECT_DOUBLE_EQ(VisualInformationWeight(0., 3.), 1.);
  EXPECT_GT(VisualInformationWeight(1., 3.), 0.);
  EXPECT_LT(VisualInformationWeight(1., 3.), 1.);
}
