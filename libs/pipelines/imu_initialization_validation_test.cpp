#include "imu_initialization_validation.h"
#include <cassert>
#include <limits>
using namespace cuvslam;
using namespace cuvslam::pipelines;
int main() {
  assert(ValidInitInterval(1000000000, 1033333333, 1.f / 30));
  assert(!ValidInitInterval(1000000000, 1033333333, .03f));
  assert(!ValidInitInterval(1000000000, 1000000000, 0));
  assert(!ValidInitInterval(1000000000, 1033333333, std::numeric_limits<float>::quiet_NaN()));
  std::vector<sba_imu::Pose> p(3);
  for (auto& x : p) {
    x.w_from_imu = Isometry3T::Identity();
    x.velocity.setZero();
    x.gyro_bias.setZero();
    x.acc_bias.setZero();
  }
  assert(ValidInitCandidate(p, Matrix3T::Identity()));
  Matrix3T r = Matrix3T::Identity();
  r(0, 0) = -1;
  assert(!ValidInitCandidate(p, r));
  p[1].velocity[0] = std::numeric_limits<float>::infinity();
  assert(!ValidInitCandidate(p, Matrix3T::Identity()));
  sba_imu::Pose anchor;
  anchor.w_from_imu = Isometry3T::Identity();
  anchor.velocity = Vector3T(1, 0, 0);
  anchor.gyro_bias.setZero();
  anchor.acc_bias.setZero();
  Vector3T g(0, -9.81f, 0), velocity;
  for (float dt : {.1f, .2f}) {
    sba_imu::IMUPreintegration pi(Vector3T::Zero(), Vector3T::Zero());
    pi.dT_s = dt;
    pi.dR.setIdentity();
    pi.dV = Vector3T(2 * dt, 9.81f * dt, 0);
    pi.dP.setZero();
    assert(InitVelocityAt(anchor, pi, 1000000000, 1000000000 + int64_t(std::llround(dt * 1e9)), g, velocity));
    assert(std::abs(velocity.x() - (1 + 2 * dt)) < 1e-5f);
    assert(std::abs(velocity.y()) < 1e-5f);
    assert(!InitVelocityAt(anchor, pi, 1000000000, 1500000000, g, velocity));
  }
}
