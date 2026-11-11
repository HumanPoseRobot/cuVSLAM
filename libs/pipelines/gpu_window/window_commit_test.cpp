#include <cassert>
#include <limits>
#include "window_commit_validation.h"
using namespace cuvslam;
int main() {
  map::State a{Isometry3T::Identity(), Vector3T::Zero(), Vector3T::Zero(), Vector3T::Zero()};
  assert(sba_imu::SameCommitState(a, a));
  auto b = a;
  b.velocity.x() = .01f;
  assert(!sba_imu::SameCommitState(a, b));
  b = a;
  b.acc_bias.z() = .01f;
  assert(!sba_imu::SameCommitState(a, b));
  b = a;
  b.gyro_bias.y() = .001f;
  assert(!sba_imu::SameCommitState(a, b));
  b = a;
  b.rig_from_w.translation().x() = .01f;
  assert(!sba_imu::SameCommitState(a, b));
  b = a;
  b.velocity.x() = std::numeric_limits<float>::quiet_NaN();
  assert(!sba_imu::SameCommitState(b, a));
}
