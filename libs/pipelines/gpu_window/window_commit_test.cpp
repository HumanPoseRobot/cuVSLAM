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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include "camera/camera.h"
#include "window_backend.h"
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
  {
    setenv("CUVSLAM_WINDOW_SOLVER", "schur_fp64", 1);
    camera::PinholeCameraModel cam(Vector2T(640, 480), Vector2T(320, 320), Vector2T(320, 240));
    camera::Rig rig;
    rig.num_cameras = 2;
    for (int c = 0; c < 2; ++c) {
      rig.camera_from_rig[c] = Isometry3T::Identity();
      rig.camera_from_rig[c].translation().x() = -.1f * c;
      rig.intrinsics[c] = &cam;
    }
    imu::ImuCalibration cal;
    imu::ImuMeasurementStorage storage(100);
    for (int i = 0; i <= 12; ++i)
      storage.push_back(imu::ImuMeasurement{1000000000LL + 5000000LL * i, Vector3T(0, 0, 9.81f), Vector3T::Zero()});
    sba_imu::Pose previous;
    previous.w_from_imu.translation() = Vector3T(.2f, -.1f, 1.5f);
    previous.info = Matrix15T::Identity() * 100;
    const auto tracked = previous;
    std::unordered_map<TrackId, Vector3T> points;
    std::vector<camera::Observation> obs;
    std::vector<pipelines::Landmark> fresh;
    for (int j = 0; j < 30; ++j) {
      Vector3T point = previous.w_from_imu * Vector3T(.15f * (j % 6 - 2), .1f * (j / 6 - 2), 3.f + .04f * j);
      points.emplace(j, point);
      fresh.push_back({TrackId(j), point});
      for (int c = 0; c < 2; ++c) {
        Vector3T pc = rig.camera_from_rig[c] * previous.w_from_imu.inverse() * point;
        obs.emplace_back(c, j, pc.hnormalized(), Matrix2T::Identity() * 10000);
      }
    }
    map::UnifiedMap map(20);
    map.add_keyframe(
        1000000000LL,
        map::State{previous.w_from_imu.inverse(), previous.velocity, previous.acc_bias, previous.gyro_bias}, {}, obs,
        fresh);
    pipelines::InertialPnPSettings settings;
    settings.min_observations = 12;
    settings.max_iteration = 3;
    settings.robustifier_scale = 2;
    Vector3T gravity(0, 0, -9.81f);
    sba_imu::WindowBackend backend;
    sba_imu::Pose result;
    result.w_from_imu.translation() = Vector3T(7, 8, 9);
    result.velocity = Vector3T(4, 5, 6);
    result.info = Matrix15T::Identity() * 37;
    const auto original = result;
    bool reached = false;
    const bool ok = backend.solve(
        {map, cal, rig, storage, points, obs, gravity, settings, previous, tracked, 1000000000LL, 1050000000LL,
         [&](const auto&, const auto&) {
           reached = true;
           return false;
         }},
        result);
    assert(reached);
    assert(!ok);
    assert((result.w_from_imu.matrix().array() == original.w_from_imu.matrix().array()).all());
    assert((result.velocity.array() == original.velocity.array()).all());
    assert((result.info.array() == original.info.array()).all());
    assert(backend.commit(map, cal, 1050000000LL));
    const auto state = map.get_recent_submap(1).consecutive_keyframes.back().keyframe->get_state();
    assert((state.rig_from_w.matrix().array() == previous.w_from_imu.inverse().matrix().array()).all());
    std::cout << "Rejected real window leaves caller pose, information and map unchanged\n";
  }
}
