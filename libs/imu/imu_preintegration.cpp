
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

#include "imu/imu_preintegration.h"

#include <algorithm>
#include <Eigen/Geometry>

#include "common/rotation_utils.h"
#include "imu/linear_solver.h"
#include "imu/positive_matrix.h"

namespace cuvslam::sba_imu {
namespace {
Eigen::Matrix3d Hat(const Eigen::Vector3d& x) {
  Eigen::Matrix3d out;
  out << 0.,-x.z(),x.y(),x.z(),0.,-x.x(),-x.y(),x.x(),0.;
  return out;
}
Eigen::Matrix3d Exp(const Eigen::Vector3d& x) {
  const double n=x.norm();
  if (n<1e-12) return Eigen::Matrix3d::Identity()+Hat(x);
  return Eigen::AngleAxisd(n,x/n).toRotationMatrix();
}
struct Segment {
  Eigen::Matrix3d rotation;
  Eigen::Vector3d velocity,position;
};
Segment IntegrateSegment(const Eigen::Vector3d& a, const Eigen::Vector3d& w, double dt) {
  Segment result;
  result.rotation=Exp(w*dt);
  // Closed-form SO(3) integrals evaluated as stable Taylor series.
  const auto k=Hat(w*dt);
  Eigen::Vector3d term=a,dv=a,dp=.5*a;
  for (int i=1;i<=12;++i) {
    term=(k*term/i).eval();
    dv+=term/(i+1);
    dp+=term/((i+1)*(i+2));
  }
  result.velocity=dt*dv;
  result.position=dt*dt*dp;
  return result;
}
}  // namespace


IMUPreintegration::IMUPreintegration() { Initialize(Vector3T::Zero(), Vector3T::Zero()); }

IMUPreintegration::IMUPreintegration(const Vector3T& gyro_bias, const Vector3T& acc_bias) {
  Initialize(gyro_bias, acc_bias);
}

IMUPreintegration::IMUPreintegration(const imu::ImuCalibration& calib, const imu::ImuMeasurementStorage& storage,
                                     const Vector3T& gyro_bias, const Vector3T& acc_bias, int64_t start_time_ns,
                                     int64_t end_time_ns) {
  Initialize(gyro_bias, acc_bias);

  if (end_time_ns <= start_time_ns || storage.empty() || calib.frequency() <= 0.f) {
    return;
  }
  // A future-only sample must never be treated as coverage of an earlier frame.
  // A causal caller may not have the next sample yet: hold the last measurement
  // only over the remaining fraction of one nominal sampling period.
  auto first = storage.begin();
  auto last = std::prev(storage.end());
  const double max_tail_ns = 1.5e9 / calib.frequency();
  if (start_time_ns < first->time_ns || end_time_ns - last->time_ns > max_tail_ns) {
    return;
  }
  auto sample_at = [&](int64_t time_ns) {
    auto hi = std::lower_bound(storage.begin(), storage.end(), time_ns,
                               [](const auto& m, int64_t t) { return m.time_ns < t; });
    if (hi == storage.end()) {
      auto m = *last;
      m.time_ns = time_ns;
      return m;
    }
    if (hi->time_ns == time_ns || hi == storage.begin()) {
      return *hi;
    }
    auto lo = std::prev(hi);
    const float alpha = static_cast<float>(double(time_ns - lo->time_ns) / (hi->time_ns - lo->time_ns));
    imu::ImuMeasurement m;
    m.time_ns = time_ns;
    m.linear_acceleration = (1.f - alpha) * lo->linear_acceleration + alpha * hi->linear_acceleration;
    m.angular_velocity = (1.f - alpha) * lo->angular_velocity + alpha * hi->angular_velocity;
    return m;
  };
  auto previous = sample_at(start_time_ns);
  auto append_segment = [&](const imu::ImuMeasurement& current) {
    const float dt = static_cast<float>((current.time_ns - previous.time_ns) * 1e-9);
    if (dt <= 0.f) {
      return;
    }
    imu::ImuMeasurement midpoint;
    midpoint.time_ns = current.time_ns;
    midpoint.linear_acceleration = 0.5f * (previous.linear_acceleration + current.linear_acceleration);
    midpoint.angular_velocity = 0.5f * (previous.angular_velocity + current.angular_velocity);
    measurements_.push_back(midpoint);
    measurement_durations_s_.push_back(dt);
    update_state(calib, midpoint, dt);
    previous = current;
  };
  for (auto it = std::upper_bound(storage.begin(), storage.end(), start_time_ns,
                                  [](int64_t t, const auto&m) { return t < m.time_ns; });
       it != storage.end() && it->time_ns < end_time_ns; ++it) {
    append_segment(*it);
  }
  append_segment(sample_at(end_time_ns));
}

void IMUPreintegration::Initialize(const Vector3T& gyro_bias, const Vector3T& acc_bias) {
  gyro_bias_ = gyro_bias;
  acc_bias_ = acc_bias;

  gyro_bias_updated_ = gyro_bias;
  acc_bias_updated_ = acc_bias;

  gyro_bias_diff_.setZero();
  acc_bias_diff_.setZero();

  cov_matrix_.setZero();
  acc_random_walk_accum_cov_matrix_.setZero();
  gyro_random_walk_accum_cov_matrix_.setZero();

  info_matrix_ = std::nullopt;
  acc_random_walk_accum_info_matrix_ = std::nullopt;
  gyro_random_walk_accum_info_matrix_ = std::nullopt;

  dR.setIdentity();
  dV.setZero();
  dP.setZero();
  JRg.setZero();
  JVg.setZero();
  JVa.setZero();
  JPg.setZero();
  JPa.setZero();
  dT_s = 0;
}

void IMUPreintegration::update_state(const imu::ImuCalibration& calib, const imu::ImuMeasurement& m, float delta_t_s) {
  Matrix9T A = Matrix9T::Identity();
  Eigen::Matrix<float, 9, 3> B = Eigen::Matrix<float, 9, 3>::Zero();
  Eigen::Matrix<float, 9, 3> C = Eigen::Matrix<float, 9, 3>::Zero();

  Vector3T lin_acc = m.linear_acceleration - acc_bias_;
  Vector3T ang_vel = m.angular_velocity - gyro_bias_;

  const Eigen::Vector3d a=lin_acc.cast<double>(), w=ang_vel.cast<double>();
  const double dt=delta_t_s;
  const auto step=IntegrateSegment(a,w,dt);
  const auto step_rotation=step.rotation.cast<float>();
  const auto dv=step.velocity.cast<float>();
  const auto dp=step.position.cast<float>();
  A.block<3,3>(0,0)=step_rotation.transpose();
  A.block<3,3>(3,0)=-dR*SkewSymmetric(dv);
  A.block<3,3>(6,0)=-dR*SkewSymmetric(dp);
  A.block<3,3>(6,3)=Matrix3T::Identity()*delta_t_s;
  // Differentiate the selected discrete model in double precision. The
  // same input Jacobians drive the bias correction and covariance update.
  constexpr double epsilon=1e-4;
  for (int i=0;i<3;++i) {
    const Eigen::Vector3d e=epsilon*Eigen::Vector3d::Unit(i);
    const auto ap=IntegrateSegment(a+e,w,dt);
    const auto am=IntegrateSegment(a-e,w,dt);
    const auto wp=IntegrateSegment(a,w+e,dt);
    const auto wm=IntegrateSegment(a,w-e,dt);
    const Eigen::Matrix3d dr=step.rotation.transpose()*(wp.rotation-wm.rotation)/(2*epsilon);
    C.block<3,1>(0,i)=Eigen::Vector3d(dr(2,1)-dr(1,2),dr(0,2)-dr(2,0),dr(1,0)-dr(0,1)).cast<float>()*.5f;
    C.block<3,1>(3,i)=dR*((wp.velocity-wm.velocity)/(2*epsilon)).cast<float>();
    C.block<3,1>(6,i)=dR*((wp.position-wm.position)/(2*epsilon)).cast<float>();
    B.block<3,1>(3,i)=dR*((ap.velocity-am.velocity)/(2*epsilon)).cast<float>();
    B.block<3,1>(6,i)=dR*((ap.position-am.position)/(2*epsilon)).cast<float>();
  }
  JPg+=JVg*delta_t_s+A.block<3,3>(6,0)*JRg-C.block<3,3>(6,0);
  JVg+=A.block<3,3>(3,0)*JRg-C.block<3,3>(3,0);
  JRg=(A.block<3,3>(0,0)*JRg-C.block<3,3>(0,0)).eval();
  JPa+=JVa*delta_t_s-B.block<3,3>(6,0);
  JVa-=B.block<3,3>(3,0);
  dP+=dV*delta_t_s+dR*dp;
  dV+=dR*dv;
  dR=common::CalculateRotationFromSVD<float,3>(dR*step_rotation);
  const float period_ratio = calib.frequency() * delta_t_s;
  cov_matrix_ = A * cov_matrix_ * A.transpose() +
                (B * calib.acc_noise() * B.transpose() + C * calib.gyro_noise() * C.transpose()) / period_ratio;

  acc_random_walk_accum_cov_matrix_ += calib.acc_random_walk() * period_ratio;
  gyro_random_walk_accum_cov_matrix_ += calib.gyro_random_walk() * period_ratio;

  dT_s += delta_t_s;

  info_matrix_ = std::nullopt;
  acc_random_walk_accum_info_matrix_ = std::nullopt;
  gyro_random_walk_accum_info_matrix_ = std::nullopt;
}

void IMUPreintegration::IntegrateNewMeasurement(const imu::ImuCalibration& calib, const imu::ImuMeasurement& m) {
  float delta_t_s;
  if (measurements_.empty()) {
    delta_t_s = 1.f / calib.frequency();
  } else {
    delta_t_s = static_cast<float>(m.time_ns - measurements_.back().time_ns) * 1e-9f;
  }

  if (delta_t_s <= 0.f) {
    return;
  }
  measurements_.push_back(m);
  measurement_durations_s_.push_back(delta_t_s);

  update_state(calib, m, delta_t_s);
}

void IMUPreintegration::Reintegrate(const imu::ImuCalibration& calib) {
  Initialize(gyro_bias_updated_, acc_bias_updated_);

  for (size_t i = 0; i < measurements_.size(); i++) {
    update_state(calib, measurements_[i], measurement_durations_s_[i]);
  }
}

void IMUPreintegration::SetNewBias(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias) {
  gyro_bias_updated_ = new_gyro_bias;
  acc_bias_updated_ = new_acc_bias;

  gyro_bias_diff_ = gyro_bias_updated_ - gyro_bias_;
  acc_bias_diff_ = acc_bias_updated_ - acc_bias_;
}

void IMUPreintegration::GetDeltaBias(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias,
                                     Vector3T& gyro_bias_delta, Vector3T& acc_bias_delta) const {
  gyro_bias_delta = new_gyro_bias - gyro_bias_;
  acc_bias_delta = new_acc_bias - acc_bias_;
}

Matrix3T IMUPreintegration::GetDeltaRotation(const Vector3T& new_gyro_bias) const {
  Matrix3T R;
  math::Exp(R, JRg * (new_gyro_bias - gyro_bias_));
  return common::CalculateRotationFromSVD<float, 3>(dR * R);
}

Vector3T IMUPreintegration::GetDeltaVelocity(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias) const {
  Vector3T dbg = new_gyro_bias - gyro_bias_;
  Vector3T dba = new_acc_bias - acc_bias_;
  return dV + JVg * dbg + JVa * dba;
}

Vector3T IMUPreintegration::GetDeltaPosition(const Vector3T& new_gyro_bias, const Vector3T& new_acc_bias) const {
  Vector3T dbg = new_gyro_bias - gyro_bias_;
  Vector3T dba = new_acc_bias - acc_bias_;
  return dP + JPg * dbg + JPa * dba;
}

float IMUPreintegration::GetDeltaT_s() const { return dT_s; }

void IMUPreintegration::GetDeltaBias(Vector3T& gyro_bias_delta, Vector3T& acc_bias_delta) const {
  gyro_bias_delta = gyro_bias_diff_;
  acc_bias_delta = acc_bias_diff_;
}

const Vector3T& IMUPreintegration::GetOriginalGyroBias() const { return gyro_bias_; }

const Vector3T& IMUPreintegration::GetOriginalAccBias() const { return acc_bias_; }

void IMUPreintegration::GetUpdatedBias(Vector3T& gyro_bias_updated, Vector3T& acc_bias_updated) const {
  gyro_bias_updated = gyro_bias_updated_;
  acc_bias_updated = acc_bias_updated_;
}

void IMUPreintegration::InfoMatrix(Matrix9T& info) const {
  if (info_matrix_) {
    info = *info_matrix_;
    return;
  }

  if (empty()) {
    info.setZero();
    return;
  }
  if (!SolveLinearEquation(cov_matrix_, Matrix9T::Identity(), info) || !MakeMatrixPositive(info)) {
    info.setZero();
  }
  info_matrix_ = info;
}

void IMUPreintegration::InfoGyroRWMatrix(Matrix3T& info) const {
  if (gyro_random_walk_accum_info_matrix_) {
    info = *gyro_random_walk_accum_info_matrix_;
    return;
  }
  if (empty()) {
    info.setZero();
    return;
  }
  if (!SolveLinearEquation(gyro_random_walk_accum_cov_matrix_, Matrix3T::Identity(), info) ||
      !MakeMatrixPositive(info)) {
    info.setZero();
  }
  gyro_random_walk_accum_info_matrix_ = info;
}

void IMUPreintegration::InfoAccRWMatrix(Matrix3T& info) const {
  if (acc_random_walk_accum_info_matrix_) {
    info = *acc_random_walk_accum_info_matrix_;
    return;
  }
  if (empty()) {
    info.setZero();
    return;
  }
  if (!SolveLinearEquation(acc_random_walk_accum_cov_matrix_, Matrix3T::Identity(), info) ||
      !MakeMatrixPositive(info)) {
    info.setZero();
  }
  acc_random_walk_accum_info_matrix_ = info;
}

bool IMUPreintegration::empty() const { return measurements_.empty(); }

size_t IMUPreintegration::size() const { return measurements_.size(); }

}  // namespace cuvslam::sba_imu
