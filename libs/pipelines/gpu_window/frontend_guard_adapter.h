#pragma once
#include "frontend_guard.h"
#include "camera/observation.h"
#include "imu/imu_sba_problem.h"
#include "common/log.h"
#include <set>
#include <string>
namespace cuvslam::pipelines::frontend_guard {
inline bool propagate(const sba_imu::Pose&left,sba_imu::Pose&pred){
 Matrix P;if(!inverse(left.info.cast<double>(),P))return false;const auto&pi=left.preintegration;
 Eigen::Matrix3d R=left.w_from_imu.linear().cast<double>();Eigen::Matrix3d dR=pi.GetDeltaRotation(left.gyro_bias).cast<double>();
 Matrix F=transition(R,dR,pi.GetDeltaPosition(left.gyro_bias,left.acc_bias).cast<double>(),pi.GetDeltaVelocity(left.gyro_bias,left.acc_bias).cast<double>(),pi.JRg.cast<double>(),pi.JPg.cast<double>(),pi.JPa.cast<double>(),pi.JVg.cast<double>(),pi.JVa.cast<double>(),pi.dT_s);
 Eigen::Matrix<double,15,9>L=Eigen::Matrix<double,15,9>::Zero();L.block<3,3>(0,0).setIdentity();L.block<3,3>(3,6)=dR.transpose();L.block<3,3>(6,3)=R;
 Matrix Q=L*pi.cov_matrix_.cast<double>()*L.transpose();Q.block<3,3>(9,9)+=pi.gyro_random_walk_accum_cov_matrix_.cast<double>();Q.block<3,3>(12,12)+=pi.acc_random_walk_accum_cov_matrix_.cast<double>();
 Matrix covariance=F*P*F.transpose()+Q;Matrix information;if(!inverse(covariance,information))return false;pred.info=information.cast<float>();return pred.info.allFinite();
}
template<class Landmarks>
inline bool validate(const imu::ImuCalibration&cal,const camera::Rig&rig,const Landmarks&landmarks,const std::vector<camera::Observation>&observations,const sba_imu::Pose&prediction,const sba_imu::Pose&candidate,int required,double dt,int64_t time_ns,const char*source,bool degraded=false,bool multi_frame=false){
 Quality quality;int pixel2_inliers=0;std::array<int,camera::Rig::kMaxCameras>counts{};std::array<Eigen::Vector2d,camera::Rig::kMaxCameras> lo,hi;for(int i=0;i<rig.num_cameras;++i){lo[i].setConstant(1e30);hi[i].setConstant(-1e30);}std::set<TrackId>unique;
 bool finite=candidate.w_from_imu.matrix().allFinite()&&candidate.velocity.allFinite()&&candidate.gyro_bias.allFinite()&&candidate.acc_bias.allFinite()&&candidate.info.allFinite();
 if(finite){
  const Isometry3T corrected=cal.rig_from_imu()*candidate.w_from_imu.inverse();
  for(const auto&o:observations){auto it=landmarks.find(o.id);if(it==landmarks.end()||o.cam_id>=rig.num_cameras)continue;++quality.total;const Vector3T pc=rig.camera_from_rig[o.cam_id]*corrected*it->second;if(!pc.allFinite()||pc.z()<=.05f)continue;const Vector2T error=(pc.hnormalized()-o.xy).cwiseProduct(rig.intrinsics[o.cam_id]->getFocal());if(!error.allFinite())continue;
       if(error.norm()<=2.f)++pixel2_inliers;
       const Eigen::Vector2d normalized=(pc.hnormalized()-o.xy).template cast<double>();
       const Eigen::Matrix2d information=9.*o.xy_info.template cast<double>();

       // Same whitened residual and 2-DoF rejection threshold as the window.
       // The pixel cap prevents arbitrarily weak information accepting gross errors.
       if(!observation_consistent(information,normalized,error.cast<double>()))continue;
       ++counts[o.cam_id];++quality.inliers;unique.insert(o.id);const Eigen::Vector2d xy=o.xy.cwiseProduct(rig.intrinsics[o.cam_id]->getFocal()).template cast<double>();lo[o.cam_id]=lo[o.cam_id].cwiseMin(xy);hi[o.cam_id]=hi[o.cam_id].cwiseMax(xy);}
 }
 quality.unique=unique.size();for(int i=0;i<rig.num_cameras;++i){if(counts[i]>=5&&(hi[i]-lo[i]).minCoeff()>=16.)++quality.supported_cameras;}
 Matrix P,C;Decision decision;Eigen::Matrix<double,15,1>delta=Eigen::Matrix<double,15,1>::Zero();
 if(finite&&inverse(prediction.info.cast<double>(),P)&&inverse(candidate.info.cast<double>(),C)){
  Eigen::Matrix3d rotation=prediction.w_from_imu.linear().cast<double>().transpose()*candidate.w_from_imu.linear().cast<double>();Eigen::AngleAxisd aa(rotation);delta.segment<3>(0)=aa.angle()*aa.axis();delta.segment<3>(3)=prediction.w_from_imu.linear().cast<double>().transpose()*(candidate.w_from_imu.translation()-prediction.w_from_imu.translation()).cast<double>();delta.segment<3>(6)=(candidate.velocity-prediction.velocity).cast<double>();delta.segment<3>(9)=(candidate.gyro_bias-prediction.gyro_bias).cast<double>();delta.segment<3>(12)=(candidate.acc_bias-prediction.acc_bias).cast<double>();Matrix T=Matrix::Identity();T.block<3,3>(0,0)=rotation;T.block<3,3>(3,3)=rotation;C=T*C*T.transpose();decision=assess(quality,required,delta,P,C,dt,degraded,multi_frame);
 }else{decision.reasons=Invalid;decision.score.fill(std::numeric_limits<double>::infinity());}
 std::string camera_counts;
 for (int cam = 0; cam < rig.num_cameras; ++cam) {
   if (cam != 0) camera_counts += ',';
   camera_counts += std::to_string(counts[cam]);
 }
 TraceMessage("FRONTEND_GATE t=%lld source=%s accept=%d reasons=%u total=%d inliers=%d unique=%d cameras=%d pixel2=%d counts=%s dp=%.9g dv=%.9g dbg=%.9g dba=%.9g scores=%.9g,%.9g,%.9g,%.9g,%.9g",(long long)time_ns,source,int(decision.accepted),decision.reasons,quality.total,quality.inliers,quality.unique,quality.supported_cameras,pixel2_inliers,camera_counts.c_str(),delta.segment<3>(3).norm(),delta.segment<3>(6).norm(),delta.segment<3>(9).norm(),delta.segment<3>(12).norm(),decision.score[0],decision.score[1],decision.score[2],decision.score[3],decision.score[4]);return decision.accepted;
}
}
