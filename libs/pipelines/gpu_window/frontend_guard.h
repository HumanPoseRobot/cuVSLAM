#pragma once
#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
namespace cuvslam::pipelines::frontend_guard {
using Matrix = Eigen::Matrix<double,15,15>;
struct Quality {int total=0,inliers=0,unique=0,supported_cameras=0;};
struct Decision {bool accepted=false;unsigned reasons=0;std::array<double,5> score{};};
enum Reason : unsigned { Invalid=1, Observations=2, Position=4, Velocity=8, GyroBias=16, AccBias=32, Rotation=64 };
inline bool inverse(const Matrix& in,Matrix& out){
 if(!in.allFinite())return false;
 Eigen::LDLT<Matrix> f((in+in.transpose())*.5);
 if(f.info()!=Eigen::Success||!f.isPositive()||f.vectorD().minCoeff()<=0)return false;
 out=f.solve(Matrix::Identity());out=(out+out.transpose()).eval()*.5;return out.allFinite();
}
inline Eigen::Matrix3d skew(const Eigen::Vector3d& x){Eigen::Matrix3d a;a<<0,-x.z(),x.y(),x.z(),0,-x.x(),-x.y(),x.x(),0;return a;}
// All covariance coordinates: body rotation/body translation, world velocity, IMU biases.
inline Matrix transition(const Eigen::Matrix3d&R,const Eigen::Matrix3d&dR,const Eigen::Vector3d&dP,const Eigen::Vector3d&dV,const Eigen::Matrix3d&JRg,const Eigen::Matrix3d&JPg,const Eigen::Matrix3d&JPa,const Eigen::Matrix3d&JVg,const Eigen::Matrix3d&JVa,double dt){
 Matrix F=Matrix::Identity();const Eigen::Matrix3d Rt=(R*dR).transpose();
 F.block<3,3>(0,0)=dR.transpose();F.block<3,3>(0,9)=JRg;
 F.block<3,3>(3,0)=-dR.transpose()*skew(dP);F.block<3,3>(3,3)=dR.transpose();F.block<3,3>(3,6)=dt*Rt;F.block<3,3>(3,9)=dR.transpose()*JPg;F.block<3,3>(3,12)=dR.transpose()*JPa;
 F.block<3,3>(6,0)=-R*skew(dV);F.block<3,3>(6,9)=R*JVg;F.block<3,3>(6,12)=R*JVa;return F;
}
inline std::array<double,3> correction_limits(double dt){double s=std::sqrt(dt/(1./30.));return {.35*s,.015*s,.15*s};}
inline bool observation_consistent(const Eigen::Matrix2d& information,const Eigen::Vector2d& normalized_error,const Eigen::Vector2d& pixel_error){
 if(!information.allFinite()||!normalized_error.allFinite()||!pixel_error.allFinite()||pixel_error.norm()>8.)return false;
 Eigen::Matrix2d symmetric=(information+information.transpose())*.5;
 Eigen::LDLT<Eigen::Matrix2d> factor(symmetric);
 if(factor.info()!=Eigen::Success||!factor.isPositive()||factor.vectorD().minCoeff()<=0)return false;
 const double score=normalized_error.dot(symmetric*normalized_error);return std::isfinite(score)&&score<=9.21;
}
inline bool prediction_budget(double& seconds,double dt,bool visual_update){
 if(visual_update){seconds=0;return true;}
 if(!std::isfinite(dt)||dt<=0){seconds=std::numeric_limits<double>::infinity();return false;}
 seconds+=dt;return seconds<=.5;
}
inline Decision assess(const Quality&q,int required,const Eigen::Matrix<double,15,1>&delta,const Matrix&pred_cov,const Matrix&candidate_cov,double dt,bool degraded=false,bool multi_frame=false){
 Decision d;d.score.fill(std::numeric_limits<double>::infinity());Matrix scratch;
 if(!(dt>0&&dt<=1)||!delta.allFinite()||!inverse(pred_cov,scratch)||!inverse(candidate_cov,scratch)){d.reasons=Invalid;return d;}
 // The optimizer's minimum candidate count is not the post-fit consensus size.
 // Absolute consensus, distinct landmarks and two spatially supported cameras
 // determine usable constraints; unrelated outliers do not dilute that count.
 const int minimum_inliers=degraded?12:std::max(20,(required+1)/2);
 const bool strong_single_camera = !degraded && !multi_frame &&
     q.unique >= 30 && q.supported_cameras >= 1;
 if(q.inliers<minimum_inliers||q.unique<std::max(12,minimum_inliers/2)||
    (q.supported_cameras<(degraded?1:2) && !strong_single_camera))d.reasons|=Observations;
 const std::array<double,5> floor{.0025,.005,.03,.001,.03};
 const std::array<unsigned,5> reason{Rotation,Position,Velocity,GyroBias,AccBias};
 for(int i=0;i<5;++i){Eigen::Matrix3d S=4.*(pred_cov.block<3,3>(3*i,3*i)+candidate_cov.block<3,3>(3*i,3*i))+floor[i]*floor[i]*Eigen::Matrix3d::Identity();Eigen::LDLT<Eigen::Matrix3d>f(S);const Eigen::Vector3d x=delta.segment<3>(3*i);d.score[i]=x.dot(f.solve(x));if(!std::isfinite(d.score[i])||d.score[i]>16.27)d.reasons|=reason[i];}
 // Trust-region limits apply to estimator corrections, never to platform speed.
 // Scale with sqrt(dt) like process uncertainty. These are engineering bounds,
 // not a chi-square assertion for correlated prior/posterior estimates.
 auto limits=correction_limits(dt);
 // A multi-keyframe posterior can legitimately resolve uncertainty accumulated
 // before this camera frame. Do not treat that correction as one-frame bias
 // random walk. Bound it by the predicted marginal uncertainty as well.
 if(multi_frame){
  for(int i=0;i<3;++i){
   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(pred_cov.block<3,3>(6+3*i,6+3*i));
   if(eig.info()!=Eigen::Success){d.reasons|=Invalid;continue;}
   limits[i]=std::max(limits[i],3.*std::sqrt(std::max(0.,eig.eigenvalues().maxCoeff())));
  }
 }
 // Only a well-supported, position-consistent foreground update may
 // resolve a larger velocity error left by uncertain IMU initialization.
 if(!degraded && !multi_frame && q.unique>=40 && q.supported_cameras>=2 &&
    delta.segment<3>(3).norm()<=.05 && !(d.reasons & (Position|Rotation))){
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(pred_cov.block<3,3>(6,6));
  if(eig.info()==Eigen::Success)
   limits[0]=std::max(limits[0],3.*std::sqrt(std::max(0.,eig.eigenvalues().maxCoeff())));
 }
 if(delta.segment<3>(6).norm()>limits[0])d.reasons|=Velocity;
 if(delta.segment<3>(9).norm()>limits[1])d.reasons|=GyroBias;
 if(delta.segment<3>(12).norm()>limits[2])d.reasons|=AccBias;
 if(degraded){
  // Short tracking-only recovery: fewer well-distributed landmarks are usable
  // only for small VI corrections. Never grant normal map-update authority.
  const std::array<double,4> minimum{.02,.15,.005,.05};
  const std::array<unsigned,4> weak_reason{Position,Velocity,GyroBias,AccBias};
  for(int i=0;i<4;++i){
   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(pred_cov.block<3,3>(3+3*i,3+3*i));
   if(eig.info()!=Eigen::Success){d.reasons|=Invalid;continue;}
   const double bound=std::max(minimum[i],3.*std::sqrt(std::max(0.,eig.eigenvalues().maxCoeff())));
   if(delta.segment<3>(3+3*i).norm()>bound)d.reasons|=weak_reason[i];
  }
 }
 d.accepted=d.reasons==0;return d;
}
}
