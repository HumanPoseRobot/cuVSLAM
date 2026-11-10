#include "frontend_guard.h"
#include <cassert>
#include <iostream>
#include <Eigen/Geometry>
using namespace cuvslam::pipelines::frontend_guard;
int main(){
 double seconds=0;
 for(int i=0;i<30;++i){assert(prediction_budget(seconds,1./30,true));}
 assert(seconds==0);
 for(int i=0;i<15;++i){assert(prediction_budget(seconds,1./30,false));}
 assert(!prediction_budget(seconds,1./30,false));
 assert(prediction_budget(seconds,1./30,true)&&seconds==0);
 assert(!prediction_budget(seconds,0,false));
 {
  Matrix uncertain=Matrix::Identity()*.001;
  uncertain.block<3,3>(6,6)=Eigen::Matrix3d::Identity()*.08;
  Eigen::Matrix<double,15,1> delta=Eigen::Matrix<double,15,1>::Zero();
  Quality strong{487,237,237,4};
  delta[6]=.42;
  assert(assess(strong,25,delta,uncertain,uncertain,1./30).accepted);
  delta[3]=.06;
  assert(assess(strong,25,delta,uncertain,uncertain,1./30).reasons&Velocity);
  delta.setZero();
  Quality single{409,33,33,1};
  assert(assess(single,25,delta,uncertain,uncertain,1./30).accepted);
  assert(assess(single,25,delta,uncertain,uncertain,1./30,false,true).reasons&Observations);
 }
 Quality q{200,130,70,2};Matrix p=Matrix::Identity()*.001;Eigen::Matrix<double,15,1>x=Eigen::Matrix<double,15,1>::Zero();
 assert(assess(q,25,x,p,p,1./30).accepted);
 Quality sparse_static{223,31,31,2};assert(assess(sparse_static,25,x,p,p,1./30).accepted);
 Quality supported_consensus{229,21,21,2};assert(assess(supported_consensus,25,x,p,p,1./30).accepted);
 Quality many_outliers{2000,21,21,2};assert(assess(many_outliers,25,x,p,p,1./30).accepted);
 Quality single_camera{200,20,20,1};assert(assess(single_camera,25,x,p,p,1./30).reasons&Observations);
 assert(assess(single_camera,25,x,p,p,1./30,true).accepted);
 Quality repeated_landmarks{200,30,5,2};assert(assess(repeated_landmarks,25,x,p,p,1./30).reasons&Observations);
 Eigen::Matrix2d image_information=10000.*Eigen::Matrix2d::Identity();
 assert(observation_consistent(image_information,Eigen::Vector2d(.025,0),Eigen::Vector2d(2.5,0)));
 assert(!observation_consistent(image_information,Eigen::Vector2d(.031,0),Eigen::Vector2d(3.1,0)));
 assert(!observation_consistent(-image_information,Eigen::Vector2d(.001,0),Eigen::Vector2d(.1,0)));
 assert(!observation_consistent(Eigen::Matrix2d::Identity(),Eigen::Vector2d(.01,0),Eigen::Vector2d(9,0)));
 Quality short_consensus{222,17,17,2};assert(!assess(short_consensus,25,x,p,p,1./30).accepted);assert(assess(short_consensus,25,x,p,p,1./30,true).accepted);
 x[3]=.021;Matrix precise=Matrix::Identity()*1e-6;assert(assess(short_consensus,25,x,precise,p,1./30,true).reasons&Position);
 assert(assess(short_consensus,25,x,p,p,1./30,true).accepted);x[3]=.10;assert(assess(short_consensus,25,x,p,p,1./30,true).reasons&Position);x.setZero();
 Quality weak{345,8,6,1};assert(!assess(weak,25,x,p,p,1./30,true).accepted);assert(assess(weak,25,x,p,p,1./30).reasons&Observations);
 x[6]=.485936;assert(assess(q,25,x,p,p,1./30).reasons&Velocity);
 x.setZero();x[12]=.76977;assert(assess(q,25,x,p,p,1./30).reasons&AccBias);
 x.setZero();x[9]=.03;assert(assess(q,25,x,p,p,1./30).reasons&GyroBias);
 x.setZero();x[3]=1.;assert(assess(q,25,x,p,p,1./30).reasons&Position);
 x.setZero();x[12]=.2;Matrix uncertain=p;uncertain.block<3,3>(12,12)=Eigen::Matrix3d::Identity()*.01;
 assert(assess(q,25,x,uncertain,p,1./30).reasons&AccBias);
 assert(assess(q,25,x,uncertain,p,1./30,false,true).accepted);
 x[12]=.4;assert(assess(q,25,x,uncertain,p,1./30,false,true).reasons&AccBias);
 x[12]=.2;assert(assess(q,25,x,p,p,1./30,false,true).reasons&AccBias);
 x.setZero();Matrix bad=p;bad(0,0)=-1;assert(assess(q,25,x,bad,p,1./30).reasons&Invalid);
 // An arbitrarily fast consistent motion has zero innovation and is accepted.
 assert(assess(q,25,x,p,p,1./30).accepted);
 auto I=Eigen::Matrix3d::Identity();auto Z=Eigen::Matrix3d::Zero();auto z=Eigen::Vector3d::Zero();Matrix F=transition(I,I,z,z,Z,Z,Z,Z,Z,.1);Matrix grown=F*p*F.transpose();assert(grown(3,3)>p(3,3));assert(std::abs(F(3,6)-.1)<1e-12);

 const Eigen::Matrix3d R=Eigen::AngleAxisd(.4,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix(),dR=Eigen::AngleAxisd(.2,Eigen::Vector3d(2,-1,1).normalized()).toRotationMatrix();
 const Eigen::Vector3d dP(.1,-.02,.03),dV(.2,.1,-.3);const Eigen::Matrix3d Jr=.02*Eigen::Matrix3d::Identity(),Jp=.003*Eigen::Matrix3d::Identity(),Jpa=-.005*Eigen::Matrix3d::Identity(),Jv=.01*Eigen::Matrix3d::Identity(),Jva=-.1*Eigen::Matrix3d::Identity();
 Matrix analytic=transition(R,dR,dP,dV,Jr,Jp,Jpa,Jv,Jva,.1),numeric;
 auto exp=[](const Eigen::Vector3d& v)->Eigen::Matrix3d {double a=v.norm();return a<1e-20?Eigen::Matrix3d::Identity():Eigen::AngleAxisd(a,v/a).toRotationMatrix();};
 auto predict=[&](const Eigen::Matrix<double,15,1>& e){Eigen::Matrix<double,15,1> y=e;Eigen::Matrix3d rotated=R*exp(e.segment<3>(0));Eigen::AngleAxisd aa((R*dR).transpose()*rotated*dR*exp(Jr*e.segment<3>(9)));y.segment<3>(0)=aa.angle()*aa.axis();y.segment<3>(3)=(R*dR).transpose()*(R*e.segment<3>(3)+.1*e.segment<3>(6)+rotated*(dP+Jp*e.segment<3>(9)+Jpa*e.segment<3>(12))-R*dP);y.segment<3>(6)=e.segment<3>(6)+rotated*(dV+Jv*e.segment<3>(9)+Jva*e.segment<3>(12))-R*dV;return y;};
 for(int c=0;c<15;++c){Eigen::Matrix<double,15,1>e=Eigen::Matrix<double,15,1>::Zero();e[c]=1e-6;numeric.col(c)=(predict(e)-predict(-e))/(2e-6);}
 double error=(analytic-numeric).cwiseAbs().maxCoeff();assert(error<1e-8);std::cout<<"prediction Jacobian finite-difference max error "<<error<<"\n";
 Matrix inverse_cov;assert(inverse(grown,inverse_cov));std::cout<<"frontend guard tests passed\n";
}
