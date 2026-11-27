#pragma once
#include <Eigen/Dense>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <cmath>
namespace block_lm_metric {
inline bool Enabled(){const char*p=std::getenv("VIO_LM_METRIC");if(!p||!std::strcmp(p,"component_diagonal"))return false;if(!std::strcmp(p,"block_trace"))return true;throw std::invalid_argument("unknown VIO_LM_METRIC");}
// Within each equal-unit orthonormal variable block, trace(H_ii)/dim times I
// is equivariant under a change of frame. This changes only the trust-region
// metric, not the sensor objective, marginal posterior or acceptance gates.
template<class Derived> inline double MeanSafe(const Eigen::MatrixBase<Derived>&raw){
 if(!raw.allFinite())throw std::invalid_argument("nonfinite LM curvature");
 const double maximum=raw.cwiseAbs().maxCoeff();if(maximum==0.)return 0.;
 double mean=(raw.array()/maximum).sum()/double(raw.size());
 const double bound=64.*std::numeric_limits<double>::epsilon()*raw.size();
 if(mean < -bound)throw std::invalid_argument("negative LM block curvature");
 if(mean<0.)mean=0.;return maximum*mean;
}
inline Eigen::VectorXd Diagonal(const Eigen::VectorXd&raw){if(!Enabled())return raw;if(raw.size()<2||(raw.size()-2)%3)throw std::invalid_argument("invalid navigation/map/gravity block layout");Eigen::VectorXd result=raw;for(int k=0;k<raw.size()-2;k+=3)result.segment<3>(k).setConstant(MeanSafe(raw.segment<3>(k)));result.tail<2>().setConstant(MeanSafe(raw.tail<2>()));return result;}
inline Eigen::Vector3d Point(const Eigen::Vector3d&raw){return Enabled()?Eigen::Vector3d::Constant(MeanSafe(raw)):raw;}
}
