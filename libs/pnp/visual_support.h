#pragma once
#include <Eigen/Dense>
#include <limits>
namespace cuvslam::pnp {
// Only fitted independent geometric evidence determines support. Unrelated
// outliers elsewhere in the rig must not dilute a usable camera's consensus.
inline bool HasObservableVisualSupport(const Eigen::MatrixXd& jacobian,size_t inliers,size_t required) {
 if(inliers<required||jacobian.rows()!=2*Eigen::Index(inliers)||jacobian.cols()!=6||!jacobian.allFinite()){return false;}
 Eigen::MatrixXd normalized=jacobian;
 for(int i=0;i<6;++i){const double scale=normalized.col(i).norm();if(!(scale>0)||!std::isfinite(scale)){return false;}normalized.col(i)/=scale;}
 Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(normalized);
 qr.setThreshold(64.*std::numeric_limits<double>::epsilon());return qr.rank()==6;
}
}
