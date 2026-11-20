#pragma once
#include <cmath>
namespace cuvslam::pnp {
struct CauchyParameters { float weight, b, c; };
// cuNLS stores rho=b*log(1+c*s), rho'=1/(1+c*s).
// Thus its provided derivative is consistent only for b*c=1.
// Mode 1 preserves the legacy COST exactly, correcting its derivative through
// the outer weight. Mode 2 uses the intended Cauchy squared width chi2=9.21.
inline CauchyParameters VisualCauchyParameters(float weight,int mode) {
 const float chi2=9.21f,legacy_c=std::sqrt(chi2);
 if(mode==1)return {weight*legacy_c,1.f/legacy_c,legacy_c};
 if(mode==2)return {weight,chi2,1.f/chi2};
 return {weight,1.f,legacy_c};
}
}
