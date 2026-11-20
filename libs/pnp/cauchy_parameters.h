#pragma once
#include <cmath>
#include <cstdlib>
namespace cuvslam::pnp {
struct CauchyParameters {
  float weight, b, c;
};
// Explicit visual-loss parameterizations. The corrected cuNLS robustifier
// evaluates rho, rho' and rho'' consistently for any positive b,c.
// Mode 1 preserves the legacy cost; mode 2 uses squared width chi2=9.21.
inline CauchyParameters VisualCauchyParameters(float weight, int mode) {
  const float chi2 = 9.21f, legacy_c = std::sqrt(chi2);
  if (mode == 1) {
    return {weight * legacy_c, 1.f / legacy_c, legacy_c};
  }
  if (mode == 2) {
    return {weight, chi2, 1.f / chi2};
  }
  return {weight, 1.f, legacy_c};
}
}  // namespace cuvslam::pnp
