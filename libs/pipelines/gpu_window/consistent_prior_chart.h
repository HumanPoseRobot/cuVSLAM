#pragma once
#include "window_math.h"
namespace gpu_window::posterior {
// Extend the ordinary positive-cosine SO(3) logarithm continuously to inherited
// FP32 matrices. At zero antisymmetric part its derivative is y/(2*cos),
// not y/2 unless the input is exactly orthogonal. Keep the nonzero-angle and
// near-pi branches, biases, translations and gravity charts unchanged.
template <class T>
void PriorChartDifference(const State& state, const State& anchor, const T* x, T* residual) {
  difference(state, anchor, x, residual);
  T rotation[9], position[3], velocity[3], transpose[9], relative[9];
  pose(state, x, rotation, position, velocity);
  trans(anchor.R, transpose);
  mul(transpose, rotation, relative);
  const T y[3] = {relative[7]-relative[5], relative[2]-relative[6], relative[3]-relative[1]};
  const T squared_sine = .25*(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);
  const T cosine = .5*(relative[0]+relative[4]+relative[8]-1.);
  if (-cosine < 0. && squared_sine < 1e-20) {
    // Taylor series atan(s/c)/(2*s) = 1/(2*c) - s^2/(6*c^3) + O(s^4).
    const T coefficient = .5/cosine-squared_sine/(6.*cosine*cosine*cosine);
    for (int k=0;k<3;++k) { residual[k] = coefficient*y[k]; }
  }
}
}  // namespace gpu_window::posterior
