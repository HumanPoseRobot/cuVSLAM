
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

#include <cfloat>
#include <cmath>
#include "cuda_modules/cuda_kernels/cuda_kernels.h"
namespace cuvslam::cuda {
namespace {
__device__ bool inside(float x, float y, Size size) {
  return x >= 0.f && y >= 0.f && x < float(size.width) && y < float(size.height);
}
__global__ void check_kernel(cudaTextureObject_t source, cudaTextureObject_t target, Size source_size, Size target_size,
                             const TrackData* original, TrackData* calibrated, const int* indices, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  auto& trial = calibrated[i];
  trial.cache_index = -1;  // This candidate still needs calibrated tracking.
  const auto& accepted = original[indices[i]];
  if (!accepted.track_status) {
    return;
  }
  float a[NCC_DIM * NCC_DIM], b[NCC_DIM * NCC_DIM], ma = 0.f, mb = 0.f;
  const auto A = trial.initial_guess_map;
  int n = 0;
  for (int y = 0; y < NCC_DIM; ++y) {
    for (int x = 0; x < NCC_DIM; ++x) {
      const float dx = float(x - (NCC_DIM - 1) / 2), dy = float(y - (NCC_DIM - 1) / 2);
      const float sx = trial.track.x + dx, sy = trial.track.y + dy;
      const float tx = accepted.track.x + A.x * dx + A.y * dy, ty = accepted.track.y + A.z * dx + A.w * dy;
      if (!inside(sx, sy, source_size) || !inside(tx, ty, target_size)) {
        return;
      }
      a[n] = tex2D<float>(source, sx, sy);
      b[n] = tex2D<float>(target, tx, ty);
      ma += a[n];
      mb += b[n];
      ++n;
    }
  }
  ma /= n;
  mb /= n;
  float va = 0.f, vb = 0.f, cross = 0.f;
  for (int k = 0; k < n; ++k) {
    const float da = a[k] - ma, db = b[k] - mb;
    va += da * da;
    vb += db * db;
    cross += da * db;
  }
  const float product = va * vb;
  const float ncc = product > FLT_EPSILON ? cross / sqrtf(product) : 0.f;
  trial.ncc = ncc;
  if (isfinite(ncc) && ncc > accepted.ncc_threshold) {
    // ST skips this candidate, and the host retains the ORIGINAL LK output,
    // including its coordinates and information matrix, without re-estimation.
    trial.cache_index = -2;
    trial.search_radius_px = -1.f;
  }
}
}  // namespace
cudaError_t calibrated_patch_check(cudaTextureObject_t primary, cudaTextureObject_t secondary, Size primary_size,
                                   Size secondary_size, const TrackData* original, TrackData* calibrated,
                                   const int* indices, int count, cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  check_kernel<<<(count + 63) / 64, 64, 0, stream>>>(primary, secondary, primary_size, secondary_size, original,
                                                     calibrated, indices, count);
  return cudaGetLastError();
}
}  // namespace cuvslam::cuda
