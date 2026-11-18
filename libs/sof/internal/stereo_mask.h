
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

#pragma once

#include "common/image.h"
#include "common/vector_2t.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include <stdexcept>
#endif

namespace cuvslam::sof {

// The mask can have a different size and uses the same nearest-pixel
// convention as temporal tracking. Stage device masks once per invocation.
class StereoMask {
public:
  StereoMask(const ImageSource& mask, const ImageMeta& meta) : meta_(meta), pixels_(static_cast<const uint8_t*>(mask.data)) {
#ifdef USE_CUDA
    if (pixels_ && mask.memory_type == ImageSource::Device) {
      device_copy_.resize(meta.mask_shape.height, meta.mask_shape.width);
      const auto copied = cudaMemcpy2D(device_copy_.data(), meta.mask_shape.width, pixels_, mask.pitch,
                              meta.mask_shape.width, meta.mask_shape.height, cudaMemcpyDeviceToHost);
      if (copied != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(copied));
      }
      pixels_ = device_copy_.data();
    }
#endif
  }

  bool accepts(const Vector2T& uv) const {
    if (!uv.allFinite() || uv.x() < 0 || uv.y() < 0 ||
        uv.x() >= meta_.shape.width || uv.y() >= meta_.shape.height) {
      return false;
    }
    if (!pixels_) {
      return true;
    }
    const auto& m = meta_.mask_shape;
    const int x = static_cast<int>(uv.x()) * static_cast<int64_t>(m.width) / meta_.shape.width;
    const int y = static_cast<int>(uv.y()) * static_cast<int64_t>(m.height) / meta_.shape.height;
    return pixels_[static_cast<int64_t>(y) * m.width + x] == 0;
  }

private:
  const ImageMeta& meta_;
  const uint8_t* pixels_;
  ImageMatrix<uint8_t> device_copy_;
};
}  // namespace cuvslam::sof
