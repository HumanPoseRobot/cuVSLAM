#include <chrono>
#include <thread>
#include "sof/image_context.h"

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

#include "common/include_gtest.h"
#include "common/types.h"
#include "sof/sof_config.h"
#include "utils/image_loader.h"

namespace test::sof {
using namespace cuvslam;
using namespace cuvslam::sof;

TEST(ImageTest, GF2TGradientMapTest) {
  const std::string testDataFolder = CUVSLAM_TEST_ASSETS;
  const std::string filePath = testDataFolder + "sof/left.png";
  utils::ImageLoaderT imgLoad;
  ASSERT_TRUE(imgLoad.load(filePath));
  ImagePyramidT image(imgLoad.getImage().cast<float>());
  ASSERT_NE(image.buildPyramid(), 0);

  Settings sof_settings;

  GradientPyramidT gradMapBuilder;

  EXPECT_TRUE(gradMapBuilder.set(image));  // horizontal_ == false

  for (int level = 0; level < gradMapBuilder.getLevelsCount(); ++level) {
    const ImageMatrixT &in = image[level];
    const Index n_cols = in.cols();
    const Index n_rows = in.rows();
    const ImageMatrixT &out_x = gradMapBuilder.gradX()[level];
    const ImageMatrixT &out_y = gradMapBuilder.gradY()[level];

    // check shapes
    EXPECT_EQ(n_cols, out_x.cols());
    EXPECT_EQ(n_rows, out_x.rows());

    EXPECT_EQ(n_cols, out_y.cols());
    EXPECT_EQ(n_rows, out_y.rows());

    if (gradMapBuilder.isGradientsLazyEvaluated(level)) {
      // check every level for zero-border
      EXPECT_TRUE((out_x.topRows(3).array() == 0).all());
      EXPECT_TRUE((out_x.bottomRows(3).array() == 0).all());
      EXPECT_TRUE((out_x.leftCols(3).array() == 0).all());
      EXPECT_TRUE((out_x.rightCols(3).array() == 0).all());

      EXPECT_TRUE((out_y.topRows(3).array() == 0).all());
      EXPECT_TRUE((out_y.bottomRows(3).array() == 0).all());
      EXPECT_TRUE((out_y.leftCols(3).array() == 0).all());
      EXPECT_TRUE((out_y.rightCols(3).array() == 0).all());
    }
  }

  EXPECT_TRUE(gradMapBuilder.set(image, true));  // horizontal_ == true

  for (int level = 0; level < gradMapBuilder.getLevelsCount(); ++level) {
    const ImageMatrixT &in = image[level];
    const Index n_cols = in.cols();
    const Index n_rows = in.rows();
    const ImageMatrixT &out_x = gradMapBuilder.gradX()[level];

    // check shapes
    EXPECT_EQ(n_cols, out_x.cols());
    EXPECT_EQ(n_rows, out_x.rows());

    if (gradMapBuilder.isGradientsLazyEvaluated(level)) {
      // check every level for zero-border
      EXPECT_TRUE((out_x.topRows(3).array() == 0).all());
      EXPECT_TRUE((out_x.bottomRows(3).array() == 0).all());
      EXPECT_TRUE((out_x.leftCols(3).array() == 0).all());
      EXPECT_TRUE((out_x.rightCols(3).array() == 0).all());
    }
  }
}

TEST(ImageTest, Visual_GF2TFeatureDetectorTest) {
  const std::string testDataFolder = CUVSLAM_TEST_ASSETS;

  utils::ImageLoaderT imgLoad;
  ASSERT_TRUE(imgLoad.load(testDataFolder + "sof/left.png"));

  ImagePyramidT image(imgLoad.getImage().cast<float>());
  ASSERT_NE(image.buildPyramid(), 0);
  EXPECT_GT(image.getLevelsCount(), 0);

  Settings sof_settings;
  GradientPyramidT gradientBuilder;
  EXPECT_TRUE(gradientBuilder.set(image));
}

}  // namespace test::sof

#ifdef USE_CUDA
TEST(ImageContextGPU, CrossStreamCacheWaitsForPixelsAndProvidesCompleteGradients) {
  using namespace cuvslam;
  constexpr int width = 128, height = 128;
  sof::ImageContext context(ImageShape{width, height}, true, false);
  ImageMeta meta;
  meta.shape = {width, height};
  context.set_image_meta(meta);
  std::vector<uint8_t> pixels(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      pixels[y * width + x] = uint8_t(20 + y + x % 10);
    }
  }
  ImageSource source;
  source.data = pixels.data();
  source.image_encoding = ImageEncoding::MONO8;
  cuda::Stream producer, consumer, reader;
  CUDA_CHECK(cudaLaunchHostFunc(
      producer.get_stream(), [](void *) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }, nullptr));
  ASSERT_TRUE(context.build_gpu_image_pyramid(source, false, producer.get_stream()));
  ASSERT_TRUE(context.build_gpu_image_pyramid(source, false, consumer.get_stream()));
  // A preceding X-only user must not poison an XY user's cached gradients.
  ASSERT_TRUE(context.build_gpu_gradient_pyramid(true, consumer.get_stream()));
  ASSERT_TRUE(context.build_gpu_gradient_pyramid(false, reader.get_stream()));
  const auto &actual = context.gpu_gradient_pyramid().gradY()[0];
  std::vector<float> values(width * height), expected(width * height);
  CUDA_CHECK(cudaMemcpy2DAsync(values.data(), width * sizeof(float), actual.ptr(), actual.pitch(),
                               width * sizeof(float), height, cudaMemcpyDeviceToHost, reader.get_stream()));
  CUDA_CHECK(cudaStreamSynchronize(reader.get_stream()));
  CUDA_CHECK(cudaStreamSynchronize(producer.get_stream()));
  cuda::GPUGradientPyramid reference(width, height);
  auto stream = reader.get_stream();
  ASSERT_TRUE(reference.set(context.gpu_image_pyramid(), stream, false));
  const auto &gy = reference.gradY()[0];
  CUDA_CHECK(cudaMemcpy2DAsync(expected.data(), width * sizeof(float), gy.ptr(), gy.pitch(), width * sizeof(float),
                               height, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ASSERT_GT(std::abs(expected[(height / 2) * width + width / 2]), .01f);
  for (int y = 8; y < height - 8; ++y) {
    for (int x = 8; x < width - 8; ++x) {
      ASSERT_NEAR(values[y * width + x], expected[y * width + x], 1e-6f);
    }
  }
}
#endif
