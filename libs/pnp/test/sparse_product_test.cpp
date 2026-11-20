#include <cunls/common/cusparse_helper.h>
#include <cunls/common/device_vector.h>
#include <cunls/minimizer/sparse_matrix.h>
#include <gtest/gtest.h>
#include <limits>
TEST(SparseProductContract, OutputHistoryDoesNotAffectOverwriteProduct) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  {
    cunls::cuSPARSEHandle handle;
    cunls::CSRSparseMatrix matrix;
    matrix.row_offsets = cunls::dvector<int>(std::vector<int>{0, 2, 4});
    matrix.col_ids = cunls::dvector<int>(std::vector<int>{0, 1, 0, 1});
    matrix.values = cunls::dvector<float>(std::vector<float>{1, 2, 3, 4});
    cunls::dvector<float> x(std::vector<float>{1, 2});
    cunls::dvector<uint8_t> buffer;
    for (bool knownDims : {false, true}) {
      for (bool poison : {false, true}) {
        cunls::dvector<float> result;
        if (poison) {
          result = cunls::dvector<float>(std::vector<float>(2, std::numeric_limits<float>::quiet_NaN()));
        }
        if (knownDims) {
          cunls::ComputeRHS(stream, handle.GetHandle(stream), matrix, 2, 2, 4, x, result, buffer);
        } else {
          cunls::ComputeRHS(stream, handle.GetHandle(stream), matrix, x, result, buffer);
        }
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        float output[2];
        ASSERT_EQ(cudaMemcpy(output, result.data(), 8, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_FLOAT_EQ(output[0], -7.f);
        EXPECT_FLOAT_EQ(output[1], -10.f);
      }
    }
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
