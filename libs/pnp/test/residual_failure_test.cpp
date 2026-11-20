#include <cunls/common/cublas_helper.h>
#include <cunls/common/device_vector.h>
#include <cunls/factor/factor_batch.h>
#include <cunls/factor/information_factor_batch.h>
#include <cunls/factor/sized_factor_batch.h>
#include <cunls/factor/weighted_factor_batch.h>
#include <cunls/minimizer/levenberg_marquardt_minimizer.h>
#include <cunls/minimizer/problem.h>
#include <cunls/robustifier/loss_function_batch.h>
#include <cunls/robustifier/scaled_loss_function_batch.h>
#include <cunls/state/se3_state_batch.h>
#include <gtest/gtest.h>
#include <stdexcept>
namespace {
struct InjectedFactor final : cunls::SizedFactorBatch<2, 6> {
  explicit InjectedFactor(bool pass) : pass(pass) {}
  bool pass;
  bool Evaluate(float* r, float* j, float const* const*, cudaStream_t stream) const override {
    cudaMemsetAsync(r, 0, 2 * sizeof(float), stream);
    if (j) {
      cudaMemsetAsync(j, 0, 12 * sizeof(float), stream);
    }
    return pass;
  }
  size_t NumFactors() const override { return 1; }
};
struct InjectedLoss final : cunls::LossFunctionBatch {
  bool Evaluate(float*, float3* out, int count, cudaStream_t stream) const override {
    cudaMemsetAsync(out, 0, count * sizeof(float3), stream);
    return false;
  }
};
}  // namespace
TEST(ResidualFailureContract, FailedFactorAndLossStopBeforePublishingState) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  cunls::cuBLASHandle handle;
  const std::vector<float> initial{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  for (int mode = 0; mode < 5; ++mode) {
    SCOPED_TRACE(mode);
    const bool factorPass = mode == 1 || mode == 2;
    cunls::DeviceVector<float> data(initial);
    cunls::SE3StateBatch state(handle, data.data(), 1);
    InjectedFactor factor(factorPass);
    InjectedLoss loss;
    cunls::ScaledLossFunctionBatch<InjectedLoss> scaled(1.f);
    cunls::WeightedFactorBatch<InjectedFactor> weighted(1.f, false);
    cunls::DeviceVector<float> sqrtInfo(std::vector<float>{1, 0, 0, 1});
    cunls::InformationFactorBatch<InjectedFactor> informed(
        handle, reinterpret_cast<const cunls::Matrix<2>*>(sqrtInfo.data()), 1, false);
    cunls::Problem problem;
    problem.AddStateBatch(&state);
    if (mode == 0) {
      problem.AddFactorBatch(&factor, {data.data()});
    }
    if (mode == 1) {
      problem.AddFactorBatch(&factor, &loss, {data.data()});
    }
    if (mode == 2) {
      problem.AddFactorBatch(&factor, &scaled, {data.data()});
    }
    if (mode == 3) {
      problem.AddFactorBatch(&weighted, {data.data()});
    }
    if (mode == 4) {
      problem.AddFactorBatch(&informed, {data.data()});
    }
    cunls::LevenbergMarquardtMinimizer solver;
    EXPECT_THROW(solver.Minimize(stream, problem, true), std::runtime_error);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<float> final(16);
    ASSERT_EQ(cudaMemcpy(final.data(), data.data(), 64, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(final, initial);
  }
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
