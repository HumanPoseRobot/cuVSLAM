#include <cunls/cunls.h>
#include <stdexcept>
#include <chrono>
#include <memory>
#include "gpu_joint_math_ad.h"
namespace gpu_vi {
inline void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template <int Kind, class T>
__device__ void residual(const Data& d, const Observation* obs, int i, const T* x, T* r) {
  if constexpr (Kind == 0) imu_res(d, x, r);
  if constexpr (Kind == 1) prior_res(d, x, r);
  if constexpr (Kind == 2) visual_res(d, obs[i], x, r);
}
// Forward-mode differentiation on device, one tangent direction per thread.
// The numerical-difference implementation remains an independent reference.
template <int Kind, int N>
__global__ void evaluate_kernel(const Data* d, const Observation* obs, int count, float const* const* states, float* r,
                                float* J) {
  int i = blockIdx.x;
  int c = threadIdx.x - 1;
  if (i >= count || c >= 24) return;
  double x[24];
  for (int j = 0; j < 9; ++j) {
    x[j] = states[3 * i][j];
    x[9 + j] = states[3 * i + 1][j];
  }
  for (int j = 0; j < 6; ++j) x[18 + j] = states[3 * i + 2][j];
  double y[15];
  if (c < 0) {
    residual<Kind>(*d, obs, i, x, y);
    for (int k = 0; k < N; ++k) r[i * N + k] = y[k];
    return;
  }
  if (!J) return;
  Jet dx[24], dy[15];
  for (int k = 0; k < 24; ++k) dx[k] = Jet(x[k], k == c ? 1. : 0.);
  residual<Kind>(*d, obs, i, dx, dy);
  for (int k = 0; k < N; ++k) J[(i * N + k) * 24 + c] = dy[k].d;
}
template <int Kind, int N>
class Factor : public cunls::SizedFactorBatch<N, 9, 9, 6> {
  const Data* data_;
  const Observation* obs_;
  int count_;

public:
  Factor(const Data* d, const Observation* o, int n) : data_(d), obs_(o), count_(n) {}
  size_t NumFactors() const override { return count_; }
  bool Evaluate(float* r, float* j, float const* const* s, cudaStream_t stream) const override {
    evaluate_kernel<Kind, N><<<count_, 32, 0, stream>>>(data_, obs_, count_, s, r, j);
    return cudaGetLastError() == cudaSuccess;
  }
};
struct Buffers {
  cunls::DeviceVector<Data> d;
  cunls::DeviceVector<Observation> obs;
  cunls::DeviceVector<float> x;
  Buffers(const Data& a, const std::vector<Observation>& b, const float* state)
      : d(std::vector<Data>{a}), obs(b), x(std::vector<float>(state, state + 24)) {}
};
Evaluation EvaluateGpu(const Data& d, const std::vector<Observation>& obs, const float state[24]) {
  Buffers b(d, obs, state);
  Evaluation out;
  std::vector<float*> pointers;
  for (size_t i = 0; i < obs.size(); ++i) {
    pointers.push_back(b.x.data());
    pointers.push_back(b.x.data() + 9);
    pointers.push_back(b.x.data() + 18);
  }
  cunls::DeviceVector<float*> ps(pointers);
  auto run = [&](auto& f, int n, std::vector<float>& r, std::vector<float>& j) {
    cunls::DeviceVector<float> dr(n), dj(n * 24);
    if (!f.Evaluate(dr.data(), dj.data(), ps.data(), 0)) throw std::runtime_error("GPU factor failed");
    r.resize(n);
    j.resize(n * 24);
    check(cudaMemcpy(r.data(), dr.data(), n * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(j.data(), dj.data(), n * 24 * sizeof(float), cudaMemcpyDeviceToHost));
  };
  Factor<0, 9> imu(b.d.data(), nullptr, 1);
  Factor<1, 15> prior(b.d.data(), nullptr, 1);
  Factor<2, 2> visual(b.d.data(), b.obs.data(), obs.size());
  run(imu, 9, out.imu, out.imu_j);
  run(prior, 15, out.prior, out.prior_j);
  run(visual, obs.size() * 2, out.visual, out.visual_j);
  return out;
}
Result OptimizeGpu(const Data& d, const std::vector<Observation>& obs, float huber, int iterations, bool reuse_lm,
                   bool fixed_order_normal_equations) {
  const auto start = std::chrono::steady_clock::now();
  struct Stream {
    cudaStream_t value;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
  } stream;
  Result result;
  Buffers b(d, obs, result.state);
  const auto buffers_ready = std::chrono::steady_clock::now();
  cunls::VectorStateBatch<9> poses(b.x.data(), 2);
  cunls::VectorStateBatch<6> bias(b.x.data() + 18, 1);
  cunls::Problem problem;
  problem.AddStateBatch(&poses);
  problem.AddStateBatch(&bias);
  std::vector<float*> one = {poses.StateBlockDevicePtr(0), poses.StateBlockDevicePtr(1), bias.StateBlockDevicePtr(0)},
                      many;
  many.reserve(3 * obs.size());
  for (size_t i = 0; i < obs.size(); ++i) {
    many.insert(many.end(), one.begin(), one.end());
  }
  Factor<0, 9> imu(b.d.data(), nullptr, 1);
  Factor<1, 15> prior(b.d.data(), nullptr, 1);
  Factor<2, 2> visual(b.d.data(), b.obs.data(), obs.size());
  cunls::HuberLossFunctionBatch loss(huber);
  problem.AddFactorBatch(&imu, one);
  problem.AddFactorBatch(&prior, one);
  problem.AddFactorBatch(&visual, &loss, many);
  if (!problem.CheckConsistency()) throw std::runtime_error("Invalid GPU VI graph");
  cunls::LevenbergMarquardtMinimizerOptions options;
  options.initial_lambda = 1e-4;
  options.lambda_upscale = 10;
  options.lambda_downscale = .3;
  options.base_options.max_num_iterations = iterations;
  options.base_options.column_scaling = cunls::ColumnScaling::HessianDiagonal;
  options.base_options.sparse_linear_solver_type = cunls::SparseLinearSolverType::DenseLDLT;
  options.base_options.state_tolerance = 1e-14;
  options.base_options.cost_tolerance = 1e-9;
  // Minimize reinitializes the graph, state snapshots and damping every call.
  // Keep the allocations thread-local and use a fresh instance after a device switch.
  struct Workspace {
    int device = -1;
    int iterations = 0;
    std::unique_ptr<cunls::LevenbergMarquardtMinimizer> minimizer;
  };
  thread_local Workspace workspace;
  std::unique_ptr<cunls::LevenbergMarquardtMinimizer> temporary;
  auto* storage = &temporary;
  if (reuse_lm) {
    int device = -1;
    check(cudaGetDevice(&device));
    if (workspace.device == -1 || workspace.device == device) {
      if (workspace.iterations != iterations) {
        workspace.minimizer.reset();
      }
      workspace.device = device;
      workspace.iterations = iterations;
      storage = &workspace.minimizer;
    }
  }
  if (!*storage) {
    *storage = std::make_unique<cunls::LevenbergMarquardtMinimizer>(options);
  }
  const auto graph_ready = std::chrono::steady_clock::now();
  cunls::MinimizerSummary summary;
  try {
    summary = (*storage)->Minimize(stream.value, problem, fixed_order_normal_equations);
  } catch (...) {
    cudaStreamSynchronize(stream.value);
    storage->reset();
    throw;
  }
  const auto lm_returned = std::chrono::steady_clock::now();
  check(cudaMemcpy(result.state, b.x.data(), 24 * sizeof(float), cudaMemcpyDeviceToHost));
  const auto copied = std::chrono::steady_clock::now();
  const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  result.buffer_ms = ms(start, buffers_ready);
  result.graph_ms = ms(buffers_ready, graph_ready);
  result.lm_ms = ms(graph_ready, lm_returned);
  result.copy_ms = ms(lm_returned, copied);
  result.initial_cost = summary.initial_cost;
  result.final_cost = summary.final_cost;
  result.iterations = summary.num_iterations;
  return result;
}
}  // namespace gpu_vi
