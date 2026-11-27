#include "joint_gpu_dense.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include "common/log.h"
#include "joint_lapack.h"
namespace cuvslam::pnp::joint_geometry::dense {
namespace {
using Matrix = Eigen::MatrixXd;
thread_local GpuMetrics metrics;
void Check(cudaError_t value) {
  if (value != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(value));
  }
}
void Check(cublasStatus_t value) {
  if (value != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("FP64 cuBLAS call failed");
  }
}
void Check(cusolverStatus_t value) {
  if (value != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error("FP64 cuSOLVER call failed");
  }
}
size_t Bytes(size_t count) {
  if (count > std::numeric_limits<size_t>::max() / sizeof(double)) {
    throw std::overflow_error("GPU matrix allocation overflow");
  }
  return count * sizeof(double);
}
struct Buffer {
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void* pointer = nullptr;
  size_t capacity = 0;
  void Ensure(size_t bytes) {
    if (bytes <= capacity) {
      return;
    }
    void* fresh = nullptr;
    Check(cudaMalloc(&fresh, bytes));
    if (pointer) {
      cudaFree(pointer);
    }
    pointer = fresh;
    capacity = bytes;
    ++metrics.allocations;
  }
  void Release() {
    if (pointer) {
      cudaFree(pointer);
    }
    pointer = nullptr;
    capacity = 0;
  }
  ~Buffer() { Release(); }
  double* D() { return static_cast<double*>(pointer); }
  int* I() { return static_cast<int*>(pointer); }
};
struct Workspace {
  Workspace() = default;
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  int device = -1;
  cudaStream_t stream = nullptr;
  cusolverDnHandle_t solver = nullptr;
  cublasHandle_t blas_handle = nullptr;
  Buffer a, b, c, tau, work, info;
  void Init(int dev) {
    device = dev;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Check(cusolverDnCreate(&solver));
    Check(cusolverDnSetStream(solver, stream));
    Check(cublasCreate(&blas_handle));
    Check(cublasSetStream(blas_handle, stream));
    Check(cublasSetPointerMode(blas_handle, CUBLAS_POINTER_MODE_HOST));
    Check(cublasSetMathMode(blas_handle, CUBLAS_PEDANTIC_MATH));
    info.Ensure(sizeof(int));
  }
  ~Workspace() {
    int previous = -1;
    cudaGetDevice(&previous);
    if (device >= 0 && previous != device) {
      cudaSetDevice(device);
    }
    if (stream) {
      cudaStreamSynchronize(stream);
    }
    a.Release();
    b.Release();
    c.Release();
    tau.Release();
    work.Release();
    info.Release();
    if (blas_handle) {
      cublasDestroy(blas_handle);
    }
    if (solver) {
      cusolverDnDestroy(solver);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
    if (previous >= 0 && previous != device) {
      cudaSetDevice(previous);
    }
  }
  void Put(Buffer& dest, const double* source, size_t count) {
    dest.Ensure(Bytes(count));
    Check(cudaMemcpyAsync(dest.pointer, source, Bytes(count), cudaMemcpyHostToDevice, stream));
  }
  void Get(const Buffer& source, double* dest, size_t count) {
    Check(cudaMemcpyAsync(dest, source.pointer, Bytes(count), cudaMemcpyDeviceToHost, stream));
    Check(cudaStreamSynchronize(stream));
  }
  bool Status() {
    int status = 0;
    Check(cudaMemcpyAsync(&status, info.pointer, sizeof(int), cudaMemcpyDeviceToHost, stream));
    Check(cudaStreamSynchronize(stream));
    return status == 0;
  }
  Matrix GetR(int m, int n) {
    const int k = std::min(m, n);
    Matrix result(k, n);
    Check(cudaMemcpy2DAsync(result.data(), Bytes(k), a.pointer, Bytes(m), Bytes(k), n, cudaMemcpyDeviceToHost, stream));
    Check(cudaStreamSynchronize(stream));
    return result;
  }
  bool Factor(const Matrix& input) {
    const int m = size(input.rows()), n = size(input.cols()), k = std::min(m, n);
    if (!m || !n || !input.allFinite()) {
      return false;
    }
    Put(a, input.data(), input.size());
    tau.Ensure(Bytes(k));
    int length = 0;
    Check(cusolverDnDgeqrf_bufferSize(solver, m, n, a.D(), m, &length));
    if (length < 0) {
      return false;
    }
    work.Ensure(Bytes(std::max(1, length)));
    Check(cudaMemsetAsync(info.pointer, 0, sizeof(int), stream));
    Check(cusolverDnDgeqrf(solver, m, n, a.D(), m, tau.D(), work.D(), length, info.I()));
    return Status();
  }
  bool Qt(int m, int k, Matrix& rhs) {
    const int n = size(rhs.cols());
    if (rhs.rows() != m || k > m) {
      return false;
    }
    Put(b, rhs.data(), rhs.size());
    int length = 0;
    Check(cusolverDnDormqr_bufferSize(solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, n, k, a.D(), m, tau.D(), b.D(), m,
                                      &length));
    if (length < 0) {
      return false;
    }
    work.Ensure(Bytes(std::max(1, length)));
    Check(cudaMemsetAsync(info.pointer, 0, sizeof(int), stream));
    Check(cusolverDnDormqr(solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, n, k, a.D(), m, tau.D(), b.D(), m, work.D(),
                           length, info.I()));
    if (!Status()) {
      return false;
    }
    Get(b, rhs.data(), rhs.size());
    return rhs.allFinite();
  }
};
thread_local std::unique_ptr<Workspace> workspace;
Workspace& GetWorkspace() {
  int device = -1;
  Check(cudaGetDevice(&device));
  if (!workspace || workspace->device != device) {
    workspace.reset();
    auto fresh = std::make_unique<Workspace>();
    fresh->Init(device);
    workspace = std::move(fresh);
  }
  return *workspace;
}
void Failure(const char* stage, const std::exception& error) {
  ++metrics.failures;
  TraceMessage("JOINT_GPU_FALLBACK stage=%s detail=%s", stage, error.what());
  workspace.reset();
}
}  // namespace
GpuMetrics GetGpuMetrics() { return metrics; }
bool GpuGram(const Matrix& input, Matrix& result) {
  try {
    if (&input == &result) {
      Matrix copy = input;
      return GpuGram(copy, result);
    }

    if (!input.size() || !input.allFinite()) {
      return false;
    }
    auto& s = GetWorkspace();
    const int m = size(input.rows()), n = size(input.cols());
    s.Put(s.a, input.data(), input.size());
    s.b.Ensure(Bytes(size_t(n) * n));
    Check(cudaMemsetAsync(s.b.pointer, 0, Bytes(size_t(n) * n), s.stream));
    const double one = 1., zero = 0.;
    Check(
        cublasDgemm(s.blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &one, s.a.D(), m, s.a.D(), m, &zero, s.b.D(), n));
    result.resize(n, n);
    s.Get(s.b, result.data(), result.size());
    if (!result.allFinite()) {
      return false;
    }
    ++metrics.gram;
    return true;
  } catch (const std::exception& e) {
    Failure("gram", e);
    return false;
  }
}
bool GpuSolveSpd(const Matrix& input, const Eigen::VectorXd& rhs, Eigen::VectorXd& result) {
  try {
    const int n = size(input.rows());
    if (!n || input.cols() != n || rhs.size() != n || !input.allFinite() || !rhs.allFinite()) {
      return false;
    }
    auto& s = GetWorkspace();
    s.Put(s.a, input.data(), input.size());
    s.Put(s.b, rhs.data(), rhs.size());
    int length = 0;
    Check(cusolverDnDpotrf_bufferSize(s.solver, CUBLAS_FILL_MODE_LOWER, n, s.a.D(), n, &length));
    if (length < 0) {
      return false;
    }
    s.work.Ensure(Bytes(std::max(1, length)));
    Check(cudaMemsetAsync(s.info.pointer, 0, sizeof(int), s.stream));
    Check(cusolverDnDpotrf(s.solver, CUBLAS_FILL_MODE_LOWER, n, s.a.D(), n, s.work.D(), length, s.info.I()));
    if (!s.Status()) {
      return false;
    }
    Check(cudaMemsetAsync(s.info.pointer, 0, sizeof(int), s.stream));
    Check(cusolverDnDpotrs(s.solver, CUBLAS_FILL_MODE_LOWER, n, 1, s.a.D(), n, s.b.D(), n, s.info.I()));
    if (!s.Status()) {
      return false;
    }
    result.resize(n);
    s.Get(s.b, result.data(), n);
    if (!result.allFinite()) {
      return false;
    }
    ++metrics.solve;
    return true;
  } catch (const std::exception& e) {
    Failure("solve", e);
    return false;
  }
}
bool GpuProjectFullRank(const Matrix& normalized, const Matrix& kept, const Eigen::VectorXd& r, Matrix& rows,
                        Eigen::VectorXd& residual) {
  try {
    auto& s = GetWorkspace();
    const int m = size(normalized.rows()), n = size(normalized.cols());
    if (!n || n > m || kept.rows() != m || r.size() != m || !s.Factor(normalized)) {
      return false;
    }
    Matrix qr = s.GetR(m, n);
    if (!CertifyFullRank(qr, m)) {
      ++metrics.rank_fallbacks;
      return false;
    }
    Matrix transformed(m, kept.cols() + 1);
    transformed.leftCols(kept.cols()) = kept;
    transformed.rightCols(1) = r;
    if (!s.Qt(m, n, transformed)) {
      return false;
    }
    rows = transformed.bottomRows(m - n).leftCols(kept.cols());
    residual = transformed.bottomRows(m - n).rightCols(1);
    ++metrics.project;
    return true;
  } catch (const std::exception& e) {
    Failure("project", e);
    return false;
  }
}
bool GpuCompress(const Matrix& input, const Eigen::VectorXd& r, Matrix& w, Eigen::VectorXd& residual) {
  try {
    auto& s = GetWorkspace();
    const int m = size(input.rows()), n = size(input.cols()), k = std::min(m, n);
    if (!m || !n || r.size() != m || !s.Factor(input)) {
      return false;
    }
    Matrix qr = s.GetR(m, n);
    Matrix transformed = r;
    if (!s.Qt(m, k, transformed)) {
      return false;
    }
    w = Matrix::Zero(n, n);
    w.topRows(k) = qr.topRows(k).triangularView<Eigen::Upper>();
    residual = Eigen::VectorXd::Zero(n);
    residual.head(k) = transformed.col(0).head(k);
    ++metrics.compress;
    return w.allFinite() && residual.allFinite();
  } catch (const std::exception& e) {
    Failure("compress", e);
    return false;
  }
}
bool WarmGpu() {
  Matrix a = Matrix::Identity(16, 16), out;
  Eigen::VectorXd rhs = Eigen::VectorXd::Ones(16), r;
  return GpuGram(a, out) && GpuSolveSpd(a, rhs, r) && GpuCompress(a, rhs, out, r);
}
}  // namespace cuvslam::pnp::joint_geometry::dense
