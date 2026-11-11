#pragma once

#include "common/log.h"

#include <cunls/linear_solver/csr_sparse_linear_solver.h>
#include <cunls/minimizer/problem.h>
#include <cunls/minimizer/levenberg_marquardt_minimizer.h>

namespace gpu_window {
void check(cudaError_t error);
namespace schur {

// The damped, column-scaled normal system is [A B; B^T C]. Each
// landmark contributes an independent 3x3 block to C. Pack B by point
// coordinate so the navigation-column threads can read it contiguously.

__global__ void unpack(int n, const int* rows, const int* cols, const float* values,
                       double* nav, double* cross, double* blocks, int* status, double diagonal_shift) {
  const int row = blockIdx.x;
  for (int k = rows[row] + threadIdx.x; k < rows[row + 1]; k += blockDim.x) {
    const int col = cols[k];
    const double value = values[k] + (row == col ? diagonal_shift : 0);
    if (row < n && col < n) {
      nav[row * n + col] = value;
    } else if (row < n) {
      cross[(col - n) * n + row] = value;
    } else if (col >= n) {
      if ((row - n) / 3 == (col - n) / 3) {
        blocks[(row - n) * 3 + (col - n) % 3] = value;
      } else if (value != 0) {
        atomicExch(status, 1);
      }
    }
  }
}

__device__ void triangular3(const double* l, const double* b, double* x) {
  for (int i = 0; i < 3; ++i) {
    double v = b[i];
    for (int j = 0; j < i; ++j) {
      v -= l[3 * i + j] * x[j];
    }
    x[i] = v / l[3 * i + i];
  }
  for (int i = 2; i >= 0; --i) {
    double v = x[i];
    for (int j = i + 1; j < 3; ++j) {
      v -= l[3 * j + i] * x[j];
    }
    x[i] = v / l[3 * i + i];
  }
}

// Factor C once per landmark and compute C^-1 [B^T | rhs_point].
__global__ void eliminatePoints(int n, const double* blocks, const double* cross, const float* rhs,
                                double* factors, double* projected, int* status) {
  const int point = blockIdx.x;
  __shared__ double l[9];
  __shared__ bool valid;
  if (threadIdx.x == 0) {
    valid = true;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j <= i; ++j) {
        double v = blocks[9 * point + 3 * i + j];
        for (int k = 0; k < j; ++k) {
          v -= l[3 * i + k] * l[3 * j + k];
        }
        if (i == j) {
          if (!(v > 0) || !isfinite(v)) {
            valid = false;
            v = 1;
          }
          l[3 * i + j] = ::sqrt(v);
        } else {
          l[3 * i + j] = v / l[3 * j + j];
        }
      }
    }
    if (!valid) {
      atomicExch(status, 2);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 9; i += blockDim.x) {
    factors[9 * point + i] = i / 3 >= i % 3 ? l[i] : 0;
  }
  for (int col = threadIdx.x; col <= n; col += blockDim.x) {
    double b[3], x[3];
    for (int k = 0; k < 3; ++k) {
      b[k] = col == n ? rhs[n + 3 * point + k] : cross[(3 * point + k) * n + col];
    }
    triangular3(l, b, x);
    for (int k = 0; k < 3; ++k) {
      projected[(3 * point + k) * (n + 1) + col] = x[k];
    }
  }
}

// Assemble the lower triangle of A - B C^-1 B^T and its reduced RHS.
__global__ void reduce(int n, int point_dim, const double* cross, const double* projected,
                       const float* rhs, double* nav, double* reduced_rhs) {
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (col > row && col != n) {
    return;
  }
  if (col > n) {
    return;
  }
  double sum = 0;
  for (int k = 0; k < point_dim; ++k) {
    sum += cross[k * n + row] * projected[k * (n + 1) + col];
  }
  if (col == n) {
    reduced_rhs[row] = rhs[row] - sum;
  } else {
    nav[row * n + col] -= sum;
  }
}

// At most 90 navigation coordinates fit in one block. Use double precision
// when subtracting nearly equal terms and when factoring the reduced system.
__global__ void solveNavigation(int n, double* nav, double* rhs, float* result, int* status) {
  const int row = threadIdx.x;
  for (int k = 0; k < n; ++k) {
    if (row == k) {
      double v = nav[k * n + k];
      for (int j = 0; j < k; ++j) {
        const double x = nav[k * n + j];
        v -= x * x;
      }
      if (!(v > 0) || !isfinite(v)) {
        atomicExch(status, 3);
        v = 1;
      }
      nav[k * n + k] = ::sqrt(v);
    }
    __syncthreads();
    if (row > k && row < n) {
      double v = nav[row * n + k];
      for (int j = 0; j < k; ++j) {
        v -= nav[row * n + j] * nav[k * n + j];
      }
      nav[row * n + k] = v / nav[k * n + k];
    }
    __syncthreads();
  }
  if (row == 0) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        rhs[i] -= nav[i * n + j] * rhs[j];
      }
      rhs[i] /= nav[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
      for (int j = i + 1; j < n; ++j) {
        rhs[i] -= nav[j * n + i] * rhs[j];
      }
      rhs[i] /= nav[i * n + i];
      result[i] = rhs[i];
    }
  }
}

__global__ void recover(int n, int points, const double* factors, const double* cross, const double* nav_result,
                        const float* rhs, float* result) {
  const int point = blockIdx.x * blockDim.x + threadIdx.x;
  if (point >= points) {
    return;
  }
  double b[3], x[3];
  for (int i = 0; i < 3; ++i) {
    const int row = 3 * point + i;
    b[i] = rhs[n + row];
    for (int j = 0; j < n; ++j) {
      b[i] -= cross[row * n + j] * nav_result[j];
    }
  }
  triangular3(factors + 9 * point, b, x);
  for (int i = 0; i < 3; ++i) {
    result[n + 3 * point + i] = x[i];
  }
}

class Solver final : public cunls::CSRSparseLinearSolver {
public:
  explicit Solver(int navigation_size) : n_(navigation_size) {}

  bool Initialize(cudaStream_t, const cunls::Problem& problem, const cunls::CSRSparseMatrix& matrix,
                  const cunls::dvector<float>& rhs, cunls::dvector<float>& result) override {
    const auto& states = problem.GetStateBatches();
    if (states.empty() || states.size() > 2 || states[0]->TangentSize() != 15 ||
        states[0]->NumStateBlocks() * 15 != size_t(n_) || n_ > 90 || n_ <= 0 ||
        (states.size() == 2 && states[1]->TangentSize() != 3)) {
      return false;
    }
    points_ = states.size() == 2 ? states[1]->NumStateBlocks() : 0;
    if (matrix.NumRows() != size_t(n_ + 3 * points_) || matrix.NumRows() != rhs.size() ||
        rhs.size() != result.size()) {
      return false;
    }
    nav_.resize(n_ * n_);
    cross_.resize(3 * points_ * n_);
    blocks_.resize(9 * points_);
    factors_.resize(9 * points_);
    projected_.resize(3 * points_ * (n_ + 1));
    reduced_rhs_.resize(n_);
    status_.resize(1);
    return true;
  }

  bool Solve(cudaStream_t stream, const cunls::CSRSparseMatrix& matrix, const cunls::dvector<float>& rhs,
             cunls::dvector<float>& result) override {
    // A float Hessian can lose positive definiteness near a weakly
    // constrained direction. Retry only a failed factorization, in scaled units.
    constexpr double shifts[] = {0, 1e-6, 1e-5, 1e-4, 1e-3};
    for (double shift : shifts) {
      const int status = SolveOnce(stream, matrix, rhs, result, shift);
      if (status == 0) {
        if (shift > 0) {
          TraceMessage("SCALE_SCHUR_RETRY shift=%.6g nav=%d points=%d", shift, n_, points_);
        }
        return true;
      }
      if (status == 1) {
        throw std::runtime_error("GPU Schur graph has cross-landmark coupling");
      }
    }
    throw std::runtime_error("GPU Schur linear solve remained indefinite after diagonal retries");
  }

private:
  int SolveOnce(cudaStream_t stream, const cunls::CSRSparseMatrix& matrix, const cunls::dvector<float>& rhs,
                cunls::dvector<float>& result, double shift) {
    check(cudaMemsetAsync(nav_.data(), 0, nav_.size() * sizeof(double), stream));
    check(cudaMemsetAsync(status_.data(), 0, sizeof(int), stream));
    if (points_) {
      check(cudaMemsetAsync(cross_.data(), 0, cross_.size() * sizeof(double), stream));
      check(cudaMemsetAsync(blocks_.data(), 0, blocks_.size() * sizeof(double), stream));
    }
    unpack<<<matrix.NumRows(), 64, 0, stream>>>(n_, matrix.row_offsets.data(),
        matrix.col_ids.data(), matrix.values.data(), nav_.data(), cross_.data(), blocks_.data(), status_.data(), shift);
    if (points_) {
      eliminatePoints<<<points_, 128, 0, stream>>>(n_, blocks_.data(), cross_.data(), rhs.data(),
          factors_.data(), projected_.data(), status_.data());
    }
    reduce<<<n_, 128, 0, stream>>>(n_, 3 * points_, cross_.data(), projected_.data(), rhs.data(),
        nav_.data(), reduced_rhs_.data());
    solveNavigation<<<1, 128, 0, stream>>>(n_, nav_.data(), reduced_rhs_.data(), result.data(), status_.data());
    if (points_) {
      recover<<<(points_ + 127) / 128, 128, 0, stream>>>(n_, points_, factors_.data(), cross_.data(),
          reduced_rhs_.data(), rhs.data(), result.data());
    }
    check(cudaGetLastError());
    int status = 0;
    check(cudaMemcpyAsync(&status, status_.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return status;
  }

  int n_;
  int points_ = 0;
  cunls::dvector<double> nav_, cross_, blocks_, factors_, projected_, reduced_rhs_;
  cunls::dvector<int> status_;
};

class Minimizer final : public cunls::LevenbergMarquardtMinimizer {
public:
  Minimizer(const cunls::LevenbergMarquardtMinimizerOptions& options, int n, bool use_schur)
      : LevenbergMarquardtMinimizer(options) {
    if (use_schur) {
      solver_ = std::make_unique<Solver>(n);
    }
  }
};

}  // namespace schur
}  // namespace gpu_window
