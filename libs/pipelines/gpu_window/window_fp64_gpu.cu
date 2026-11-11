#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "window_fp64.h"
#include "window_math.h"

namespace gpu_window::fp64 {
namespace {
void Check(cudaError_t e) {
  if (e != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(e));
  }
}
template <class T>
struct Buffer {
  T* ptr = nullptr;
  size_t size = 0;
  explicit Buffer(size_t n = 0) : size(n) {
    if (n) {
      Check(cudaMalloc(&ptr, n * sizeof(T)));
    }
  }
  ~Buffer() {
    if (ptr) {
      cudaFree(ptr);
    }
  }
  Buffer(const Buffer&) = delete;
  void Upload(const T* p, size_t n, cudaStream_t s) {
    if (n) {
      Check(cudaMemcpyAsync(ptr, p, n * sizeof(T), cudaMemcpyHostToDevice, s));
    }
  }
  std::vector<T> Download(cudaStream_t s) {
    std::vector<T> v(size);
    if (size) {
      Check(cudaMemcpyAsync(v.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost, s));
    }
    Check(cudaStreamSynchronize(s));
    return v;
  }
};
struct Prior {
  State nodes[Slots], anchors[Slots];
  double W[Dim * Dim], offset[Dim];
};
__global__ void PriorEval(int n, const Prior* p, const double* x, double* r, double* j, bool jac) {
  int c = int(threadIdx.x) - 1;
  if (c >= n || (!jac && c >= 0)) {
    return;
  }
  gpu_vi::Jet d[Dim], a[15];
  for (int s = 0; s < n / 15; ++s) {
    for (int k = 0; k < 15; ++k) {
      a[k] = {x[15 * s + k], 15 * s + k == c ? 1. : 0.};
    }
    difference(p->nodes[s], p->anchors[s], a, d + 15 * s);
  }
  for (int i = 0; i < n; ++i) {
    gpu_vi::Jet v(p->offset[i]);
    for (int k = 0; k < n; ++k) {
      v += p->W[i * Dim + k] * d[k];
    }
    if (c < 0) {
      r[i] = v.v;
    } else {
      j[i * n + c] = v.d;
    }
  }
}
__global__ void EdgeEval(const State* ns, const Edge* edges, const double* x, double* r, double* j, bool jac) {
  int z = blockIdx.x, c = int(threadIdx.x) - 1;
  if (c >= 30 || (!jac && c >= 0)) {
    return;
  }
  const auto& e = edges[z];
  gpu_vi::Jet a[30], b[15];
  for (int k = 0; k < 30; ++k) {
    a[k] = {x[15 * (k < 15 ? e.left : e.right) + k % 15], k == c ? 1. : 0.};
  }
  edge_res(e, ns[e.left], ns[e.right], a, b);
  for (int k = 0; k < 15; ++k) {
    if (c < 0) {
      r[15 * z + k] = b[k].v;
    } else {
      j[450 * z + 30 * k + c] = b[k].d;
    }
  }
}
__global__ void VisualEval(int n, const State* ns, const double* points, const Visual* vs, const double* x, double* r,
                           double* j, bool jac) {
  int z = blockIdx.x, c = int(threadIdx.x) - 1;
  if (c >= 18 || (!jac && c >= 0)) {
    return;
  }
  const auto& v = vs[z];
  gpu_vi::Jet a[18], b[2];
  for (int k = 0; k < 18; ++k) {
    a[k] = {x[k < 15 ? 15 * v.node + k : n + 3 * v.point + k - 15], k == c ? 1. : 0.};
  }
  visual_res(v, ns[v.node], points + 3 * v.point, a, b);
  for (int k = 0; k < 2; ++k) {
    if (c < 0) {
      r[2 * z + k] = b[k].v;
    } else {
      j[36 * z + 18 * k + c] = b[k].d;
    }
  }
}
__device__ double Weight(const double* r, int i, double huber) {
  double n = ::hypot(r[2 * i], r[2 * i + 1]);
  return n > huber ? huber / n : 1.;
}
// Each normal entry has a single owner and a fixed observation order. No floating
// atomicAdd or cross-stream scratch memory is used in the FP64 assembly.
__global__ void AssembleNav(int n, int ne, const Edge* edges, const double* pr, const double* pj, const double* er,
                            const double* ej, const double* vr, const double* vj, const int* offsets, const int* ids,
                            double huber, double* H, double* g) {
  int a = blockIdx.x, b = threadIdx.x;
  if (b > n) {
    return;
  }
  double sum = 0;
  for (int k = 0; k < n; ++k) {
    sum += pj[k * n + a] * (b == n ? pr[k] : pj[k * n + b]);
  }
  for (int z = 0; z < ne; ++z) {
    auto e = edges[z];
    int ca = a / 15 == e.left ? a % 15 : (a / 15 == e.right ? 15 + a % 15 : -1);
    int cb = b / 15 == e.left ? b % 15 : (b / 15 == e.right ? 15 + b % 15 : -1);
    if (ca < 0 || (b < n && cb < 0)) {
      continue;
    }
    for (int k = 0; k < 15; ++k) {
      sum += ej[z * 450 + k * 30 + ca] * (b == n ? er[z * 15 + k] : ej[z * 450 + k * 30 + cb]);
    }
  }
  if (b == n || a / 15 == b / 15) {
    int node = a / 15;
    for (int at = offsets[node]; at < offsets[node + 1]; ++at) {
      int z = ids[at];
      double w = Weight(vr, z, huber);
      for (int k = 0; k < 2; ++k) {
        sum += w * vj[z * 36 + k * 18 + a % 15] * (b == n ? vr[z * 2 + k] : vj[z * 36 + k * 18 + b % 15]);
      }
    }
  }
  if (b == n) {
    g[a] = sum;
  } else {
    H[a * n + b] = sum;
  }
}
__global__ void AssemblePoints(int n, const Visual* vs, const int* offsets, const int* ids, const double* vr,
                               const double* vj, double huber, double* B, double* C, double* g) {
  int p = blockIdx.x;
  for (int item = threadIdx.x; item < 3 * n + 12; item += blockDim.x) {
    double sum = 0;
    for (int at = offsets[p]; at < offsets[p + 1]; ++at) {
      int z = ids[at];
      double w = Weight(vr, z, huber);
      if (item < 3 * n) {
        int k = item / n, a = item % n;
        if (vs[z].node != a / 15) {
          continue;
        }
        for (int r = 0; r < 2; ++r) {
          sum += w * vj[z * 36 + r * 18 + 15 + k] * vj[z * 36 + r * 18 + a % 15];
        }
      } else if (item < 3 * n + 9) {
        int k = item - 3 * n;
        for (int r = 0; r < 2; ++r) {
          sum += w * vj[z * 36 + r * 18 + 15 + k / 3] * vj[z * 36 + r * 18 + 15 + k % 3];
        }
      } else {
        int k = item - 3 * n - 9;
        for (int r = 0; r < 2; ++r) {
          sum += w * vj[z * 36 + r * 18 + 15 + k] * vr[z * 2 + r];
        }
      }
    }
    if (item < 3 * n) {
      B[p * 3 * n + item] = sum;
    } else if (item < 3 * n + 9) {
      C[p * 9 + item - 3 * n] = sum;
    } else {
      g[n + 3 * p + item - 3 * n - 9] = sum;
    }
  }
}
__global__ void Dampen(int n, int np, const double* H, const double* C, const double* g, double lambda, double* nav,
                       double* blocks, double* rhs) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < n * n) {
    nav[k] = H[k] + (k / n == k % n ? lambda * ::fmax(H[k], 1e-12) : 0.);
  }
  if (k < np * 9) {
    blocks[k] = C[k] + (k % 9 / 3 == k % 3 ? lambda * ::fmax(C[k], 1e-12) : 0.);
  }
  if (k < n + 3 * np) {
    rhs[k] = -g[k];
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
__global__ void eliminatePoints(int n, const double* blocks, const double* cross, const double* rhs, double* factors,
                                double* projected, int* status) {
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
__global__ void reduce(int n, int point_dim, const double* cross, const double* projected, const double* rhs,
                       double* nav, double* reduced_rhs) {
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
__global__ void solveNavigation(int n, double* nav, double* rhs, double* result, int* status) {
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
                        const double* rhs, double* result) {
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

// Audit the original undamped quadratic and damped equation residual, in FP64.
// Adding an unreported diagonal shift is forbidden: a failed solve returns to
// the common LM controller, which increases its own lambda and retries.
__global__ void StepMetrics(int n, int np, const double* H, const double* B, const double* C, const double* g,
                            const double* dx, double lambda, double* metrics) {
  int row = blockIdx.x * blockDim.x + threadIdx.x, dim = n + 3 * np;
  if (row >= dim) {
    return;
  }
  double hdx = 0, diag = 0;
  if (row < n) {
    for (int k = 0; k < n; ++k) {
      hdx += H[row * n + k] * dx[k];
    }
    for (int k = 0; k < 3 * np; ++k) {
      hdx += B[k * n + row] * dx[n + k];
    }
    diag = H[row * n + row];
  } else {
    int pt = (row - n) / 3, r = (row - n) % 3;
    for (int k = 0; k < n; ++k) {
      hdx += B[(row - n) * n + k] * dx[k];
    }
    for (int k = 0; k < 3; ++k) {
      hdx += C[9 * pt + 3 * r + k] * dx[n + 3 * pt + k];
    }
    diag = C[9 * pt + 4 * r];
  }
  double residual = hdx + lambda * ::fmax(diag, 1e-12) * dx[row] + g[row];
  metrics[3 * row] = -g[row] * dx[row] - .5 * dx[row] * hdx;
  metrics[3 * row + 1] = residual * residual;
  metrics[3 * row + 2] = g[row] * g[row];
}
class Gpu final : public Backend {
  const Problem& p;
  int n, np, nv, ne;
  cudaStream_t stream;
  Buffer<Prior> prior;
  Buffer<State> nodes;
  Buffer<Edge> edges;
  Buffer<Visual> visuals;
  Buffer<double> points, x, pr, pj, er, ej, vr, vj, H, B, C, g, nav, blocks, rhs, factors, projected, reduced, dx,
      metrics;
  Buffer<int> node_offsets, node_ids, point_offsets, point_ids, status;

public:
  explicit Gpu(const Problem& problem)
      : p(problem),
        n(p.count * 15),
        np(p.points.size()),
        nv(p.visuals.size()),
        ne(p.edges.size()),
        prior(1),
        nodes(p.count),
        edges(ne),
        visuals(nv),
        points(3 * np),
        x(n + 3 * np),
        pr(n),
        pj(n * n),
        er(15 * ne),
        ej(450 * ne),
        vr(2 * nv),
        vj(36 * nv),
        H(n * n),
        B(3 * np * n),
        C(9 * np),
        g(n + 3 * np),
        nav(n * n),
        blocks(9 * np),
        rhs(n + 3 * np),
        factors(9 * np),
        projected(3 * np * (n + 1)),
        reduced(n),
        dx(n + 3 * np),
        metrics(3 * (n + 3 * np)),
        node_offsets(p.count + 1),
        node_ids(nv),
        point_offsets(np + 1),
        point_ids(nv),
        status(1) {
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Prior h{};
    std::copy(p.nodes.begin(), p.nodes.end(), h.nodes);
    std::copy(p.anchors.begin(), p.anchors.end(), h.anchors);
    std::copy(p.W.begin(), p.W.end(), h.W);
    std::copy(p.offset.begin(), p.offset.end(), h.offset);
    prior.Upload(&h, 1, stream);
    nodes.Upload(p.nodes.data(), p.count, stream);
    edges.Upload(p.edges.data(), ne, stream);
    visuals.Upload(p.visuals.data(), nv, stream);
    points.Upload(reinterpret_cast<const double*>(p.points.data()), 3 * np, stream);
    auto index = [&](int count, bool point, Buffer<int>& offset, Buffer<int>& ids) {
      std::vector<int> o(count + 1), v;
      v.reserve(nv);
      for (int k = 0; k < count; ++k) {
        o[k] = v.size();
        for (int i = 0; i < nv; ++i) {
          if ((point ? p.visuals[i].point : p.visuals[i].node) == k) {
            v.push_back(i);
          }
        }
      }
      o[count] = v.size();
      offset.Upload(o.data(), o.size(), stream);
      ids.Upload(v.data(), v.size(), stream);
      Check(cudaStreamSynchronize(stream));
    };
    index(p.count, false, node_offsets, node_ids);
    index(np, true, point_offsets, point_ids);
    Check(cudaStreamSynchronize(stream));
  }
  ~Gpu() override {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }
  double Evaluate(const std::vector<double>& state, bool jac) override {
    x.Upload(state.data(), state.size(), stream);
    PriorEval<<<1, 128, 0, stream>>>(n, prior.ptr, x.ptr, pr.ptr, pj.ptr, jac);
    if (ne) {
      EdgeEval<<<ne, 32, 0, stream>>>(nodes.ptr, edges.ptr, x.ptr, er.ptr, ej.ptr, jac);
    }
    if (nv) {
      VisualEval<<<nv, 32, 0, stream>>>(n, nodes.ptr, points.ptr, visuals.ptr, x.ptr, vr.ptr, vj.ptr, jac);
    }
    if (jac) {
      AssembleNav<<<n, 128, 0, stream>>>(n, ne, edges.ptr, pr.ptr, pj.ptr, er.ptr, ej.ptr, vr.ptr, vj.ptr,
                                         node_offsets.ptr, node_ids.ptr, p.huber, H.ptr, g.ptr);
      if (np) {
        AssemblePoints<<<np, 128, 0, stream>>>(n, visuals.ptr, point_offsets.ptr, point_ids.ptr, vr.ptr, vj.ptr,
                                               p.huber, B.ptr, C.ptr, g.ptr);
      }
    }
    Check(cudaGetLastError());
    // Only scalar-cost reduction is on the host; all factor evaluation, normal
    // assembly and Schur elimination above/below remain on the GPU in FP64.
    const auto a = pr.Download(stream), b = er.Download(stream), c = vr.Download(stream);
    double cost = 0;
    for (double v : a) {
      cost += .5 * v * v;
    }
    for (double v : b) {
      cost += .5 * v * v;
    }
    for (size_t i = 0; i < c.size(); i += 2) {
      double r = std::hypot(c[i], c[i + 1]);
      cost += r > p.huber ? p.huber * (r - .5 * p.huber) : .5 * r * r;
    }
    return cost;
  }
  Step Solve(double lambda) override {
    Check(cudaMemsetAsync(status.ptr, 0, sizeof(int), stream));
    int count = std::max({n * n, 9 * np, n + 3 * np});
    Dampen<<<(count + 127) / 128, 128, 0, stream>>>(n, np, H.ptr, C.ptr, g.ptr, lambda, nav.ptr, blocks.ptr, rhs.ptr);
    if (np) {
      eliminatePoints<<<np, 128, 0, stream>>>(n, blocks.ptr, B.ptr, rhs.ptr, factors.ptr, projected.ptr, status.ptr);
    }
    reduce<<<n, 128, 0, stream>>>(n, 3 * np, B.ptr, projected.ptr, rhs.ptr, nav.ptr, reduced.ptr);
    solveNavigation<<<1, 128, 0, stream>>>(n, nav.ptr, reduced.ptr, dx.ptr, status.ptr);
    if (np) {
      recover<<<(np + 127) / 128, 128, 0, stream>>>(n, np, factors.ptr, B.ptr, reduced.ptr, rhs.ptr, dx.ptr);
    }
    Check(cudaGetLastError());
    if (status.Download(stream)[0]) {
      return Step{};
    }
    StepMetrics<<<(n + 3 * np + 127) / 128, 128, 0, stream>>>(n, np, H.ptr, B.ptr, C.ptr, g.ptr, dx.ptr, lambda,
                                                              metrics.ptr);
    auto values = metrics.Download(stream);
    Step step;
    step.dx = dx.Download(stream);
    double residual = 0, rhs2 = 0;
    for (size_t i = 0; i < values.size(); i += 3) {
      step.predicted += values[i];
      residual += values[i + 1];
      rhs2 += values[i + 2];
    }
    step.relative_residual = std::sqrt(residual / std::max(rhs2, 1e-30));
    step.valid = std::all_of(step.dx.begin(), step.dx.end(), [](double v) { return std::isfinite(v); }) &&
                 std::isfinite(step.predicted) && std::isfinite(step.relative_residual) &&
                 step.relative_residual < 1e-6;
    return step;
  }
};
}  // namespace
std::unique_ptr<Backend> MakeGpu(const Problem& p) { return std::make_unique<Gpu>(p); }
}  // namespace gpu_window::fp64
