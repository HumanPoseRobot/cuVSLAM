#include <cunls/cunls.h>
#include <stdexcept>
#include "window_math.h"
namespace gpu_window {
void check(cudaError_t e) {
  if (e != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(e));
  }
}
struct Prior {
  State nodes[Slots], anchors[Slots];
  double W[Dim * Dim], offset[Dim];
};
__global__ void edge_kernel(const State* ns, const Edge* es, int n, float const* const* ptr, float* r, float* J) {
  int i = blockIdx.x, c = threadIdx.x - 1;
  if (i >= n || c >= 30) {
    return;
  }
  gpu_vi::Jet x[30], y[15];
  for (int k = 0; k < 30; ++k) {
    x[k] = {ptr[2 * i + k / 15][k % 15], k == c ? 1. : 0.};
  }
  const Edge& e = es[i];
  edge_res(e, ns[e.left], ns[e.right], x, y);
  for (int k = 0; k < 15; ++k) {
    if (c < 0) {
      r[i * 15 + k] = y[k].v;
    } else if (J) {
      J[(i * 15 + k) * 30 + c] = y[k].d;
    }
  }
}
__global__ void visual_kernel(const State* ns, const double* pts, const Visual* vs, int n, float const* const* ptr,
                              float* r, float* J) {
  int i = blockIdx.x, c = threadIdx.x - 1;
  if (i >= n || c >= 18) {
    return;
  }
  gpu_vi::Jet x[18], y[2];
  for (int k = 0; k < 18; ++k) {
    x[k] = {k < 15 ? ptr[2 * i][k] : ptr[2 * i + 1][k - 15], k == c ? 1. : 0.};
  }
  const auto& v = vs[i];
  visual_res(v, ns[v.node], pts + 3 * v.point, x, y);
  for (int k = 0; k < 2; ++k) {
    if (c < 0) {
      r[2 * i + k] = y[k].v;
    } else if (J) {
      J[(2 * i + k) * 18 + c] = y[k].d;
    }
  }
}
__global__ void prior_kernel(const Prior* p, float const* const* ptr, float* r, float* J) {
  int c = threadIdx.x - 1;
  if (c >= Dim) {
    return;
  }
  gpu_vi::Jet delta[Dim], x[15];
  for (int s = 0; s < Slots; ++s) {
    for (int k = 0; k < 15; ++k) {
      x[k] = {ptr[s][k], 15 * s + k == c ? 1. : 0.};
    }
    difference(p->nodes[s], p->anchors[s], x, delta + 15 * s);
  }
  for (int i = 0; i < Dim; ++i) {
    gpu_vi::Jet y(p->offset[i]);
    for (int j = 0; j < Dim; ++j) {
      y += p->W[i * Dim + j] * delta[j];
    }
    if (c < 0) {
      r[i] = y.v;
    } else if (J) {
      J[i * Dim + c] = y.d;
    }
  }
}
class EF : public cunls::SizedFactorBatch<15, 15, 15> {
  const State* n_;
  const Edge* e_;
  int size_;

public:
  EF(const State* n, const Edge* e, int z) : n_(n), e_(e), size_(z) {}
  size_t NumFactors() const override { return size_; }
  bool Evaluate(float* r, float* j, float const* const* s, cudaStream_t st) const override {
    edge_kernel<<<size_, 32, 0, st>>>(n_, e_, size_, s, r, j);
    return cudaGetLastError() == cudaSuccess;
  }
};
class VF : public cunls::SizedFactorBatch<2, 15, 3> {
  const State* n_;
  const double* p_;
  const Visual* v_;
  int size_;

public:
  VF(const State* n, const double* p, const Visual* v, int z) : n_(n), p_(p), v_(v), size_(z) {}
  size_t NumFactors() const override { return size_; }
  bool Evaluate(float* r, float* j, float const* const* s, cudaStream_t st) const override {
    visual_kernel<<<size_, 32, 0, st>>>(n_, p_, v_, size_, s, r, j);
    return cudaGetLastError() == cudaSuccess;
  }
};
class PF : public cunls::SizedFactorBatch<Dim, 15, 15, 15, 15, 15, 15> {
  const Prior* p_;

public:
  explicit PF(const Prior* p) : p_(p) {}
  size_t NumFactors() const override { return 1; }
  bool Evaluate(float* r, float* j, float const* const* s, cudaStream_t st) const override {
    prior_kernel<<<1, 128, 0, st>>>(p_, s, r, j);
    return cudaGetLastError() == cudaSuccess;
  }
};
struct Buffers {
  cunls::DeviceVector<State> nodes;
  cunls::DeviceVector<Edge> edges;
  cunls::DeviceVector<Visual> visuals;
  cunls::DeviceVector<double> points;
  cunls::DeviceVector<Prior> prior;
  cunls::DeviceVector<float> x, px;
  std::vector<float*> ep, vp, pp;
  Buffers(const Problem& p, const Result& r)
      : nodes(std::vector<State>(p.nodes.begin(), p.nodes.end())),
        edges(p.edges),
        visuals(p.visuals),
        x(std::vector<float>(r.x.begin(), r.x.end())),
        px(r.points.empty() ? std::vector<float>(3 * p.points.size(), 0) : r.points) {
    std::vector<double> points_h;
    for (auto& a : p.points) {
      points_h.insert(points_h.end(), a.begin(), a.end());
    }
    points = cunls::DeviceVector<double>(points_h);
    Prior h{};
    for (int i = 0; i < Slots; ++i) {
      h.nodes[i] = p.nodes[i];
      h.anchors[i] = p.anchors[i];
      pp.push_back(x.data() + i * 15);
    }
    std::copy(p.W.begin(), p.W.end(), h.W);
    std::copy(p.offset.begin(), p.offset.end(), h.offset);
    prior = cunls::DeviceVector<Prior>(std::vector<Prior>{h});
    for (auto& e : p.edges) {
      ep.push_back(x.data() + 15 * e.left);
      ep.push_back(x.data() + 15 * e.right);
    }
    for (auto& v : p.visuals) {
      vp.push_back(x.data() + 15 * v.node);
      vp.push_back(px.data() + 3 * v.point);
    }
  }
};
Evaluation Evaluate(const Problem& p, const Result& r) {
  Buffers b(p, r);
  Evaluation out;
  EF ef(b.nodes.data(), b.edges.data(), p.edges.size());
  VF vf(b.nodes.data(), b.points.data(), b.visuals.data(), p.visuals.size());
  PF pf(b.prior.data());
  auto run = [&](auto& f, std::vector<float*>& ptr, int rows, int cols, std::vector<float>& rs,
                 std::vector<float>& js) {
    if (rows == 0) {
      return;
    }
    cunls::DeviceVector<float*> dp(ptr);
    cunls::DeviceVector<float> dr(rows), dj(rows * cols);
    if (!f.Evaluate(dr.data(), dj.data(), dp.data(), 0)) {
      throw std::runtime_error("window factor evaluation failed");
    }
    rs.resize(rows);
    js.resize(rows * cols);
    check(cudaMemcpy(rs.data(), dr.data(), rows * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(js.data(), dj.data(), rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
  };
  run(ef, b.ep, 15 * p.edges.size(), 30, out.edge, out.edge_j);
  run(vf, b.vp, 2 * p.visuals.size(), 18, out.visual, out.visual_j);
  run(pf, b.pp, Dim, Dim, out.prior, out.prior_j);
  return out;
}
Result Optimize(const Problem& p) {
  struct Stream {
    cudaStream_t s;
    Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(s); }
  } stream;
  Result r;
  Buffers b(p, r);
  cunls::VectorStateBatch<15> states(b.x.data(), Slots);
  cunls::VectorStateBatch<3> points(b.px.data(), p.points.size());
  cunls::Problem problem;
  problem.AddStateBatch(&states);
  if (!p.points.empty()) {
    problem.AddStateBatch(&points);
  }
  EF ef(b.nodes.data(), b.edges.data(), p.edges.size());
  VF vf(b.nodes.data(), b.points.data(), b.visuals.data(), p.visuals.size());
  PF pf(b.prior.data());
  cunls::HuberLossFunctionBatch huber(p.huber);
  problem.AddFactorBatch(&pf, b.pp);
  if (!p.edges.empty()) {
    problem.AddFactorBatch(&ef, b.ep);
  }
  if (!p.visuals.empty()) {
    problem.AddFactorBatch(&vf, &huber, b.vp);
  }
  if (!problem.CheckConsistency()) {
    throw std::runtime_error("invalid window graph");
  }
  cunls::LevenbergMarquardtMinimizerOptions opt;
  opt.initial_lambda = 1e-3;
  opt.lambda_upscale = 10;
  opt.lambda_downscale = .3;
  opt.base_options.max_num_iterations = p.iterations;
  opt.base_options.column_scaling = cunls::ColumnScaling::HessianDiagonal;
  opt.base_options.sparse_linear_solver_type = cunls::SparseLinearSolverType::BlockSparsePCG;
  opt.base_options.sparse_linear_solver_config.block_sparse_pcg_options.max_iterations = 400;
  opt.base_options.sparse_linear_solver_config.block_sparse_pcg_options.relative_tolerance = 1e-5;
  opt.base_options.cost_tolerance = 1e-7;
  opt.base_options.state_tolerance = 1e-10;
  cunls::LevenbergMarquardtMinimizer lm(opt);
  auto summary = lm.Minimize(stream.s, problem);
  check(cudaMemcpy(r.x.data(), b.x.data(), Dim * sizeof(float), cudaMemcpyDeviceToHost));
  r.points.resize(p.points.size() * 3);
  if (!r.points.empty()) {
    check(cudaMemcpy(r.points.data(), b.px.data(), r.points.size() * sizeof(float), cudaMemcpyDeviceToHost));
  }
  r.before = summary.initial_cost;
  r.after = summary.final_cost;
  r.iterations = summary.num_iterations;
  return r;
}
}  // namespace gpu_window
