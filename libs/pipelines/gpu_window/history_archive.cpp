#include "history_archive.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include "window_linear.h"

namespace gpu_window::posterior::history {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
namespace gaussian = cuvslam::pnp::joint_geometry;
constexpr double kInheritedRankTolerance = 1e-10;  // Existing project_out policy, not a scene gate.
int Width(const Key& key) { return key.kind == Key::Navigation ? 15 : key.kind == Key::Point ? 3 : 2; }
std::vector<Key> Keys(const Prior& prior) {
  std::vector<Key> result;
  for (auto time : prior.times) {
    result.push_back({Key::Navigation, time});
  }
  for (auto id : prior.ids) {
    result.push_back({Key::Point, id});
  }
  result.push_back({Key::Gravity, 0});
  return result;
}
std::vector<Key> Keys(const Graph& graph) {
  Prior prior;
  prior.times = graph.times;
  prior.ids = graph.ids;
  return Keys(prior);
}
std::map<Key, int> Offsets(const std::vector<Key>& keys) {
  std::map<Key, int> offsets;
  int at = 0;
  for (const auto& key : keys) {
    if (!offsets.emplace(key, at).second) {
      throw std::invalid_argument("duplicate archive variable");
    }
    at += Width(key);
  }
  return offsets;
}
int Dimension(const std::vector<Key>& keys) {
  int n = 0;
  for (const auto& key : keys) {
    n += Width(key);
  }
  return n;
}
using Means = AnchorValues;
Means Values(const Graph& graph) {
  Means means;
  for (size_t i = 0; i < graph.times.size(); ++i) {
    means.nodes.emplace(graph.times[i], graph.nodes[i]);
  }
  for (size_t i = 0; i < graph.ids.size(); ++i) {
    means.points.emplace(graph.ids[i], graph.points[i]);
  }
  means.gravity = graph.gravity;
  means.fixed_pose_time = graph.fixed_pose_time;
  return means;
}
Prior Blueprint(const std::vector<Key>& keys, const Means& means) {
  Prior prior;
  bool seen_points = false, seen_gravity = false;
  for (const auto& key : keys) {
    if (key.kind == Key::Navigation) {
      if (seen_points || seen_gravity) {
        throw std::invalid_argument("archive variable layout");
      }
      prior.times.push_back(key.id);
      prior.nodes.push_back(means.nodes.at(key.id));
    } else if (key.kind == Key::Point) {
      if (seen_gravity) {
        throw std::invalid_argument("archive point after gravity");
      }
      seen_points = true;
      prior.ids.push_back(key.id);
      prior.points.push_back(means.points.at(key.id));
    } else {
      seen_gravity = true;
    }
  }
  if (!seen_gravity) {
    throw std::invalid_argument("archive gravity missing");
  }
  prior.gravity = means.gravity;
  prior.fixed_pose_time = means.fixed_pose_time;
  return prior;
}
Graph MeanGraph(const std::vector<Key>& keys, const Means& means) {
  const auto p = Blueprint(keys, means);
  Graph graph;
  graph.times = p.times;
  graph.nodes = p.nodes;
  graph.ids = p.ids;
  graph.points = p.points;
  graph.gravity = p.gravity;
  graph.fixed_pose_time = p.fixed_pose_time;
  return graph;
}
struct Local {
  std::vector<Key> keys;
  Factor factor;
};
Local Evaluate(const Prior& prior, const Means& means) {
  const auto keys = Keys(prior);
  auto graph = MeanGraph(keys, means);
  graph.prior = prior;
  return {keys, Assemble(graph, Linearize(graph, false))};
}
Local Combine(const std::vector<Local>& factors) {
  std::set<Key> union_keys;
  for (const auto& f : factors) {
    union_keys.insert(f.keys.begin(), f.keys.end());
  }
  std::vector<Key> keys(union_keys.begin(), union_keys.end());
  const auto target = Offsets(keys);
  std::vector<Factor> expanded;
  for (const auto& f : factors) {
    const auto source = Offsets(f.keys);
    Factor row;
    row.rows = Matrix::Zero(f.factor.rows.rows(), Dimension(keys));
    row.residual = f.factor.residual;
    row.evidence = f.factor.evidence;
    for (const auto& key : f.keys) {
      row.rows.middleCols(target.at(key), Width(key)) = f.factor.rows.middleCols(source.at(key), Width(key));
    }
    expanded.push_back(std::move(row));
  }
  return {keys, gaussian::combine(expanded, Dimension(keys))};
}
struct Split {
  Local conditional, marginal;
  std::vector<Key> frontal;
};
Split Project(const Local& factor, const std::set<Key>& frontal) {
  const auto offsets = Offsets(factor.keys);
  std::vector<int> nuisance, keep;
  std::vector<Key> kept_keys, front;
  for (const auto& key : factor.keys) {
    if (frontal.count(key)) {
      front.push_back(key);
    } else {
      kept_keys.push_back(key);
    }
    auto& columns = frontal.count(key) ? nuisance : keep;
    for (int k = 0; k < Width(key); ++k) {
      columns.push_back(offsets.at(key) + k);
    }
  }
  if (nuisance.empty()) {
    throw std::invalid_argument("archive projection has no frontal");
  }
  if (!factor.factor.rows.rows()) {
    Factor conditional, marginal;
    conditional.rows = Matrix::Zero(0, Dimension(factor.keys));
    conditional.residual = Vector::Zero(0);
    marginal.rows = Matrix::Zero(0, Dimension(kept_keys));
    marginal.residual = Vector::Zero(0);
    return {{factor.keys, std::move(conditional)}, {kept_keys, std::move(marginal)}, front};
  }
  Matrix scaled(factor.factor.rows.rows(), nuisance.size());
  for (size_t i = 0; i < nuisance.size(); ++i) {
    scaled.col(i) = factor.factor.rows.col(nuisance[i]);
    const double norm = scaled.col(i).norm();
    if (norm > 1e-30) {
      scaled.col(i) /= norm;
    }
  }
  Eigen::ColPivHouseholderQR<Matrix> qr(scaled);
  qr.setThreshold(kInheritedRankTolerance);
  const int rank = qr.rank();
  const Matrix transformed = qr.householderQ().adjoint() * factor.factor.rows;
  const Vector residual = qr.householderQ().adjoint() * factor.factor.residual;
  Factor conditional, marginal;
  conditional.rows = transformed.topRows(rank);
  conditional.residual = residual.head(rank);
  marginal.rows.resize(transformed.rows() - rank, keep.size());
  for (size_t i = 0; i < keep.size(); ++i) {
    marginal.rows.col(i) = transformed.col(keep[i]).tail(transformed.rows() - rank);
  }
  marginal.residual = residual.tail(residual.size() - rank);
  return {{factor.keys, std::move(conditional)}, {kept_keys, std::move(marginal)}, front};
}
Local Compact(Local local) {
  if (local.factor.rows.rows() > local.factor.rows.cols() + 1) {
    local.factor = CompressLocalFactor(local.factor);
  }
  return local;
}
}  // namespace

bool Archive::OwnsPoint(int64_t id) const { return owners_.count({Key::Point, id}) != 0; }
Stats Archive::GetStats() const {
  Stats result = stats_;
  result.live_records = records_.size();
  return result;
}
void Archive::Add(const AnchorValues& mean, const std::vector<Key>& keys, Factor factor,
                  const std::vector<Key>& frontal) {
  const auto source = Offsets(keys);
  std::set<Key> front(frontal.begin(), frontal.end());
  std::vector<Key> used, used_frontal;
  for (const auto& key : keys) {
    // Keep exact nonzeros only; no magnitude or depth threshold truncates correlations.
    const bool nonzero =
        factor.rows.rows() && factor.rows.middleCols(source.at(key), Width(key)).cwiseAbs().maxCoeff() != 0.;
    if (nonzero || front.count(key) || key.kind == Key::Gravity) {
      used.push_back(key);
      if (front.count(key)) {
        used_frontal.push_back(key);
      }
    }
  }
  if (used_frontal.empty()) {
    return;
  }
  Factor small;
  small.rows.resize(factor.rows.rows(), Dimension(used));
  small.residual = std::move(factor.residual);
  const auto dest = Offsets(used);
  for (const auto& key : used) {
    small.rows.middleCols(dest.at(key), Width(key)) = factor.rows.middleCols(source.at(key), Width(key));
  }
  auto record = std::make_shared<Record>();
  record->conditional = StoreLinearizedPrior(Blueprint(used, mean), std::move(small));
  record->frontal = used_frontal;
  const uint64_t index = next_id_++;
  for (const auto& key : used_frontal) {
    if (!owners_.emplace(key, index).second) {
      throw std::logic_error("archive variable already has an owner");
    }
  }
  const auto& coefficients = record->conditional.factor();
  stats_.coefficient_bytes += sizeof(double) * (coefficients.rows.size() + coefficients.residual.size());
  records_.emplace(index, std::move(record));
  ++stats_.captured;
}
void Archive::Consume(const std::set<uint64_t>& selected) {
  for (auto index : selected) {
    const auto record = records_.at(index);
    for (const auto& key : record->frontal) {
      const auto owner = owners_.find(key);
      if (owner == owners_.end() || owner->second != index) {
        throw std::logic_error("archive conditional ownership mismatch");
      }
      owners_.erase(owner);
    }
    const auto& coefficients = record->conditional.factor();
    stats_.coefficient_bytes -= sizeof(double) * (coefficients.rows.size() + coefficients.residual.size());
    records_.erase(index);
  }
}

void Archive::Capture(const Graph& graph, const Linearization& consumed, const std::vector<int64_t>& keep_times,
                      const std::vector<int64_t>& keep_ids) {
  const auto means = Values(graph);
  std::set<Key> retained{{Key::Gravity, 0}};
  for (auto time : keep_times) {
    retained.insert({Key::Navigation, time});
  }
  for (auto id : keep_ids) {
    retained.insert({Key::Point, id});
  }
  if (!graph.prior.empty()) {
    auto keys = Keys(graph);
    auto factor = Assemble(graph, consumed);
    std::set<Key> frontal;
    for (const auto& key : keys) {
      if (!retained.count(key)) {
        if (key.kind == Key::Point && owners_.count(key)) {
          const auto point = std::find(graph.ids.begin(), graph.ids.end(), key.id) - graph.ids.begin();
          const int column = 15 * graph.nodes.size() + 3 * point;
          if (factor.rows.rows() && factor.rows.middleCols(column, 3).cwiseAbs().maxCoeff() != 0.) {
            throw std::logic_error("unrestored historical point has nonzero active information");
          }
          // No observation and no prior membership: leave its existing archive
          // owner untouched. This is an exact structural zero, not a threshold.
          continue;
        }
        frontal.insert(key);
      }
    }
    if (!frontal.empty()) {
      auto split = Project({keys, std::move(factor)}, frontal);
      Add(means, split.conditional.keys, std::move(split.conditional.factor), split.frontal);
    }
    return;
  }
  // Initial history: independent map points are split before the dense factor
  // is allocated. Each small conditional keeps its source pose uncertainty.
  const int navigation = 15 * graph.nodes.size();
  Prior reduced_blueprint;
  reduced_blueprint.times = graph.times;
  reduced_blueprint.ids = keep_ids;
  const auto reduced_keys = Keys(reduced_blueprint);
  const auto reduced_offsets = Offsets(reduced_keys);
  std::vector<Factor> factors;
  for (const auto& block : consumed.blocks) {
    const bool dropped_point = block.point_index >= 0 && !retained.count({Key::Point, graph.ids[block.point_index]});
    Matrix state = block.state;
    Vector residual = block.residual;
    if (dropped_point) {
      Prior point_blueprint;
      point_blueprint.times = graph.times;
      point_blueprint.ids = {graph.ids[block.point_index]};
      const auto keys = Keys(point_blueprint);
      Factor row;
      row.rows = Matrix::Zero(state.rows(), navigation + 3 + 2);
      row.rows.leftCols(navigation) = state.leftCols(navigation);
      row.rows.middleCols(navigation, 3) = block.point;
      row.rows.rightCols<2>() = state.rightCols<2>();
      row.residual = residual;
      auto split = Project({keys, std::move(row)}, {{Key::Point, graph.ids[block.point_index]}});
      Add(means, split.conditional.keys, std::move(split.conditional.factor), split.frontal);
      state = split.marginal.factor.rows;
      residual = split.marginal.factor.residual;
    }
    Factor row;
    row.rows = Matrix::Zero(state.rows(), Dimension(reduced_keys));
    row.rows.leftCols(navigation) = state.leftCols(navigation);
    row.rows.rightCols<2>() = state.rightCols<2>();
    if (block.point_index >= 0 && !dropped_point) {
      row.rows.middleCols(reduced_offsets.at({Key::Point, graph.ids[block.point_index]}), 3) = block.point;
    }
    row.residual = residual;
    factors.push_back(std::move(row));
  }
  if (consumed.constant > 0.) {
    Factor constant;
    constant.rows = Matrix::Zero(1, Dimension(reduced_keys));
    constant.residual = Vector::Constant(1, std::sqrt(2. * consumed.constant));
    factors.push_back(std::move(constant));
  }
  std::set<Key> frontal;
  for (const auto& key : reduced_keys) {
    if (!retained.count(key)) {
      frontal.insert(key);
    }
  }
  if (!frontal.empty()) {
    auto split = Project({reduced_keys, gaussian::combine(factors, Dimension(reduced_keys))}, frontal);
    Add(means, split.conditional.keys, std::move(split.conditional.factor), split.frontal);
  }
}

Stats Archive::Restore(Graph& graph, const std::set<int64_t>* observed_ids) {
  if (graph.prior.empty()) {
    throw std::invalid_argument("restore requires the current joint prior");
  }
  std::set<Key> active;
  for (const auto& key : Keys(graph.prior)) {
    active.insert(key);
  }
  std::set<Key> requested;
  for (auto id : graph.ids) {
    const Key key{Key::Point, id};
    if (!active.count(key) && owners_.count(key) && (!observed_ids || observed_ids->count(id))) {
      requested.insert(key);
    }
  }
  if (requested.empty()) {
    return GetStats();
  }
  std::set<uint64_t> selected, visiting;
  std::function<void(Key)> visit = [&](Key key) {
    if (active.count(key)) {
      return;
    }
    const auto found = owners_.find(key);
    if (found == owners_.end()) {
      throw std::logic_error("historical separator has no conditional owner");
    }
    const uint64_t id = found->second;
    if (selected.count(id)) {
      return;
    }
    if (!visiting.insert(id).second) {
      throw std::logic_error("historical conditional cycle");
    }
    const auto& record = *records_.at(id);
    if (record.conditional.fixed_pose_time != graph.fixed_pose_time) {
      throw std::invalid_argument("historical archive belongs to another map gauge");
    }
    const std::set<Key> frontal(record.frontal.begin(), record.frontal.end());
    for (const auto& dependency : Keys(record.conditional)) {
      if (!frontal.count(dependency)) {
        visit(dependency);
      }
    }
    visiting.erase(id);
    selected.insert(id);
  };
  for (const auto& key : requested) {
    visit(key);
  }

  Means means = Values(graph);
  std::vector<Key> elimination_order;
  for (auto index : selected) {
    const auto& record = *records_.at(index);
    const auto& p = record.conditional;
    for (const auto& key : record.frontal) {
      if (!active.count(key)) {
        if (key.kind == Key::Navigation) {
          const auto it = std::find(p.times.begin(), p.times.end(), key.id);
          means.nodes[key.id] = p.nodes.at(it - p.times.begin());
        } else if (key.kind == Key::Point) {
          const auto it = std::find(p.ids.begin(), p.ids.end(), key.id);
          means.points[key.id] = p.points.at(it - p.ids.begin());
        }
      }
      if (!active.count(key) && !requested.count(key)) {
        elimination_order.push_back(key);
      }
    }
  }
  // Only the separator means are taken from the live graph. A reintroduced
  // point's legacy map coordinate is a seed, never a zero-variance observation.
  for (const auto& key : active) {
    if (key.kind == Key::Navigation) {
      const auto it = std::find(graph.times.begin(), graph.times.end(), key.id);
      if (it == graph.times.end()) {
        throw std::logic_error("active historical state missing");
      }
      means.nodes[key.id] = graph.nodes.at(it - graph.times.begin());
    } else if (key.kind == Key::Point) {
      const auto it = std::find(graph.ids.begin(), graph.ids.end(), key.id);
      if (it == graph.ids.end()) {
        throw std::logic_error("active historical point missing");
      }
      means.points[key.id] = graph.points.at(it - graph.ids.begin());
    }
  }
  std::map<uint64_t, Local> factors;
  std::map<Key, std::set<uint64_t>> incidence;
  uint64_t next_factor = 0;
  auto insert_factor = [&](Local factor) {
    const uint64_t index = next_factor++;
    for (const auto& key : factor.keys) {
      incidence[key].insert(index);
    }
    factors.emplace(index, std::move(factor));
  };
  for (auto index : selected) {
    insert_factor(Evaluate(records_.at(index)->conditional, means));
  }
  // Consume only inside this Archive copy. Replacements preserve correlations
  // between inactive ancestors and points that are now being restored.
  Consume(selected);
  for (const auto& key : elimination_order) {
    const auto found = incidence.find(key);
    if (found == incidence.end()) {
      continue;
    }
    const std::vector<uint64_t> incident(found->second.begin(), found->second.end());
    std::vector<Local> bucket;
    for (auto index : incident) {
      auto it = factors.find(index);
      for (const auto& variable : it->second.keys) {
        auto cell = incidence.find(variable);
        cell->second.erase(index);
        if (cell->second.empty()) {
          incidence.erase(cell);
        }
      }
      bucket.push_back(std::move(it->second));
      factors.erase(it);
    }
    auto split = Project(Combine(bucket), {key});
    Add(means, split.conditional.keys, std::move(split.conditional.factor), split.frontal);
    insert_factor(Compact(std::move(split.marginal)));
  }
  std::vector<Local> remaining;
  for (auto& [index, factor] : factors) {
    remaining.push_back(std::move(factor));
  }
  const auto conditional = Combine(remaining);
  for (const auto& key : conditional.keys) {
    if (!active.count(key) && !requested.count(key)) {
      throw std::logic_error("uneliminated historical nuisance variable");
    }
  }
  // Explicitly include the CURRENT marginal once. The selected conditionals
  // are complementary factors, not old copies of that marginal.
  auto combined = Compact(Combine({Evaluate(graph.prior, means), conditional}));
  auto prior = StoreLinearizedPrior(Blueprint(combined.keys, means), std::move(combined.factor));
  prior.visual_through = graph.prior.visual_through;
  prior.imu_left_through = graph.prior.imu_left_through;
  prior.consumed_visual = graph.prior.consumed_visual;

  // Conditional mode of resurrected points, holding the current separator
  // exact ONLY for this initial guess. The subsequent joint solve retains all
  // uncertainties and correlations in prior; no pose or point is fixed by it.
  const auto cols = Offsets(conditional.keys);
  Matrix point_rows(conditional.factor.rows.rows(), 3 * requested.size());
  int at = 0;
  Vector scaling(3 * requested.size());
  for (const auto& key : requested) {
    point_rows.middleCols<3>(at) = conditional.factor.rows.middleCols(cols.at(key), 3);
    const double norm = point_rows.middleCols<3>(at).norm() / std::sqrt(3.);
    scaling.segment<3>(at).setConstant(norm > 0. ? 1. / norm : 1.);
    at += 3;
  }
  const Matrix normalized = point_rows * scaling.asDiagonal();
  Vector seed(3 * requested.size());
  at = 0;
  for (const auto& key : requested) {
    const auto position = std::find(graph.ids.begin(), graph.ids.end(), key.id) - graph.ids.begin();
    seed.segment<3>(at) = graph.points.at(position) - means.points.at(key.id);
    at += 3;
  }
  Vector change = seed;
  if (normalized.rows()) {
    Eigen::JacobiSVD<Matrix> svd(normalized, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.info() != Eigen::Success) {
      throw std::runtime_error("historical conditional seed SVD failed");
    }
    Vector inverse = svd.singularValues();
    const double largest = inverse.size() ? inverse[0] : 0.;
    int rank = 0;
    for (int i = 0; i < inverse.size(); ++i) {
      if (inverse[i] > kInheritedRankTolerance * largest) {
        inverse[i] = 1. / inverse[i];
        ++rank;
      } else {
        inverse[i] = 0.;
      }
    }
    const Vector scaled_seed = scaling.cwiseInverse().asDiagonal() * seed;
    // An unobservable direction has no historical mean constraint. Preserve
    // the frontend seed there; add no fixed-depth information or depth cap.
    const Matrix range = svd.matrixV().leftCols(rank);
    const Vector null_seed = scaled_seed - range * (range.transpose() * scaled_seed);
    change =
        scaling.asDiagonal() *
        (svd.matrixV() * inverse.asDiagonal() * svd.matrixU().transpose() * (-conditional.factor.residual) + null_seed);
  }
  if (!change.allFinite()) {
    throw std::runtime_error("nonfinite historical conditional mean");
  }
  at = 0;
  for (const auto& key : requested) {
    const auto position = std::find(graph.ids.begin(), graph.ids.end(), key.id) - graph.ids.begin();
    Eigen::Vector3d point = means.points.at(key.id) + change.segment<3>(at);
    if (!point.allFinite()) {
      throw std::runtime_error("nonfinite resurrected map point");
    }
    auto inside_domain = [&](const Eigen::Vector3d& value) {
      for (const auto& visual : graph.visuals) {
        if (visual.point != static_cast<int>(position)) {
          continue;
        }
        const auto& node = graph.nodes.at(visual.node);
        const Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(node.R), C(visual.obs.Rci);
        const Eigen::Vector3d pc = C * R.transpose() * (value - Eigen::Map<const Eigen::Vector3d>(node.p)) +
                                   Eigen::Map<const Eigen::Vector3d>(visual.obs.tci);
        if (!pc.allFinite() || !(pc.z() > visual.obs.min_depth)) {
          return false;
        }
      }
      return true;
    };
    if (!inside_domain(point)) {
      point = graph.points.at(position);
      if (!inside_domain(point)) {
        throw std::domain_error("historical seed outside original projection domain");
      }
      ++stats_.seed_domain_fallbacks;
    }
    graph.points.at(position) = point;
    at += 3;
  }
  graph.prior = std::move(prior);
  stats_.restored += requested.size();
  ++stats_.queries;
  stats_.ancestors += elimination_order.size();
  return GetStats();
}
}  // namespace gpu_window::posterior::history
