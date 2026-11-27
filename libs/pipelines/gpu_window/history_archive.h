#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>
#include "joint_posterior.h"

namespace gpu_window::posterior {
// The same chart sealing used by Marginalize, exposed only to this internal
// archive. A factor on a local retraction is not already an anchored prior.
Prior StoreLinearizedPrior(Prior prior, Factor local);
Factor CompressLocalFactor(const Factor& factor);

namespace history {
struct Key {
  enum Kind { Navigation, Point, Gravity } kind;
  int64_t id;
  bool operator<(const Key& rhs) const { return std::tie(kind, id) < std::tie(rhs.kind, rhs.id); }
  bool operator==(const Key& rhs) const { return kind == rhs.kind && id == rhs.id; }
};
struct Stats {
  uint64_t captured = 0, restored = 0, queries = 0, ancestors = 0, seed_domain_fallbacks = 0;
  uint64_t live_records = 0, coefficient_bytes = 0;
};
struct AnchorValues {
  std::map<int64_t, State> nodes;
  std::map<int64_t, Eigen::Vector3d> points;
  Eigen::Vector3d gravity;
  int64_t fixed_pose_time;
};
struct Record {
  Prior conditional;
  std::vector<Key> frontal;
};
// Copies share immutable coefficient matrices but own their live-record and
// owner maps. A rejected online transaction cannot consume committed history.
class Archive {
public:
  void Capture(const Graph& graph, const Linearization& consumed, const std::vector<int64_t>& keep_times,
               const std::vector<int64_t>& keep_ids);
  Stats Restore(Graph& graph, const std::set<int64_t>* observed_ids = nullptr);
  Stats GetStats() const;
  bool OwnsPoint(int64_t id) const;
  size_t LiveRecords() const { return records_.size(); }
  const std::map<uint64_t, std::shared_ptr<const Record>>& RecordsForTest() const { return records_; }

private:
  void Add(const AnchorValues& mean, const std::vector<Key>& keys, Factor factor, const std::vector<Key>& frontal);
  void Consume(const std::set<uint64_t>& selected);
  uint64_t next_id_ = 0;
  Stats stats_;
  std::map<uint64_t, std::shared_ptr<const Record>> records_;
  std::map<Key, uint64_t> owners_;
};
}  // namespace history
}  // namespace gpu_window::posterior
