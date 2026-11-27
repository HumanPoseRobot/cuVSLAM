#include <fstream>
#include <iostream>
#include <set>
#include "history_archive.h"
#include "joint_online.h"
#include "observed_history_support.h"
namespace p = gpu_window::posterior;
int main(int argc, char** argv) {
  try {
    if (argc != 4) { return 2; }
    const auto full = p::LoadGraph(argv[1]);
    auto live = p::LoadGraph(argv[2]);
    p::history::Archive archive;
    archive.Capture(full, p::Linearize(full, false), live.times, live.ids);
    int64_t requested = -1;
    for (auto id : full.ids) {
      if (archive.OwnsPoint(id)) { requested = id; break; }
    }
    if (requested < 0) { throw std::runtime_error("no archived test point"); }
    const auto original = live;
    live.ids.push_back(requested);
    live.points.emplace_back(1000., -500., 200.);
    const auto empty = p::history::ObservedPointIds(live);
    if (!empty.empty()) { throw std::runtime_error("transferred prior should have no new raw factors"); }
    const auto before = archive.GetStats();
    const auto skipped = archive.Restore(live, &empty);
    if (skipped.queries != before.queries || !archive.OwnsPoint(requested) ||
        (live.prior.factor().rows - original.prior.factor().rows).norm() != 0. ||
        live.points.back() != Eigen::Vector3d(1000., -500., 200.)) {
      throw std::runtime_error("unobserved query mutated history or graph");
    }
    // Retiring this zero-information seed must not replace its existing owner.
    archive.Capture(live, p::Linearize(live, false), original.times, original.ids);
    if (!archive.OwnsPoint(requested) || archive.GetStats().captured != before.captured) {
      throw std::runtime_error("unobserved retirement changed conditional ownership");
    }
    // Derive an observation mask from a synthetic index, then test restoration
    // of the archived prior alone. The dummy observation never enters a solve.
    gpu_window::Visual observation{};
    observation.point = int(live.ids.size()) - 1;
    live.visuals.push_back(observation);
    const auto support = p::history::ObservedPointIds(live);
    live.visuals.clear();
    if (support != std::set<int64_t>{requested}) { throw std::runtime_error("support mask ID mismatch"); }
    const auto restored = archive.Restore(live, &support);
    if (restored.restored != before.restored + 1 || archive.OwnsPoint(requested) ||
        std::find(live.prior.ids.begin(), live.prior.ids.end(), requested) == live.prior.ids.end()) {
      throw std::runtime_error("supported historical query failed");
    }
    const auto twice = archive.Restore(live, &support);
    if (twice.queries != restored.queries) { throw std::runtime_error("supported query not idempotent"); }
    std::ofstream out(argv[3]);
    out << "{\"passed\":true,\"unobserved_query_unchanged\":true,\"unobserved_retirement_owner_preserved\":true,\"supported_restore\":true,\"idempotent\":true}\n";
    std::cout << "OBSERVED_HISTORY_QUERY_PASS\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
