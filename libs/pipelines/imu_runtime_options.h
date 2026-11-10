#pragma once
#include <string_view>
namespace cuvslam::pipelines::imu_runtime {
inline bool Enabled(const char* value) { return value && std::string_view(value) == "1"; }
inline bool Supported(int cameras) { return cameras == 2 || cameras == 4; }
inline bool JointEnabled(int cameras, const char* value) { return Supported(cameras) && Enabled(value); }
inline bool MetricEnabled(int cameras, bool async, const char* value) { return Supported(cameras) && !async && Enabled(value); }
}
