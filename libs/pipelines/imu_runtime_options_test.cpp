#include "imu_runtime_options.h"
#include <cassert>
int main(){ using namespace cuvslam::pipelines::imu_runtime;
for(int n: {2,4}) { assert(JointEnabled(n,"1")); assert(MetricEnabled(n,false,"1")); assert(!MetricEnabled(n,true,"1")); }
for(int n: {0,1,3,5,32}) { assert(!JointEnabled(n,"1")); assert(!MetricEnabled(n,false,"1")); }
for(const char* v: {static_cast<const char*>(nullptr),"0","","insert"}) { assert(!JointEnabled(2,v)); assert(!MetricEnabled(2,false,v)); }
}
