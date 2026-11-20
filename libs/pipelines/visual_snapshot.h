#pragma once
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include "camera/rig.h"
#include "camera/observation.h"
namespace cuvslam::pipelines::visual_audit {
template<class Landmarks>
inline void save(int64_t t,const camera::Rig& rig,const Landmarks& points,const std::vector<camera::Observation>& obs,const Isometry3T& before,const Isometry3T& after,bool ok,const std::string& suffix="") {
 const char* dir=std::getenv("VIO_VISUAL_SNAPSHOT_DIR");if(!dir){return;}
 const double seconds=t*1e-9;const char* low=std::getenv("VIO_DIAG_MIN_S");const char* high=std::getenv("VIO_DIAG_MAX_S");
 if((low&&seconds<std::atof(low))||(high&&seconds>std::atof(high))){return;}
 std::ofstream f(std::string(dir)+"/"+std::to_string(t)+suffix+".txt");f<<std::setprecision(17)<<t<<' '<<int(ok)<<' '<<rig.num_cameras<<'\n';
 for(const auto& pose:{before,after}){for(int i=0;i<4;++i){for(int j=0;j<4;++j){f<<pose.matrix()(i,j)<<' ';}}f<<'\n';}
 for(int c=0;c<rig.num_cameras;++c){const auto& pose=rig.camera_from_rig[c];for(int i=0;i<4;++i){for(int j=0;j<4;++j){f<<pose.matrix()(i,j)<<' ';}}f<<'\n';}
 for(const auto&o:obs){const auto p=points.find(o.id);if(p==points.end()||o.cam_id>=rig.num_cameras){continue;}Vector2T uv;const bool valid=rig.intrinsics[o.cam_id]->denormalizePoint(o.xy,uv);if(!valid){continue;}
 f<<o.id<<' '<<int(o.cam_id)<<' '<<p->second.transpose()<<' '<<o.xy.transpose()<<' '<<o.xy_info(0,0)<<' '<<o.xy_info(0,1)<<' '<<o.xy_info(1,1)<<' '<<uv.transpose()<<'\n';}
}
}
