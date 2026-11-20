#pragma once
#include <unordered_map>
#include "camera/rig.h"
#include "camera/observation.h"
#include "pnp/visual_support.h"
#include "pipelines/gpu_window/frontend_guard_adapter.h"
namespace cuvslam::pipelines {
struct InitVisualAdmission { size_t reference=0,predicted=0; bool full_rank=false; bool accepted=false; };
// Current-frame observations are held out from the keyframe-only initializer.
// Reuse the active frontend thresholds: no speed/height cap or scene window.
inline InitVisualAdmission CheckInitPrediction(const camera::Rig& rig,
 const std::unordered_map<TrackId,Vector3T>& points,const std::vector<camera::Observation>& observations,
 const Isometry3T& reference_world_from_rig,const Isometry3T& predicted_world_from_rig,size_t required) {
 InitVisualAdmission result;
 if(required==0 || !reference_world_from_rig.matrix().allFinite() || !predicted_world_from_rig.matrix().allFinite()) return result;
 const auto reference=reference_world_from_rig.inverse(),prediction=predicted_world_from_rig.inverse();
 Eigen::MatrixXd jacobian(2*observations.size(),6);
 auto consistent=[&](const camera::Observation& o,const Vector3T& pc) {
  if(!pc.allFinite() || pc.z()<=.05f) return false;
  Vector2T px;
  if(!frontend_guard::actual_pixel_residual(*rig.intrinsics[o.cam_id],pc.hnormalized(),o.xy,px)) return false;
  const Vector2T e=pc.hnormalized()-o.xy;
  return frontend_guard::observation_consistent(9.*o.xy_info.cast<double>(),e.cast<double>(),px.cast<double>());
 };
 for(const auto& o:observations) {
  auto it=points.find(o.id);if(it==points.end() || o.cam_id>=rig.num_cameras)continue;
  const Vector3T ref=rig.camera_from_rig[o.cam_id]*reference*it->second;
  if(!consistent(o,ref))continue; ++result.reference;
  const Vector3T pc=rig.camera_from_rig[o.cam_id]*prediction*it->second;
  if(!consistent(o,pc))continue;
  const auto p=it->second.cast<double>();Eigen::Matrix<double,3,6> tangent=Eigen::Matrix<double,3,6>::Zero();
  tangent(0,1)=p.z();tangent(0,2)=-p.y();tangent(1,0)=-p.z();tangent(1,2)=p.x();tangent(2,0)=p.y();tangent(2,1)=-p.x();tangent.rightCols<3>().setIdentity();
  const double z=pc.z();Eigen::Matrix<double,2,3> project;project<<1./z,0.,-pc.x()/(z*z),0.,1./z,-pc.y()/(z*z);
  Eigen::LLT<Eigen::Matrix2d> factor(9.*o.xy_info.cast<double>());if(factor.info()!=Eigen::Success)continue;
  const Eigen::Matrix3d rotation=(rig.camera_from_rig[o.cam_id].linear()*prediction.linear()).cast<double>();
  jacobian.block<2,6>(2*result.predicted,0)=factor.matrixU()*project*rotation*tangent;++result.predicted;
 }
 result.full_rank=pnp::HasObservableVisualSupport(jacobian.topRows(2*result.predicted),result.predicted,required);
 result.accepted=result.reference>=required&&result.full_rank;
 return result;
}
inline bool BootstrapRetentionBudget(bool healthy,bool bootstrap,bool has_map,int64_t previous,int64_t current,int64_t budget) {
 return healthy&&bootstrap&&has_map&&previous>=0&&current>previous&&budget>0&&current-previous<=budget;
}
}
