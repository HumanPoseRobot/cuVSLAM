#include "pnp/geometry_profile.h"
#include <gtest/gtest.h>
#include <fstream>
#include <iomanip>
TEST(GeometryCacheAudit, ExistingKeyframePoseCorrectionMustInvalidatePrior) {
  using namespace cuvslam;
  camera::FisheyeCameraModel lens(Vector2T(1200,1200),Vector2T(400,400),Vector2T(600,600),0,0,0,0);
  camera::Rig rig;rig.num_cameras=2;
  for(int c=0;c<2;++c){rig.intrinsics[c]=&lens;rig.camera_from_rig[c]=Isometry3T::Identity();}
  rig.camera_from_rig[1].translation().x()=-.2f;
  map::UnifiedMap history(4);
  std::unordered_map<TrackId,Vector3T> points,truth;
  std::vector<pipelines::Landmark> created;
  for(int k=0;k<30;++k){
    const Vector3T p(.14f*((k%6)-2),.13f*((k/6)-2),2.5f+.07f*k);
    truth[k+1]=p;points[k+1]=p+Vector3T(.0002f,-.0001f,.003f);
    created.push_back({TrackId(k+1),points[k+1]});
  }
  const auto observe=[&](const Isometry3T& T){
    std::vector<camera::Observation> out;
    for(const auto& [id,p]:truth){for(int c=0;c<2;++c){
      out.emplace_back(CameraId(c),id,(rig.camera_from_rig[c]*T*p).hnormalized(),Matrix2T::Identity()*(400.f*400.f/9.f));
    }}return out;
  };
  for(int k=0;k<2;++k){
    map::State state;state.rig_from_w=Isometry3T::Identity();state.rig_from_w.translation().x()=-.1f*k;
    const auto obs=observe(state.rig_from_w);
    if(k==0){state.rig_from_w.linear()=Eigen::AngleAxisf(.005f,Vector3T::UnitX()).toRotationMatrix();}
    history.add_keyframe(1000000000LL*k,state,{},obs,k==0?created:std::vector<pipelines::Landmark>{});
  }
  Isometry3T target=Isometry3T::Identity();target.translation()=Vector3T(-.25f,.01f,.02f);
  target.linear()=Eigen::AngleAxisf(.03f,Vector3T::UnitY()).toRotationMatrix();
  auto initial=target;initial.translation()+=Vector3T(.002f,.001f,.005f);
  const auto observations=observe(target);
  pnp::GeometryProfile cached;
  auto warm=initial;Matrix6T warm_info;
  ASSERT_TRUE(cached.refine(history,rig,points,observations,5,false,3000000000LL,warm,&warm_info));
  const auto graph=history.get_recent_submap(4);
  const auto stamp=graph.consecutive_keyframes.back().keyframe->time_ns();
  graph.consecutive_keyframes.front().keyframe->set_pose(Isometry3T::Identity());
  ASSERT_EQ(history.get_recent_submap(4).consecutive_keyframes.back().keyframe->time_ns(),stamp);
  auto stale=initial,fresh=initial;Matrix6T stale_info,fresh_info;
  pnp::GeometryProfile rebuilt;
  ASSERT_TRUE(cached.refine(history,rig,points,observations,5,false,3000000001LL,stale,&stale_info));
  ASSERT_TRUE(rebuilt.refine(history,rig,points,observations,5,false,3000000001LL,fresh,&fresh_info));
  const double pose_difference=(stale.matrix()-fresh.matrix()).norm();
  const double center_difference=(stale.inverse().translation()-fresh.inverse().translation()).norm();
  const double info_relative=(stale_info-fresh_info).norm()/fresh_info.norm();
  cached.reset();auto reset_pose=initial;Matrix6T reset_info;
  ASSERT_TRUE(cached.refine(history,rig,points,observations,5,false,3000000001LL,reset_pose,&reset_info));
  const double reset_difference=(reset_pose.matrix()-fresh.matrix()).norm();
  const double reset_information_difference=(reset_info-fresh_info).norm();
  EXPECT_EQ(reset_difference,0.);EXPECT_EQ(reset_information_difference,0.);
  const double angle_deg=Eigen::AngleAxisf(stale.linear()*fresh.linear().transpose()).angle()*180./3.141592653589793;
  std::ofstream out("/nas/shared/baidx/cuvslam_runs/geometry_uncertainty_audit_20261120/CACHE_AUDIT.json");
  out<<std::setprecision(17)<<"{\"same_latest_keyframe_timestamp\":true,\"pose_matrix_difference\":"<<pose_difference
     <<",\"camera_center_difference_m\":"<<center_difference<<",\"information_relative_difference\":"<<info_relative
     <<",\"rotation_difference_deg\":"<<angle_deg<<",\"explicit_reset_pose_difference\":"<<reset_difference<<",\"explicit_reset_information_difference\":"<<reset_information_difference<<",\"cached_error_to_truth\":"<<(stale.matrix()-target.matrix()).norm()<<",\"fresh_error_to_truth\":"
     <<(fresh.matrix()-target.matrix()).norm()<<"}\n";
  EXPECT_LT(pose_difference,1e-6)<<"Same current map/input/initial pose must not depend on a stale prior cache";
  EXPECT_LT(info_relative,1e-5);
}
