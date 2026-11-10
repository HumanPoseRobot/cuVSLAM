#include "window_backend.h"
#include "window_math.h"
#include "window_linear.h"
#include "common/log.h"
#include <map>
#include <set>
namespace cuvslam::sba_imu {
using WMatrix=Eigen::MatrixXd;using WVector=Eigen::VectorXd;using WinProblem=gpu_window::Problem;using WinResult=gpu_window::Result;using WinEvaluation=gpu_window::Evaluation;
using gpu_window::project_out;using gpu_window::compress_prior;using gpu_window::position_consistent;using gpu_window::Slots;using gpu_window::Dim;using gpu_window::square_root;using gpu_window::pinverse;using gpu_window::Optimize;using gpu_window::Evaluate;using gpu_window::assemble;using gpu_window::eliminate_first;
namespace {
using namespace gpu_window;
template<class M>void copy_matrix(double* p,const M&m){for(int i=0;i<m.rows();++i){for(int j=0;j<m.cols();++j){p[i*m.cols()+j]=m(i,j);}}}
gpu_window::State pack_state(const Pose&p){gpu_window::State s;copy_matrix(s.R,p.w_from_imu.linear());copy_matrix(s.p,p.w_from_imu.translation());copy_matrix(s.v,p.velocity);copy_matrix(s.bg,p.gyro_bias);copy_matrix(s.ba,p.acc_bias);return s;}
void update_pose(Pose&p,const float*x){auto a=pack_state(p);double z[15];for(int k=0;k<15;++k){z[k]=x[k];}double R[9],t[3],v[3];gpu_window::pose(a,z,R,t,v);for(int i=0;i<3;++i){p.w_from_imu.translation()[i]=t[i];p.velocity[i]=v[i];p.gyro_bias[i]+=z[9+i];p.acc_bias[i]+=z[12+i];for(int j=0;j<3;++j){p.w_from_imu.linear()(i,j)=R[3*i+j];}}}
template<int N>void whiten(double* dst,const Eigen::Matrix<double,N,N>&m){WMatrix W;WVector c;square_root(m,WVector::Zero(N),W,c);copy_matrix(dst,W);}
struct Node {int64_t t;Pose pose;std::vector<camera::Observation> observations;};
struct Graph {std::vector<Node> nodes;std::map<TrackId,Vector3T>points;std::array<gpu_window::State,Slots>anchors{};WMatrix W=WMatrix::Zero(Dim,Dim);WVector c=WVector::Zero(Dim);};
WinProblem pack(Graph&g,const imu::ImuCalibration&cal,const camera::Rig&rig,const Vector3T&gravity,int iterations,double huber,std::vector<TrackId>&ids){
 int invalid_depth=0;
 for(auto&node:g.nodes){auto&observations=node.observations;observations.erase(std::remove_if(observations.begin(),observations.end(),[&](const auto&o){auto it=g.points.find(o.id);if(it==g.points.end()||o.cam_id>=rig.num_cameras){++invalid_depth;return true;}const Vector3T pc=rig.camera_from_rig[o.cam_id]*cal.rig_from_imu()*node.pose.w_from_imu.inverse()*it->second;if(!pc.allFinite()||pc.z()<=.05f){++invalid_depth;return true;}return false;}),observations.end());}
 if(invalid_depth){TraceMessage("WINDOW_INVALID_DEPTH removed=%d",invalid_depth);}
 WinProblem p;p.count=g.nodes.size();p.iterations=iterations;p.huber=huber;p.anchors=g.anchors;copy_matrix(p.W.data(),g.W);copy_matrix(p.offset.data(),g.c);
 for(int i=0;i<Slots;++i){if(i<p.count){p.nodes[i]=pack_state(g.nodes[i].pose);}else{p.anchors[i]=p.nodes[i];for(int j=0;j<15;++j){p.W[(i*15+j)*Dim+i*15+j]=1;}}}
 std::map<TrackId,int> counts;for(auto&n:g.nodes){for(auto&o:n.observations){++counts[o.id];}}std::map<TrackId,int>index;for(auto&kv:g.points){if(counts[kv.first]>=2){index[kv.first]=ids.size();ids.push_back(kv.first);p.points.push_back({kv.second[0],kv.second[1],kv.second[2]});}}
 for(int i=0;i<p.count;++i){for(auto&o:g.nodes[i].observations){auto it=index.find(o.id);if(it==index.end()){continue;}Visual v;v.node=i;v.point=it->second;copy_matrix(v.obs.xy,o.xy);auto ci=Isometry3T(rig.camera_from_rig[o.cam_id]*cal.rig_from_imu());copy_matrix(v.obs.Rci,ci.linear());copy_matrix(v.obs.tci,ci.translation());whiten<2>(v.obs.sqrt_info,9.*o.xy_info.template cast<double>());v.obs.min_depth=.05;p.visuals.push_back(v);}}
 for(int i=0;i+1<p.count;++i){auto&l=g.nodes[i].pose;auto&r=g.nodes[i+1].pose;auto&pi=l.preintegration;Vector3T db,da;pi.GetDeltaBias(l.gyro_bias,l.acc_bias,db,da);if(db.norm()>1e-4||da.norm()>1e-4){pi.SetNewBias(l.gyro_bias,l.acc_bias);pi.Reintegrate(cal);}Edge e;e.left=i;e.right=i+1;auto&d=e.data;auto a=pack_state(l),b=pack_state(r);std::copy(a.R,a.R+9,d.R[0]);std::copy(b.R,b.R+9,d.R[1]);std::copy(a.p,a.p+3,d.p[0]);std::copy(b.p,b.p+3,d.p[1]);std::copy(a.v,a.v+3,d.v[0]);std::copy(b.v,b.v+3,d.v[1]);copy_matrix(d.bg,l.gyro_bias);copy_matrix(d.ba,l.acc_bias);copy_matrix(d.bg0,pi.GetOriginalGyroBias());copy_matrix(d.ba0,pi.GetOriginalAccBias());copy_matrix(d.dR,pi.dR);copy_matrix(d.dV,pi.dV);copy_matrix(d.dP,pi.dP);copy_matrix(d.JRg,pi.JRg);copy_matrix(d.JVg,pi.JVg);copy_matrix(d.JVa,pi.JVa);copy_matrix(d.JPg,pi.JPg);copy_matrix(d.JPa,pi.JPa);copy_matrix(d.gravity,gravity);d.dt=pi.dT_s;Matrix9T inf;pi.InfoMatrix(inf);whiten<9>(d.imu_sqrt,.002*inf.cast<double>());Matrix3T bg,ba;pi.InfoGyroRWMatrix(bg);pi.InfoAccRWMatrix(ba);Eigen::Matrix<double,6,6>rw=Eigen::Matrix<double,6,6>::Zero();rw.topLeftCorner<3,3>()=.002*bg.cast<double>();rw.bottomRightCorner<3,3>()=.002*ba.cast<double>();whiten<6>(e.rw,rw);p.edges.push_back(e);}
 return p;
}
double cost(const WinEvaluation&e,double huber){double s=0;for(auto x:e.edge){s+=.5*x*x;}for(auto x:e.prior){s+=.5*x*x;}for(size_t i=0;i<e.visual.size();i+=2){double n=std::hypot(e.visual[i],e.visual[i+1]);s+=n>huber?huber*n-.5*huber*huber:.5*n*n;}return s;}

}
struct WindowBackend::Impl {Graph active,pending;bool have_pending=false;Vector3T gravity;imu::ImuCalibration cal;camera::Rig rig;int iterations=7;double huber=1;size_t commits=0;};
WindowBackend::WindowBackend():impl_(std::make_unique<Impl>()){}
WindowBackend::~WindowBackend()=default;
void WindowBackend::discard_pending(){impl_->have_pending=false;impl_->pending=Graph{};}
void WindowBackend::reset(){impl_=std::make_unique<Impl>();}
bool WindowBackend::solve(const FrameInput& input,Pose& result){
 auto& map=input.map;const auto&cal=input.calibration;const auto&rig=input.rig;
 const auto&storage=input.measurements;const auto&landmarks=input.landmarks;
 const auto&obs=input.observations;const auto&gravity=input.gravity;const auto&settings=input.settings;
 const auto&previous=input.previous;const auto&tracked=input.tracked;
 const int64_t previous_t=input.previous_time_ns,t=input.time_ns;

 auto&d=*impl_;d.have_pending=false;if(previous_t<0||t<=previous_t){reset();return false;}if(!d.active.nodes.empty()&&(d.gravity-gravity).norm()>1e-4){reset();return solve(input,result);}if(!d.active.nodes.empty()) {
 auto recent=map.get_recent_submap(20);bool changed=false;
 for(auto&k:recent.consecutive_keyframes){for(auto&n:d.active.nodes){if(k.keyframe->time_ns()!=n.t){continue;}auto state=k.keyframe->get_state();Isometry3T mapped=state.rig_from_w.inverse()*cal.rig_from_imu();changed=changed||(mapped.matrix()-n.pose.w_from_imu.matrix()).norm()>1e-4f||(state.velocity-n.pose.velocity).norm()>1e-4f||(state.gyro_bias-n.pose.gyro_bias).norm()>1e-4f||(state.acc_bias-n.pose.acc_bias).norm()>1e-4f;}}
 if(changed){TraceMessage("WINDOW_RESET reason=external_map_correction t=%lld",(long long)t);reset();return solve(input,result);}
 }
 d.cal=cal;d.rig=rig;d.gravity=gravity;d.iterations=settings.max_iteration;d.huber=settings.robustifier_scale;
 if(d.active.nodes.empty()){Node n{previous_t,previous,{}};
 auto seed=map.get_recent_submap(1);if(!seed.consecutive_keyframes.empty()) {
   auto k=seed.consecutive_keyframes.back().keyframe;auto state=k->get_state();
   n.t=k->time_ns();n.pose.w_from_imu=state.rig_from_w.inverse()*cal.rig_from_imu();
   n.pose.velocity=state.velocity;n.pose.gyro_bias=state.gyro_bias;n.pose.acc_bias=state.acc_bias;
   for(auto&row:seed.landmark_and_obs){for(auto&entry:row){auto point=entry.landmark->get_pose();if(!point){continue;}d.active.points.emplace(entry.landmark->id(),*point);for(auto&o:entry.observations){n.observations.push_back(o);}}}
 }
 d.active.nodes.push_back(n);d.active.anchors[0]=pack_state(n.pose);WVector diag(15);diag<<1e4,1e4,1e4,1e4,1e4,1e4,25,25,25,1e4,1e4,1e4,25,25,25;d.active.W.topLeftCorner(15,15)=diag.cwiseSqrt().asDiagonal();}
 Graph g=d.active;auto&last=g.nodes.back();if(t<=last.t){reset();return false;}last.pose.preintegration=IMUPreintegration(cal,storage,last.pose.gyro_bias,last.pose.acc_bias,last.t,t);double dt=(t-last.t)*1e-9;if(std::abs(last.pose.preintegration.dT_s-dt)>1e-5||last.pose.preintegration.empty()){reset();return false;}
 Pose predicted=tracked;Node current{t,predicted,{}};const Isometry3T rig_from_world=cal.rig_from_imu()*predicted.w_from_imu.inverse();std::set<std::pair<TrackId,int>>seen;
 for(auto&o:obs){auto it=landmarks.find(o.id);if(it==landmarks.end()||o.cam_id>=rig.num_cameras||!seen.emplace(o.id,int(o.cam_id)).second){continue;}auto point=g.points.count(o.id)?g.points[o.id]:it->second;if((rig.camera_from_rig[o.cam_id]*rig_from_world*point).z()<=.05){continue;}g.points.emplace(o.id,point);current.observations.push_back(o);}
 if(current.observations.size()<size_t(settings.min_observations)){return false;}g.nodes.push_back(current);std::vector<TrackId>ids;WinResult solved;
 double after=0;
 for(int pass=0;pass<2;++pass){
   ids.clear();auto p=pack(g,cal,rig,gravity,settings.max_iteration,settings.robustifier_scale,ids);
   if(p.visuals.size()<size_t(settings.min_observations)||g.nodes.back().observations.size()<size_t(settings.min_observations)){return false;}
   solved=Optimize(p);auto evaluation=Evaluate(p,solved);after=cost(evaluation,p.huber);
   if(!std::isfinite(after)||after>solved.before+1e-4*std::max(1.f,solved.before)){
     TraceMessage("WINDOW_REJECT t=%lld reason=objective before=%.9g after=%.9g",(long long)t,solved.before,after);return false;
   }
   for(size_t i=0;i<g.nodes.size();++i){update_pose(g.nodes[i].pose,solved.x.data()+15*i);}
   for(size_t i=0;i<ids.size();++i){for(int k=0;k<3;++k){g.points[ids[i]][k]+=solved.points[3*i+k];}}
   int removed=0,behind=0;
   // Classify individual reprojections, rather than rejecting a whole window
   // because one triangulated outlier crosses the camera plane. 9.21 is the
   // 99% two-dimensional Gaussian residual threshold under the same visual info.
   for(auto&node:g.nodes){auto&observations=node.observations;
     observations.erase(std::remove_if(observations.begin(),observations.end(),[&](const auto&o){
       auto it=g.points.find(o.id);if(it==g.points.end()){++removed;return true;}
       const Vector3T pc=rig.camera_from_rig[o.cam_id]*cal.rig_from_imu()*node.pose.w_from_imu.inverse()*it->second;
       if(!pc.allFinite()||pc.z()<=.05f){++removed;++behind;return true;}
       const Vector2T error=pc.hnormalized()-o.xy;
       const double score=error.cast<double>().dot(9.*o.xy_info.template cast<double>()*error.cast<double>());
       if(!std::isfinite(score)||score>9.21){++removed;return true;}return false;
     }),observations.end());
   }
   TraceMessage("WINDOW_PRUNE t=%lld pass=%d removed=%d behind=%d current_obs=%zu",(long long)t,pass,removed,behind,g.nodes.back().observations.size());
   if(g.nodes.back().observations.size()<size_t(settings.min_observations)){return false;}
   if(!removed){break;}
 }
 // Restore the existing repaired backend's same-timestamp position-consistency gate.
 // Tracking and window factors share measurements: this is an engineering gate,
 // not an independent-sensor chi-square claim. Reject the entire transaction,
 // retaining the old window and the separately computed foreground estimate.
 Eigen::MatrixXd tracking_info=tracked.info.cast<double>();
 Eigen::LDLT<Eigen::MatrixXd> posterior(tracking_info);
 if(posterior.info()!=Eigen::Success||!posterior.isPositive()||posterior.vectorD().minCoeff()<=0){
   TraceMessage("WINDOW_GATE t=%lld accept=0 reason=invalid_tracking_covariance",(long long)t);return false;
 }
 Eigen::MatrixXd select=Eigen::MatrixXd::Zero(15,3);select.block<3,3>(3,0).setIdentity();
 Eigen::Matrix3d local_cov=posterior.solve(select).block<3,3>(3,0);
 Eigen::Matrix3d rotation=tracked.w_from_imu.linear().cast<double>();
 Eigen::Matrix3d covariance=rotation*local_cov*rotation.transpose();
 Eigen::Vector3d delta=(g.nodes.back().pose.w_from_imu.translation()-tracked.w_from_imu.translation()).cast<double>();
 double score=0;bool accept=position_consistent(covariance,delta,score);
 TraceMessage("WINDOW_GATE t=%lld accept=%d score=%.9g dp=%.9g",(long long)t,int(accept),score,delta.norm());
 if(!accept){return false;}
 // Re-express final normal equations in the newly anchored tangent coordinates.
 ids.clear();auto final=pack(g,cal,rig,gravity,settings.max_iteration,settings.robustifier_scale,ids);WinResult zero;zero.points.resize(3*ids.size());auto linear=Evaluate(final,zero);auto reduced=assemble(final,linear,false);int n=15*(g.nodes.size()-1);WMatrix final_rows;WVector final_residual;project_out(reduced.rows.leftCols(n),reduced.rows.middleCols(n,15),reduced.residual,final_rows,final_residual);g.nodes.back().pose.info=(final_rows.transpose()*final_rows).cast<float>();result=g.nodes.back().pose;
 if(!result.info.allFinite()||!result.velocity.allFinite()||!result.w_from_imu.matrix().allFinite()){return false;}if(input.accept_candidate&&!input.accept_candidate(result,g.points)){return false;}d.pending=std::move(g);d.have_pending=true;
 TraceMessage("WINDOW_SOLVE t=%lld states=%zu points=%zu obs=%zu iterations=%d before=%.9g after=%.9g",(long long)t,d.pending.nodes.size(),ids.size(),final.visuals.size(),solved.iterations,solved.before,after);return true;
}
void WindowBackend::commit(map::UnifiedMap&map,const imu::ImuCalibration&cal,int64_t t){auto&d=*impl_;if(!d.have_pending||d.pending.nodes.back().t!=t){return;}Graph g=std::move(d.pending);d.have_pending=false;
 // Defer all map writes until marginalization has completed successfully.
 const auto publish_nodes=g.nodes;const auto publish_points=g.points;
 auto publish=[&](){auto sub=map.get_recent_submap(20);for(auto&k:sub.consecutive_keyframes){for(auto&n:publish_nodes){if(k.keyframe->time_ns()==n.t){k.keyframe->set_state(map::State{(n.pose.w_from_imu*cal.rig_from_imu().inverse()).inverse(),n.pose.velocity,n.pose.acc_bias,n.pose.gyro_bias});}}}for(auto&row:sub.landmark_and_obs){for(auto&v:row){auto it=publish_points.find(v.landmark->id());if(it!=publish_points.end()){v.landmark->set_pose(it->second);}}}};
 if(g.nodes.size()==Slots){std::vector<TrackId>ids;auto p=pack(g,d.cal,d.rig,d.gravity,d.iterations,d.huber,ids);WinResult zero;zero.points.resize(3*ids.size());auto reduced=eliminate_first(assemble(p,Evaluate(p,zero),true),p.count);const int n=15*(p.count-1);WMatrix W;WVector c;compress_prior(reduced.rows.leftCols(n),reduced.residual,W,c);std::set<TrackId>remove;for(auto&o:g.nodes.front().observations){remove.insert(o.id);}g.nodes.erase(g.nodes.begin());for(auto&node:g.nodes){auto&v=node.observations;v.erase(std::remove_if(v.begin(),v.end(),[&](auto&o){return remove.count(o.id);}),v.end());}for(auto id:remove){g.points.erase(id);}g.W.setZero();g.c.setZero();g.W.topLeftCorner(n,n)=W;g.c.head(n)=c;for(int i=0;i<p.count-1;++i){g.anchors[i]=pack_state(g.nodes[i].pose);}TraceMessage("WINDOW_MARG t=%lld retained=%zu retired_points=%zu prior_norm=%.9g linear_norm=%.9g",(long long)t,g.nodes.size(),remove.size(),g.W.norm(),g.c.norm());}
 publish();
 // Drop unreferenced temporary points to keep memory bounded.
 std::set<TrackId>used;for(auto&n:g.nodes){for(auto&o:n.observations){used.insert(o.id);}}for(auto it=g.points.begin();it!=g.points.end();){if(!used.count(it->first)){it=g.points.erase(it);}else{++it;}}d.active=std::move(g);++d.commits;TraceMessage("WINDOW_COMMIT t=%lld keyframes=%zu commits=%zu",(long long)t,d.active.nodes.size(),d.commits);
}
}
