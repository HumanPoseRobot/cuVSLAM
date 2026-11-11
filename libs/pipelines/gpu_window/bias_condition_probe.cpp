#include <iostream>
#include <cmath>
#include "gpu_bias_rw_math.h"
int main(){
 gpu_bias_rw::Data d{};d.dt=.033333333;
 for(int s=0;s<2;++s) for(int k=0;k<3;++k) d.R[s][4*k]=1.;
 for(int k=0;k<3;++k) {d.dR[4*k]=1.;d.JRg[4*k]=-d.dt;d.JVa[4*k]=-d.dt;d.JPa[4*k]=-.5*d.dt*d.dt;}
 for(int k=0;k<9;++k) d.imu_sqrt[10*k]=200.;
 for(int k=0;k<15;++k) d.prior_sqrt[16*k]=3.;
 std::vector<gpu_vi::Observation> obs;
 for(int i=0;i<30;++i){gpu_vi::Observation o{};o.Rci[0]=o.Rci[4]=o.Rci[8]=1.;o.point[0]=.1*(i%5-2);o.point[1]=.1*(i/5-2);o.point[2]=3+.1*(i%3);o.xy[0]=o.point[0]/o.point[2]+.02;o.xy[1]=o.point[1]/o.point[2]+.01;o.sqrt_info[0]=o.sqrt_info[3]=200.;o.min_depth=.05;obs.push_back(o);}
 auto eliminated=gpu_vi::OptimizeGpu(d,obs,1.5,20,true);
 for(double w:{10.,1e3,1e5,1e7}){
  for(int k=0;k<6;++k)d.rw_sqrt[7*k]=w;
  auto explicit_bias=gpu_bias_rw::OptimizeGpu(d,obs,1.5,20,true);
  double delta=0,rw=0;for(int k=0;k<24;++k)delta+=std::pow(eliminated.state[k]-explicit_bias.state[k],2);
  for(int k=0;k<6;++k)rw+=std::pow(explicit_bias.state[18+k]-explicit_bias.state[24+k],2);
  std::cout<<"rw_sqrt="<<w<<" shared_cost="<<eliminated.final_cost<<" explicit_cost="<<explicit_bias.final_cost<<" state_delta="<<std::sqrt(delta)<<" bias_difference="<<std::sqrt(rw)<<"\n";
 }
}
