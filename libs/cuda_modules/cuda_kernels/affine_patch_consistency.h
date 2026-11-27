#pragma once
#include <cmath>
#ifdef __CUDACC__
#define AFFINE_HD __host__ __device__
#else
#define AFFINE_HD
#endif
namespace affine_patch {
// Zero-mean photometric residual: EVERY affine Jacobian column must be
// centered AFTER multiplying the raw image gradient by its patch coordinate.
// Parameter order: [dxx,dyx,dxy,dyy,dx,dy], row-major map increment
// [[dxx,dxy],[dyx,dyy]]. No new noise, prior, or acceptance threshold.
AFFINE_HD inline void CenteredNormal9(const float*gx,const float*gy,float raw_mean_x,float raw_mean_y,
 const float*residual,const float*weights,float*normal,float*rhs){
 double means[6]={0.,0.,0.,0.,0.,0.};
 for(int i=0;i<81;++i){const double x=i%9-4,y=i/9-4,gx_raw=double(gx[i])+raw_mean_x,gy_raw=double(gy[i])+raw_mean_y;const double j[6]={x*gx_raw,x*gy_raw,y*gx_raw,y*gy_raw,gx_raw,gy_raw};for(int k=0;k<6;++k)means[k]+=j[k]/81.;}
 double H[36]={},b[6]={};
 for(int i=0;i<81;++i){const double x=i%9-4,y=i/9-4,gx_raw=double(gx[i])+raw_mean_x,gy_raw=double(gy[i])+raw_mean_y;double j[6]={x*gx_raw,x*gy_raw,y*gx_raw,y*gy_raw,gx_raw,gy_raw};for(int k=0;k<6;++k)j[k]-=means[k];for(int a=0;a<6;++a){b[a]+=double(weights[i])*j[a]*residual[i];for(int c=0;c<6;++c)H[6*a+c]+=double(weights[i])*j[a]*j[c];}}
 for(int i=0;i<36;++i)normal[i]=float(H[i]);for(int i=0;i<6;++i)rhs[i]=float(b[i]);
}
// Maximum increment over the actual patch support, in pixels. The old pixel
// convergence tolerance is retained; shear/rotation/scale cannot be skipped.
AFFINE_HD inline bool Finite6(const float*z){for(int k=0;k<6;++k){
#ifdef __CUDA_ARCH__
 if(!isfinite(z[k]))return false;
#else
 if(!std::isfinite(z[k]))return false;
#endif
 }return true;}
AFFINE_HD inline float MaxWarpSquared9(const float*z){float maximum=0.f;for(int x=-4;x<=4;x+=8)for(int y=-4;y<=4;y+=8){const float dx=z[4]+z[0]*x+z[2]*y,dy=z[5]+z[1]*x+z[3]*y,v=dx*dx+dy*dy;if(v>maximum)maximum=v;}return maximum;}
}
#undef AFFINE_HD
