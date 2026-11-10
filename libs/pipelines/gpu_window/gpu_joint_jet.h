#pragma once
#include <cmath>
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
namespace gpu_vi {
struct Jet {
  double v, d;
  HD Jet(double value = 0, double derivative = 0) : v(value), d(derivative) {}
  HD Jet& operator+=(Jet x) {
    v += x.v;
    d += x.d;
    return *this;
  }
};
HD inline Jet operator+(Jet a, Jet b) { return {a.v + b.v, a.d + b.d}; }
HD inline Jet operator-(Jet a, Jet b) { return {a.v - b.v, a.d - b.d}; }
HD inline Jet operator-(Jet a) { return {-a.v, -a.d}; }
HD inline Jet operator*(Jet a, Jet b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
HD inline Jet operator/(Jet a, Jet b) { return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)}; }
HD inline bool operator<(Jet a, Jet b) { return a.v < b.v; }
HD inline bool operator<=(Jet a, Jet b) { return a.v <= b.v; }
HD inline Jet sqrt(Jet a) {
  double s = ::sqrt(a.v);
  return {s, s > 1e-20 ? a.d / (2 * s) : 0};
}
HD inline Jet sin(Jet a) { return {::sin(a.v), ::cos(a.v) * a.d}; }
HD inline Jet cos(Jet a) { return {::cos(a.v), -::sin(a.v) * a.d}; }
HD inline Jet atan2(Jet a, Jet b) {
  double den = a.v * a.v + b.v * b.v;
  return {::atan2(a.v, b.v), den > 1e-30 ? (b.v * a.d - a.v * b.d) / den : 0};
}
HD inline double sqrt(double x) { return ::sqrt(x); }
HD inline double sin(double x) { return ::sin(x); }
HD inline double cos(double x) { return ::cos(x); }
HD inline double atan2(double x, double y) { return ::atan2(x, y); }
}  // namespace gpu_vi
#undef HD
