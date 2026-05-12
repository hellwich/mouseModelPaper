#pragma once

// Use Ceres' rotation utilities & Jet-friendly math.
// This ensures AutoDiff works when T is ceres::Jet.
#include <ceres/rotation.h>
#include <ceres/jet.h>

namespace mouse_ba {

// Quaternion convention: [w, x, y, z]

template <typename T>
inline void quat_normalize(T q[4]) {
  const T n = ceres::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  q[0] /= n; q[1] /= n; q[2] /= n; q[3] /= n;
}

template <typename T>
inline void quat_conj(const T q[4], T qc[4]) {
  qc[0] = q[0];
  qc[1] = -q[1];
  qc[2] = -q[2];
  qc[3] = -q[3];
}

template <typename T>
inline void quat_mul(const T a[4], const T b[4], T out[4]) {
  out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
  out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
  out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
  out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

template <typename T>
inline void quat_rotate(const T q[4], const T v[3], T out[3]) {
  // out = q * [0,v] * conj(q)
  T p[4] = {T(0), v[0], v[1], v[2]};
  T qc[4]; quat_conj(q, qc);
  T tmp[4]; quat_mul(q, p, tmp);
  T res[4]; quat_mul(tmp, qc, res);
  out[0] = res[1]; out[1] = res[2]; out[2] = res[3];
}

template <typename T>
inline void so3_exp(const T w[3], T q[4]) {
  // Exponential map from so(3) (angle-axis) to quaternion.
  // Ceres implements this in a Jet-friendly way.
  ceres::AngleAxisToQuaternion(w, q);
}

template <typename T>
inline void so3_log(const T q_in[4], T w[3]) {
  // Log map from quaternion to so(3) (angle-axis).
  // Note: q and -q represent the same rotation; both are valid.
  ceres::QuaternionToAngleAxis(q_in, w);
}

} // namespace mouse_ba
