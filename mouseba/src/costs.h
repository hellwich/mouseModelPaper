#pragma once

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "so3.h"
#include "bspline.h"

namespace mouse_ba {

// Camera: world -> cam: Xc = R(qc)*Xw + tc
// Mouse pose: local -> world: Xw = R(qm)*(Bl + delta) + pm

struct ReprojectionCost {
  ReprojectionCost(double obs_x, double obs_y,
                   Eigen::Vector3d B_local,
                   double sqrt_weight)
      : obs_x_(obs_x), obs_y_(obs_y), B_local_(std::move(B_local)), sqrt_w_(sqrt_weight) {}

  template <typename T>
  bool operator()(const T* const cam_q,   // 4
                  const T* const cam_t,   // 3
                  const T* const cam_K,   // 4: fx,fy,cx,cy
                  const T* const mouse_q, // 4
                  const T* const mouse_t, // 3
                  const T* const delta,   // 3 (can be constant zero for core joints)
                  T* residuals) const {
    // Local point
    T pl[3];
    pl[0] = T(B_local_(0)) + delta[0];
    pl[1] = T(B_local_(1)) + delta[1];
    pl[2] = T(B_local_(2)) + delta[2];

    // To world
    T pw_rot[3];
    quat_rotate(mouse_q, pl, pw_rot);
    T pw[3] = {pw_rot[0] + mouse_t[0], pw_rot[1] + mouse_t[1], pw_rot[2] + mouse_t[2]};

    // To camera
    T pc_rot[3];
    quat_rotate(cam_q, pw, pc_rot);
    T pc[3] = {pc_rot[0] + cam_t[0], pc_rot[1] + cam_t[1], pc_rot[2] + cam_t[2]};

    // Project (no distortion)
    const T& fx = cam_K[0];
    const T& fy = cam_K[1];
    const T& cx = cam_K[2];
    const T& cy = cam_K[3];

    const T invz = T(1) / pc[2];
    const T u = fx * pc[0] * invz + cx;
    const T v = fy * pc[1] * invz + cy;

    residuals[0] = T(sqrt_w_) * (u - T(obs_x_));
    residuals[1] = T(sqrt_w_) * (v - T(obs_y_));
    return true;
  }

  static ceres::CostFunction* Create(double obs_x, double obs_y,
                                    const Eigen::Vector3d& B_local,
                                    double sqrt_weight) {
    return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 4, 3, 4, 4, 3, 3>(
        new ReprojectionCost(obs_x, obs_y, B_local, sqrt_weight));
  }

 private:
  double obs_x_;
  double obs_y_;
  Eigen::Vector3d B_local_;
  double sqrt_w_;
};

struct SlackPriorCost {
  explicit SlackPriorCost(double inv_sigma) : inv_sigma_(inv_sigma) {}

  template <typename T>
  bool operator()(const T* const delta, T* residuals) const {
    residuals[0] = T(inv_sigma_) * delta[0];
    residuals[1] = T(inv_sigma_) * delta[1];
    residuals[2] = T(inv_sigma_) * delta[2];
    return true;
  }

  static ceres::CostFunction* Create(double inv_sigma) {
    return new ceres::AutoDiffCostFunction<SlackPriorCost, 3, 3>(
        new SlackPriorCost(inv_sigma));
  }

  double inv_sigma_;
};

struct SlackSmoothCost {
  explicit SlackSmoothCost(double inv_sigma) : inv_sigma_(inv_sigma) {}

  template <typename T>
  bool operator()(const T* const delta_t,
                  const T* const delta_next,
                  T* residuals) const {
    residuals[0] = T(inv_sigma_) * (delta_next[0] - delta_t[0]);
    residuals[1] = T(inv_sigma_) * (delta_next[1] - delta_t[1]);
    residuals[2] = T(inv_sigma_) * (delta_next[2] - delta_t[2]);
    return true;
  }

  static ceres::CostFunction* Create(double inv_sigma) {
    return new ceres::AutoDiffCostFunction<SlackSmoothCost, 3, 3, 3>(
        new SlackSmoothCost(inv_sigma));
  }

  double inv_sigma_;
};

struct TranslationSplinePriorCost {
  TranslationSplinePriorCost(int frame_idx, int knot_interval, double inv_sigma)
      : t_(frame_idx), h_(knot_interval), inv_sigma_(inv_sigma) {}

  template <typename T>
  bool operator()(const T* const P0,
                  const T* const P1,
                  const T* const P2,
                  const T* const P3,
                  const T* const trans_t,
                  T* residuals) const {
    const T u = (T(t_) / T(h_)) - T(t_ / h_); // fractional part
    T w[4];
    cubic_bspline_weights(u, w);

    T sp[3];
    for (int k = 0; k < 3; ++k) {
      sp[k] = w[0]*P0[k] + w[1]*P1[k] + w[2]*P2[k] + w[3]*P3[k];
      residuals[k] = T(inv_sigma_) * (trans_t[k] - sp[k]);
    }
    return true;
  }

  static ceres::CostFunction* Create(int frame_idx, int knot_interval, double inv_sigma) {
    return new ceres::AutoDiffCostFunction<TranslationSplinePriorCost, 3, 3, 3, 3, 3, 3>(
        new TranslationSplinePriorCost(frame_idx, knot_interval, inv_sigma));
  }

  int t_;
  int h_;
  double inv_sigma_;
};

struct RotationSplinePriorCost {
  RotationSplinePriorCost(int frame_idx, int knot_interval, double inv_sigma)
      : t_(frame_idx), h_(knot_interval), inv_sigma_(inv_sigma) {}

  template <typename T>
  bool operator()(const T* const Phi0, // 3
                  const T* const Phi1,
                  const T* const Phi2,
                  const T* const Phi3,
                  const T* const q_frame, // 4
                  T* residuals) const {
    const T u = (T(t_) / T(h_)) - T(t_ / h_);
    T w[4];
    cubic_bspline_weights(u, w);

    T phi[3];
    for (int k = 0; k < 3; ++k) {
      phi[k] = w[0]*Phi0[k] + w[1]*Phi1[k] + w[2]*Phi2[k] + w[3]*Phi3[k];
    }

    // q_spline = exp(phi)
    T q_spline[4];
    so3_exp(phi, q_spline);

    // q_err = inv(q_spline) * q_frame
    T q_spline_conj[4];
    quat_conj(q_spline, q_spline_conj);
    T q_err[4];
    quat_mul(q_spline_conj, q_frame, q_err);

    T w_err[3];
    so3_log(q_err, w_err);

    residuals[0] = T(inv_sigma_) * w_err[0];
    residuals[1] = T(inv_sigma_) * w_err[1];
    residuals[2] = T(inv_sigma_) * w_err[2];
    return true;
  }

  static ceres::CostFunction* Create(int frame_idx, int knot_interval, double inv_sigma) {
    return new ceres::AutoDiffCostFunction<RotationSplinePriorCost, 3, 3, 3, 3, 3, 4>(
        new RotationSplinePriorCost(frame_idx, knot_interval, inv_sigma));
  }

  int t_;
  int h_;
  double inv_sigma_;
};

} // namespace mouse_ba
