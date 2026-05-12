#include <ceres/ceres.h>
#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

struct Camera {
  double fx{0}, fy{0}, cx{0}, cy{0};
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

// Templated projection (no distortion):
// Pc = R * Pw + t, u = fx * x/z + cx, v = fy * y/z + cy
template <typename T>
static inline bool Project(const Camera& cam, const T* Pw, T* uv) {
  Eigen::Matrix<T, 3, 1> P;
  P << Pw[0], Pw[1], Pw[2];

  Eigen::Matrix<T, 3, 3> R = cam.R.cast<T>();
  Eigen::Matrix<T, 3, 1> t = cam.t.cast<T>();
  Eigen::Matrix<T, 3, 1> Pc = R * P + t;

  // Guard against points behind camera or at z~0.
  if (Pc.z() <= T(1e-12)) {
    return false;
  }

  const T x = Pc.x() / Pc.z();
  const T y = Pc.y() / Pc.z();

  uv[0] = T(cam.fx) * x + T(cam.cx);
  uv[1] = T(cam.fy) * y + T(cam.cy);
  return true;
}

struct ReprojectionResidual {
  ReprojectionResidual(const Camera* cam_ptr,
                       const double u_obs,
                       const double v_obs,
                       const double sigma)
      : cam(cam_ptr), u(u_obs), v(v_obs), inv_sigma(1.0 / sigma) {}

  template <typename T>
  bool operator()(const T* const Pw, T* residuals) const {
    T uv_hat[2];
    if (!Project<T>(*cam, Pw, uv_hat)) {
      // If invalid, return large residuals to push point away.
      residuals[0] = T(1e6);
      residuals[1] = T(1e6);
      return true;
    }
    residuals[0] = (uv_hat[0] - T(u)) * T(inv_sigma);
    residuals[1] = (uv_hat[1] - T(v)) * T(inv_sigma);
    return true;
  }

  const Camera* cam;
  const double u;
  const double v;
  const double inv_sigma;
};

static std::unique_ptr<ceres::LossFunction> MakeLoss(const std::string& loss_name,
                                                    const double loss_scale) {
  if (loss_name.empty() || loss_name == "none" || loss_name == "None") {
    return nullptr;
  }
  if (loss_scale <= 0.0) {
    throw std::runtime_error("loss_scale must be > 0");
  }
  if (loss_name == "huber") {
    return std::make_unique<ceres::HuberLoss>(loss_scale);
  }
  if (loss_name == "cauchy") {
    return std::make_unique<ceres::CauchyLoss>(loss_scale);
  }
  if (loss_name == "soft_l1" || loss_name == "soft_lone") {
    return std::make_unique<ceres::SoftLOneLoss>(loss_scale);
  }
  if (loss_name == "arctan") {
    return std::make_unique<ceres::ArctanLoss>(loss_scale);
  }
  throw std::runtime_error("Unsupported loss: " + loss_name +
                           " (use none|huber|cauchy|soft_l1|arctan)");
}

static std::vector<Camera> ParseCameras(const py::dict& cameras_dict) {
  // Required keys: fx, fy, cx, cy (shape [C]), R (shape [C,3,3]), t (shape [C,3])
  auto get_arr = [&](const char* key) -> py::array {
    if (!cameras_dict.contains(key)) {
      throw std::runtime_error(std::string("cameras dict missing key '") + key + "'");
    }
    return py::cast<py::array>(cameras_dict[key]);
  };

  py::array fx_a = get_arr("fx");
  py::array fy_a = get_arr("fy");
  py::array cx_a = get_arr("cx");
  py::array cy_a = get_arr("cy");
  py::array R_a = get_arr("R");
  py::array t_a = get_arr("t");

  if (fx_a.ndim() != 1 || fy_a.ndim() != 1 || cx_a.ndim() != 1 || cy_a.ndim() != 1) {
    throw std::runtime_error("fx/fy/cx/cy must be 1D arrays of length C");
  }
  const ssize_t C = fx_a.shape(0);
  if (fy_a.shape(0) != C || cx_a.shape(0) != C || cy_a.shape(0) != C) {
    throw std::runtime_error("fx/fy/cx/cy must have the same length");
  }
  if (R_a.ndim() != 3 || R_a.shape(0) != C || R_a.shape(1) != 3 || R_a.shape(2) != 3) {
    throw std::runtime_error("R must have shape (C,3,3)");
  }
  if (t_a.ndim() != 2 || t_a.shape(0) != C || t_a.shape(1) != 3) {
    throw std::runtime_error("t must have shape (C,3)");
  }

  auto fx = fx_a.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto fy = fy_a.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto cx = cx_a.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto cy = cy_a.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto R = R_a.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto t = t_a.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();

  const auto fx_b = fx.unchecked<1>();
  const auto fy_b = fy.unchecked<1>();
  const auto cx_b = cx.unchecked<1>();
  const auto cy_b = cy.unchecked<1>();
  const auto R_b = R.unchecked<3>();
  const auto t_b = t.unchecked<2>();

  std::vector<Camera> cams;
  cams.reserve(static_cast<size_t>(C));

  for (ssize_t i = 0; i < C; ++i) {
    Camera cam;
    cam.fx = fx_b(i);
    cam.fy = fy_b(i);
    cam.cx = cx_b(i);
    cam.cy = cy_b(i);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        cam.R(r, c) = R_b(i, r, c);
      }
      cam.t(r) = t_b(i, r);
    }
    cams.push_back(cam);
  }

  return cams;
}

static py::dict SolvePointImpl(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& P_init,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& cam_idx,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& uv,
    const py::dict& cameras,
    const std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>& sigmas_opt,
    const double sigma_pix,
    const int max_num_iterations,
    const std::string& loss,
    const double loss_scale,
    const bool verbose) {

  if (P_init.ndim() != 1 || P_init.shape(0) != 3) {
    throw std::runtime_error("P_init must have shape (3,)");
  }
  if (cam_idx.ndim() != 1) {
    throw std::runtime_error("cam_idx must be 1D (N,)");
  }
  if (uv.ndim() != 2 || uv.shape(1) != 2) {
    throw std::runtime_error("uv must have shape (N,2)");
  }
  const ssize_t N = cam_idx.shape(0);
  if (uv.shape(0) != N) {
    throw std::runtime_error("cam_idx and uv must have the same length N");
  }
  if (N < 2) {
    throw std::runtime_error("Need at least 2 observations to triangulate");
  }
  if (sigma_pix <= 0.0) {
    throw std::runtime_error("sigma_pix must be > 0");
  }

  std::vector<Camera> cams = ParseCameras(cameras);
  const int C = static_cast<int>(cams.size());

  std::vector<double> sigmas;
  sigmas.resize(static_cast<size_t>(N), sigma_pix);
  if (sigmas_opt.has_value()) {
    const auto& s = sigmas_opt.value();
    if (s.ndim() != 1 || s.shape(0) != N) {
      throw std::runtime_error("sigmas must have shape (N,)");
    }
    auto sb = s.unchecked<1>();
    for (ssize_t i = 0; i < N; ++i) {
      const double si = sb(i);
      if (!(si > 0.0)) {
        throw std::runtime_error("All sigmas must be > 0");
      }
      sigmas[static_cast<size_t>(i)] = si;
    }
  }

  auto cam_idx_b = cam_idx.unchecked<1>();
  auto uv_b = uv.unchecked<2>();

  // Parameter block
  double Pw[3] = {P_init.unchecked<1>()(0), P_init.unchecked<1>()(1), P_init.unchecked<1>()(2)};

  // Create the loss function *before* creating the Problem so it outlives the Problem.
  // We pass raw pointers into ceres::Problem, so we must ensure lifetime safety.
  std::unique_ptr<ceres::LossFunction> loss_fn = MakeLoss(loss, loss_scale);

  // IMPORTANT: We typically reuse the same LossFunction pointer for all residual blocks.
  // By default, ceres::Problem may take ownership of LossFunction pointers, and if the
  // same pointer is attached to multiple residual blocks this can lead to double-free
  // when the Problem is destroyed. We therefore explicitly configure ownership.
  ceres::Problem::Options problem_options;
  problem_options.cost_function_ownership = ceres::TAKE_OWNERSHIP;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  ceres::Problem problem(problem_options);

  for (ssize_t i = 0; i < N; ++i) {
    const int ci = cam_idx_b(i);
    if (ci < 0 || ci >= C) {
      throw std::runtime_error("cam_idx contains out-of-range camera index");
    }
    const double u = uv_b(i, 0);
    const double v = uv_b(i, 1);

    auto* cost = new ceres::AutoDiffCostFunction<ReprojectionResidual, 2, 3>(
        new ReprojectionResidual(&cams[ci], u, v, sigmas[static_cast<size_t>(i)]));

    problem.AddResidualBlock(cost, loss_fn.get(), Pw);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = max_num_iterations;
  options.minimizer_progress_to_stdout = verbose;
  options.num_threads = 1;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Compute unweighted reprojection residuals in pixels
  py::array_t<double> residuals_px({N, ssize_t(2)});
  auto res_b = residuals_px.mutable_unchecked<2>();

  double sum_sq = 0.0;
  double max_norm = 0.0;
  double chi2 = 0.0;

  for (ssize_t i = 0; i < N; ++i) {
    const int ci = cam_idx_b(i);
    double uv_hat[2];
    const bool ok = Project<double>(cams[ci], Pw, uv_hat);

    double du = 0.0;
    double dv = 0.0;
    if (ok) {
      du = uv_hat[0] - uv_b(i, 0);
      dv = uv_hat[1] - uv_b(i, 1);
    } else {
      du = 1e6;
      dv = 1e6;
    }

    res_b(i, 0) = du;
    res_b(i, 1) = dv;

    const double n2 = du * du + dv * dv;
    sum_sq += n2;
    const double n = std::sqrt(n2);
    if (n > max_norm) max_norm = n;

    const double s = sigmas[static_cast<size_t>(i)];
    chi2 += (du / s) * (du / s) + (dv / s) * (dv / s);
  }

  const double rms = std::sqrt(sum_sq / (2.0 * static_cast<double>(N)));

  py::array_t<double> P_out({3});
  auto Pm = P_out.mutable_unchecked<1>();
  Pm(0) = Pw[0];
  Pm(1) = Pw[1];
  Pm(2) = Pw[2];

  py::dict out;
  out["P"] = P_out;
  out["success"] = summary.IsSolutionUsable();
  out["summary"] = summary.BriefReport();
  out["residuals_px"] = residuals_px;
  out["rms_px"] = rms;
  out["max_err_px"] = max_norm;
  out["chi2"] = chi2;
  out["num_obs"] = static_cast<int>(N);

  return out;
}

PYBIND11_MODULE(ceres_point_ba, m) {
  m.doc() = "Ceres point-only bundle adjustment for multi-view triangulation (no distortion).";

  m.def(
      "solve_point",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> P_init,
         py::array_t<int, py::array::c_style | py::array::forcecast> cam_idx,
         py::array_t<double, py::array::c_style | py::array::forcecast> uv,
         py::dict cameras,
         std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>> sigmas,
         double sigma_pix,
         int max_num_iterations,
         std::string loss,
         double loss_scale,
         bool verbose) {
        return SolvePointImpl(P_init, cam_idx, uv, cameras, sigmas, sigma_pix, max_num_iterations,
                             loss, loss_scale, verbose);
      },
      py::arg("P_init"),
      py::arg("cam_idx"),
      py::arg("uv"),
      py::arg("cameras"),
      py::arg("sigmas") = py::none(),
      py::arg("sigma_pix") = 1.0,
      py::arg("max_num_iterations") = 50,
      py::arg("loss") = "huber",
      py::arg("loss_scale") = 1.0,
      py::arg("verbose") = false,
      R"pbdoc(
Solve a single 3D point by minimizing reprojection error with Ceres.

Parameters
----------
P_init : (3,) float
    Initial 3D point in world coordinates.
cam_idx : (N,) int
    Camera indices for each observation.
uv : (N,2) float
    Observed image coordinates (u,v) for each observation.
cameras : dict
    Dict with keys: fx, fy, cx, cy (each (C,)), R ((C,3,3)), t ((C,3)).
    The model assumes Pc = R*Pw + t (world->camera), with undistorted images.
sigmas : (N,) float, optional
    Per-observation pixel sigma. Residuals are scaled by 1/sigma.
    If omitted, sigma_pix is used for all.
sigma_pix : float
    Default pixel sigma if sigmas is not given.
max_num_iterations : int
    Ceres max iterations.
loss : str
    Robust loss: 'none', 'huber', 'cauchy', 'soft_l1', 'arctan'.
loss_scale : float
    Robust loss scale parameter.
verbose : bool
    Print solver progress.

Returns
-------
dict
    P            : (3,) refined 3D point
    success      : bool
    summary      : str
    residuals_px : (N,2) reprojection residuals in pixels (unweighted)
    rms_px       : float RMS over 2N residual components
    max_err_px   : float max norm per observation
    chi2         : float sum of squared scaled residuals
    num_obs      : int
)pbdoc");
}
