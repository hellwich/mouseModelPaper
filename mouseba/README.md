# mouse_ba (Python front-end + Ceres back-end)

This project implements a model-based (pose-per-frame) bundle adjustment with:
- 3-camera reprojection residuals
- rigid mouse template points
- tiny per-joint slack offsets + priors
- soft cubic B-spline priors on translation and rotation

## Dependencies
- C++17 compiler
- CMake >= 3.18
- Eigen3
- Ceres Solver
- pybind11
- Python 3 with numpy/pandas

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

This produces a `mouse_ba` Python extension module in the build directory.

## Run example

From `mouse_ba/python`:

```bash
python example_run.py
```

Edit the file paths in `example_run.py` if needed.

## Notes
- Cameras are assumed to be in the form: `X_cam = R(q_cam) * X_world + t_cam`.
- Mouse pose is `X_world = R(q_mouse) * (B_j + delta_j) + p_mouse`.
- Distortion is not implemented yet (simulated case is distortion-free).
