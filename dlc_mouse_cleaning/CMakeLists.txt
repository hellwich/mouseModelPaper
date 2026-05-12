cmake_minimum_required(VERSION 3.18)
project(mouse_ba LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# With CMake >= 3.18 and pybind11 >= 2.6, prefer modern FindPython.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

# --- New: minimal point-only triangulation bundle adjustment module ---
pybind11_add_module(ceres_point_ba ${SRC_DIR}/ceres_point_ba_module.cpp)
target_link_libraries(ceres_point_ba PRIVATE Eigen3::Eigen Ceres::ceres)
target_include_directories(ceres_point_ba PRIVATE ${SRC_DIR})

# (Optional) Keep the old module if its source exists in src/.
if (EXISTS "${SRC_DIR}/mouse_ba_module.cpp")
  pybind11_add_module(mouse_ba ${SRC_DIR}/mouse_ba_module.cpp)
  target_link_libraries(mouse_ba PRIVATE Eigen3::Eigen Ceres::ceres)
  target_include_directories(mouse_ba PRIVATE ${SRC_DIR})
endif()

# Helpful warnings
if (MSVC)
  target_compile_options(ceres_point_ba PRIVATE /W4)
  if (TARGET mouse_ba)
    target_compile_options(mouse_ba PRIVATE /W4)
  endif()
else()
  target_compile_options(ceres_point_ba PRIVATE -Wall -Wextra -Wpedantic)
  if (TARGET mouse_ba)
    target_compile_options(mouse_ba PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()
