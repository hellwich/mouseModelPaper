cmake_minimum_required(VERSION 3.18)
project(mouse_ba LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# With CMake >= 3.18 and pybind11 >= 2.6, prefer modern FindPython.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

pybind11_add_module(mouse_ba src/mouse_ba_module.cpp)

target_link_libraries(mouse_ba PRIVATE Eigen3::Eigen Ceres::ceres)

target_include_directories(mouse_ba PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Helpful warnings
if (MSVC)
  target_compile_options(mouse_ba PRIVATE /W4)
else()
  target_compile_options(mouse_ba PRIVATE -Wall -Wextra -Wpedantic)
endif()
