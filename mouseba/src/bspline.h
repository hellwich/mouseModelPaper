#pragma once

#include <cmath>

namespace mouse_ba {

// Uniform cubic B-spline basis weights for u in [0, 1].
// Returns weights for control points i, i+1, i+2, i+3.

template <typename T>
inline void cubic_bspline_weights(const T& u, T w[4]) {
  const T u2 = u*u;
  const T u3 = u2*u;
  w[0] = (T(1) - T(3)*u + T(3)*u2 - u3) / T(6);              // (1-u)^3 / 6
  w[1] = (T(4) - T(6)*u2 + T(3)*u3) / T(6);                  // (3u^3 -6u^2 +4)/6
  w[2] = (T(1) + T(3)*u + T(3)*u2 - T(3)*u3) / T(6);          // (-3u^3+3u^2+3u+1)/6
  w[3] = u3 / T(6);
}

} // namespace mouse_ba
