#ifndef KITSUNE_TEST_SUITE_FPCMP_H
#define KITSUNE_TEST_SUITE_FPCMP_H

// Utilties to compare floating point values and arrays of floats.

#ifdef __cplusplus

#include <cstdint>
#include <type_traits>

/// Compute the relative error between two floating point numbers and return it.
/// This is may be negative.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T relErr(T actual, T expected) {
  return (expected - actual) / expected;
}

/// Check if the absolute value of the relative error between the two floating
/// point numbers is less than @ref epsilon.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
bool checkRelErr(T actual, T expected, T epsilon) {
  return std::abs(relErr(actual, expected)) > epsilon;
}

/// Check if the relative error between every corresponding element in the two
/// arrays of floating point numbers is less than @ref epsilon. Returns the
/// number of elements for which the relative error exceeded @ref epsilon.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
size_t checkRelErr(T *actual, T *expected, size_t n, T epsilon) {
  size_t errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (checkRelErr(actual[i], expected[i], epsilon))
      ++errors;
  return errors;
}

#else // C

#include <math.h>

#endif // C

#endif // KITSUNE_TEST_SUITE_FPCMP_H
