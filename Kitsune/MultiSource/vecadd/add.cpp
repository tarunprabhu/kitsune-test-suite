#include <kitsune.h>

void vecadd(int *[[kitsune::mobile]] c, const int *[[kitsune::mobile]] a,
            const int *[[kitsune::mobile]] b, long n) {
  // clang-format off
  forall (long i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
  // clang-format on
}
