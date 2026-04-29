// Simple saxpy test.

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

#include <kitsune.h>

#include "saxpy.inc"

[[clang::noinline]]
static void test(ElemT *[[kitsune::mobile]] r,
                 const ElemT *[[kitsune::mobile]] x,
                 const ElemT *[[kitsune::mobile]] y, long n) {
  // clang-format off
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const long i) {
     r[i] = A * x[i] + y[i];
  });
  // clang-format on
}

int main(int argc, char *argv[]) {
  long err;
  Kokkos::initialize(argc, argv);
  {
    ElemT *[[kitsune::mobile]] r = nullptr;
    ElemT *[[kitsune::mobile]] x = nullptr;
    ElemT *[[kitsune::mobile]] y = nullptr;
    long n = 2048;
    if (argc > 1)
      n = atol(argv[1]);

    setup(r, x, y, n);
    test(r, x, y, n);
    err = check(r, x, y, n);
    teardown(r, x, y);
  }
  Kokkos::finalize();

  return report(err);
}
