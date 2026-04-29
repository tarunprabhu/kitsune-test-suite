// Simple 1D vector addition test. This uses a Kokkos parallel_for with a
// straightforward range policy.

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

#include <kitsune.h>

#include "vecadd.inc"

[[clang::noinline]]
static void setup(kitsune::mobile_ptr<int> &a, kitsune::mobile_ptr<int> &b,
                 kitsune::mobile_ptr<int> &c, long n) {
  a.alloc(n);
  b.alloc(n);
  c.alloc(n);

  srand(7);
  for (long i = 0; i < n; ++i) {
    a[i] = rand();
    b[i] = rand();
    c[i] = 0;
  }
}

[[clang::noinline]]
static void teardown(kitsune::mobile_ptr<int> &a, kitsune::mobile_ptr<int> &b,
                     kitsune::mobile_ptr<int> &c) {
  a.free();
  b.free();
  c.free();
}

[[clang::noinline]]
static long check(const kitsune::mobile_ptr<int> &c,
                  const kitsune::mobile_ptr<int> &a,
                  const kitsune::mobile_ptr<int> &b, long n) {
  for (long i = 0; i < n; ++i)
    if (c[i] != a[i] + b[i])
      return i + 1;
  return 0;
}

[[clang::noinline]]
static void vecadd1(kitsune::mobile_ptr<int> &c,
                    const kitsune::mobile_ptr<int> &a,
                    const kitsune::mobile_ptr<int> &b, long n) {
  // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
  // We could try to find a way to make that type Kokkos-friendly, or just
  // wait until the [[kitsune::mobile_ptr]] attribute is implemented which
  // should make Kokkos happy.
  const int *[[kitsune::mobile]] bufa = a.get();
  const int *[[kitsune::mobile]] bufb = b.get();
  int *[[kitsune::mobile]] bufc = c.get();

  // clang-format off
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
    bufc[i] = bufa[i] + bufb[i];
  });
  // clang-format on
}

int main(int argc, char *argv[]) {
  long err;
  Kokkos::initialize(argc, argv);
  {
    kitsune::mobile_ptr<int> a;
    kitsune::mobile_ptr<int> b;
    kitsune::mobile_ptr<int> c;
    long n = 2048;
    if (argc > 1)
      n = atol(argv[1]);

    setup(a, b, c, n);
    vecadd1(c, a, b, n);
    err = check(c, a, b, n);
    teardown(a, b, c);
  }
  Kokkos::finalize();

  return report(err);
}
