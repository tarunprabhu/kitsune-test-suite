// Simple parallel memory copy test. This uses a Kokkos parallel_for with a
// straightforward range policy.

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <cstdlib>

#include <kitsune.h>

#include "copy.inc"

[[clang::noinline]]
static void test(kitsune::mobile_ptr<int> &dst,
                 const kitsune::mobile_ptr<int> &src, long n) {
  // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
  // We could try to find a way to make that type Kokkos-friendly, or just
  // wait until the [[kitsune::mobile_ptr]] attribute is implemented which
  // should make Kokkos happy.
  int *[[kitsune::mobile]] d = dst.get();
  const int *[[kitsune::mobile]] s = src.get();

  // clang-format off
  Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
    d[i] = s[i];
  });
  // clang-format on
}

int main(int argc, char *argv[]) {
  long err;
  Kokkos::initialize(argc, argv);
  {
    kitsune::mobile_ptr<int> dst;
    kitsune::mobile_ptr<int> src;
    long n = 2048;
    if (argc > 1)
      n = atol(argv[1]);

    setup(dst, src, n);
    test(dst, src, n);
    err = check(dst, src, n);
    teardown(dst, src);
  }
  Kokkos::finalize();

  return report(err);
}
