// Simple copy test using a single parallel for loop.

#include <cstdio>
#include <cstdlib>

#include <kitsune.h>

#include "copy.inc"

[[clang::noinline]]
static void test(kitsune::mobile_ptr<int> &dst,
                 const kitsune::mobile_ptr<int> &src, long n) {
  // clang-format off
  forall (long i = 0; i < n; ++i)
    dst[i] = src[i];
  // clang-format on
}

int main(int argc, char *argv[]) {
  long err;
  kitsune::mobile_ptr<int> dst;
  kitsune::mobile_ptr<int> src;
  long n = 2048;
  if (argc > 1)
    n = atol(argv[1]);

  setup(dst, src, n);
  test(dst, src, n);
  err = check(dst, src, n);
  teardown(dst, src);

  return report(err);
}
