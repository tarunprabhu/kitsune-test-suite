// Simple 2D vector addition test. Uses a nest of 2 parallel for loops.

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <kitsune.h>

#include "vecadd.inc"

[[clang::noinline]]
static void init(int *[[kitsune::mobile]] a, int *[[kitsune::mobile]] b,
                 int *[[kitsune::mobile]] c, long m, long n) {
  srand(7);
  for (long i = 0; i < m; ++i) {
    for (long j = 0; j < n; ++j) {
      long idx = i * n + j;
      a[idx] = rand();
      b[idx] = rand();
      c[idx] = 0;
    }
  }
}

[[clang::noinline]]
static long check(const int *[[kitsune::mobile]] c,
                  const int *[[kitsune::mobile]] a,
                  const int *[[kitsune::mobile]] b, long m, long n) {
  for (long i = 0; i < m; ++i) {
    for (long j = 0; j < n; ++j) {
      long idx = i * n + j;
      if (c[idx] != a[idx] + b[idx])
        return idx + 1;
    }
  }
  return 0;
}

[[clang::noinline]]
static void test(int *[[kitsune::mobile]] c,
                    const int *[[kitsune::mobile]] a,
                    const int *[[kitsune::mobile]] b, long m, long n) {
  // clang-format off
  forall (long i = 0; i < m; ++i) {
    forall (long j = 0; j < n; ++j) {
      long idx = i * n + j;
      c[idx] = a[idx] + b[idx];
    }
  }
  // clang-format on
}

int main(int argc, char *argv[]) {
  long err;
  int *[[kitsune::mobile]] a = NULL;
  int *[[kitsune::mobile]] b = NULL;
  int *[[kitsune::mobile]] c = NULL;
  long m = 64;
  long n = 32;
  if (argc > 1)
    m = atol(argv[1]);
  if (argc > 2)
    n = atol(argv[2]);

  a = (int *[[kitsune::mobile]])kitsune_mobile_alloc(m * n * sizeof(int));
  b = (int *[[kitsune::mobile]])kitsune_mobile_alloc(m * n * sizeof(int));
  c = (int *[[kitsune::mobile]])kitsune_mobile_alloc(m * n * sizeof(int));

  init(a, b, c, m, n);
  test(c, a, b, m, n);
  err = check(c, a, b, m, n);
  teardown(a, b, c);

  return report(err);
}
