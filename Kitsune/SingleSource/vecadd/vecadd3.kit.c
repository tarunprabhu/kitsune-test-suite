// Simple 3D vector addition test. Uses a nest of 3 parallel for loops.

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <kitsune.h>

#include "vecadd.inc"

[[clang::noinline]]
static void init(int *[[kitsune::mobile]] a, int *[[kitsune::mobile]] b,
                 int *[[kitsune::mobile]] c, long m, long n, long p) {
  srand(7);
  for (long i = 0; i < m; ++i) {
    for (long j = 0; j < n; ++j) {
      for (long k = 0; k < p; ++k) {
        long idx = i * n * p + j * p + k;
        a[idx] = rand();
        b[idx] = rand();
        c[idx] = 0;
      }
    }
  }
}

[[clang::noinline]]
static long check(const int *[[kitsune::mobile]] c,
                  const int *[[kitsune::mobile]] a,
                  const int *[[kitsune::mobile]] b, long m, long n, long p) {
  for (long i = 0; i < m; ++i) {
    for (long j = 0; j < n; ++j) {
      for (long k = 0; k < p; ++k) {
        long idx = i * n * p + j * p + k;
        if (c[idx] != a[idx] + b[idx])
          return idx + 1;
      }
    }
  }
  return 0;
}

[[clang::noinline]]
static void test(int *[[kitsune::mobile]] c, const int *[[kitsune::mobile]] a,
                 const int *[[kitsune::mobile]] b, long m, long n, long p) {
  // clang-format off
  forall (long i = 0; i < m; ++i) {
    forall (long j = 0; j < n; ++j) {
      forall (long k = 0; k < p; ++k) {
        long idx = i * n * p + j * p + k;
        c[idx] = a[idx] + b[idx];
      }
    }
  }
  // clang-format on
}

int main(int argc, char *argv[]) {
  long err;
  int *[[kitsune::mobile]] a = NULL;
  int *[[kitsune::mobile]] b = NULL;
  int *[[kitsune::mobile]] c = NULL;
  long m = 32;
  long n = 16;
  long p = 8;
  if (argc > 1)
    m = atol(argv[1]);
  if (argc > 2)
    n = atol(argv[2]);
  if (argc > 3)
    p = atol(argv[3]);

  a = (int *[[kitsune::mobile]])kitsune_mobile_alloc(m * n * p * sizeof(int));
  b = (int *[[kitsune::mobile]])kitsune_mobile_alloc(m * n * p * sizeof(int));
  c = (int *[[kitsune::mobile]])kitsune_mobile_alloc(m * n * p * sizeof(int));

  init(a, b, c, m, n, p);
  test(c, a, b, m, n, p);
  err = check(c, a, b, m, n, p);
  teardown(a, b, c);

  return report(err);
}
