// Simple 1D vector addition test.

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <kitsune.h>

#include "vecadd.inc"

[[clang::noinline]]
static void init(int *[[kitsune::mobile]] a, int *[[kitsune::mobile]] b,
                 int *[[kitsune::mobile]] c, long n) {
  srand(7);
  for (long i = 0; i < n; ++i) {
    a[i] = rand();
    b[i] = rand();
    c[i] = 0;
  }
}

[[clang::noinline]]
static long check(const int *[[kitsune::mobile]] c,
                  const int *[[kitsune::mobile]] a,
                  const int *[[kitsune::mobile]] b, long n) {
  for (long i = 0; i < n; ++i)
    if (c[i] != a[i] + b[i])
      return i + 1;
  return 0;
}

[[clang::noinline]]
static void test(int *[[kitsune::mobile]] c, const int *[[kitsune::mobile]] a,
                 const int *[[kitsune::mobile]] b, long n) {
  // clang-format off
  forall (long i = 0; i < n; ++i)
    c[i] = a[i] + b[i];
  // clang-format on
}

int main(int argc, char *argv[]) {
  long err;
  int *[[kitsune::mobile]] a = NULL;
  int *[[kitsune::mobile]] b = NULL;
  int *[[kitsune::mobile]] c = NULL;
  long n = 2048;
  if (argc > 1)
    n = atol(argv[1]);

  a = (int *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(int));
  b = (int *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(int));
  c = (int *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(int));

  init(a, b, c, n);
  test(c, a, b, n);
  err = check(c, a, b, n);
  teardown(a, b, c);

  if (err)
    printf("ERROR at index %ld\n", err - 1);
  else
    printf("PASS\n");
  return err ? 1 : 0;
}
