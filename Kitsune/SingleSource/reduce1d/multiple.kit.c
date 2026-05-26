// Multiple reductions in a loop of depth 1.

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <kitsune.h>

static const double epsilon = 1e-14;

[[clang::noinline]]
static int check(double r, const double *arr, long n) {
  double min = DBL_MAX;
  double max = DBL_MIN;
  for (long i = 0; i < n; ++i) {
    min = fmin(min, arr[i]);
    max = fmax(max, arr[i]);
  }

  return fabs(r - min * max) > epsilon;
}

[[clang::noinline]]
static void init(double *arr, long n) {
  srand(7);
  for (long i = 0; i < n; ++i)
    arr[i] = ((double)rand()) / ((double)RAND_MAX);
}

[[clang::noinline]]
static double test(const double *[[kitsune::mobile]] arr, long n) {
  double min = 10;
  double max = -10;

  // clang-format off
  forall (long i = 0; i < n; ++i) {
    __kitsune_reduce(&min, KIT_MIN, arr[i]);
    __kitsune_reduce(&max, KIT_MAX, arr[i]);
  }
  // clang-format on

  return min * max;
}

int main(int argc, char *argv[]) {
  long n = 1024;
  if (argc > 1)
    n = atol(argv[1]);

  double *[[kitsune::mobile]] arr =
      (double *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(double));

  init((double*)arr, n);
  double result = test(arr, n);
  int err = check(result, (double*)arr, n);

  kitsune_mobile_free(arr);

  if (!err)
    printf("PASS\n");
  else
    printf("FAIL\n");
  return err ? 1 : 0;
}
