// Simple reduction test with loop of depth 1 (operator: MAX)

#include <limits.h>
#include <math.h>
#include <stdio.h>

#include <kitsune.h>

[[clang::noinline]]
static int check(int r, int expected, int label) {
  if (r != expected) {
    printf("ERROR in test %d. Expected %d. Got %d\n", label, expected, r);
    return label;
  }
  return 0;
}

[[clang::noinline]]
static int test(int *[[kitsune::mobile]] arr, long n, int expected, int init,
                int label) {
  int r = init;

  // clang-format off
  forall (long i = 0; i < n; ++i) {
      __kitsune_reduce(&r, KIT_MAX, arr[i]);
  }
  // clang-format on

  return check(r, expected, label);
}

[[clang::noinline]]
static int testFirst(int *[[kitsune::mobile]] arr, long n) {
  for (long i = 0; i < n; ++i)
    arr[i] = i;
  arr[0] = n + 3;
  return test(arr, n, n + 3, INT_MIN, 1);
}

[[clang::noinline]]
static int testLast(int *[[kitsune::mobile]] arr, long n) {
  for (long i = 0; i < n; ++i)
    arr[i] = i;
  arr[n - 1] = n + 5;
  return test(arr, n, n + 5, INT_MIN, 2);
}

[[clang::noinline]]
static int testMin(int *[[kitsune::mobile]] arr, long n) {
  for (long i = 0; i < n; ++i)
    arr[i] = i;
  arr[n / 2] = INT_MAX;
  return test(arr, n, INT_MAX, INT_MIN, 3);
}

[[clang::noinline]]
static int testMax(int *[[kitsune::mobile]] arr, long n) {
  for (long i = 0; i < n; ++i)
    arr[i] = INT_MIN + 1;
  arr[n / 2] = INT_MIN;
  return test(arr, n, INT_MIN + 1, INT_MIN, 4);
}

[[clang::noinline]]
static int test0(int *[[kitsune::mobile]] arr) {
  arr[0] = 97;
  return test(arr, 0, 97, 97, 5);
}

[[clang::noinline]]
static int test1(int *[[kitsune::mobile]] arr) {
  arr[0] = 211;
  return test(arr, 1, 211, INT_MIN, 6);
}

int main(int argc, char *argv[]) {
  long n = 1024;
  if (argc > 1)
    n = atol(argv[1]);

  int *[[kitsune::mobile]] arr =
      (int *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(int));

  int err = 0;
  err += testFirst(arr, n);
  err += testLast(arr, n);
  err += testMin(arr, n);
  err += testMax(arr, n);
  if (argc == 1) {
    err += test0(arr);
    err += test1(arr);
  }

  kitsune_mobile_free(arr);

  if (!err)
    printf("PASS\n");
  return err ? 1 : 0;
}
