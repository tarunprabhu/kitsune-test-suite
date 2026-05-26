// Simple reduction test with loop of depth 1 (operator: logical XOR)

#include <limits.h>
#include <math.h>
#include <stdio.h>

#include <kitsune.h>

[[clang::noinline]]
static int check(bool r, bool expected, int label) {
  if (r != expected) {
    printf("ERROR in test %d. Expected %d. Got %d\n", label, expected, r);
    return label;
  }
  return 0;
}

[[clang::noinline]]
static int test(bool *[[kitsune::mobile]] arr, long n, bool expected, bool init,
                int label) {
  bool r = init;

  // clang-format off
  forall (long i = 0; i < n; ++i) {
    __kitsune_reduce(&r, KIT_LXOR, arr[i]);
  }
  // clang-format on

  return check(r, expected, label);
}

[[clang::noinline]]
static int testTrue(bool *[[kitsune::mobile]] arr, long n) {
  for (long i = 0; i < n; ++i)
    arr[i] = true;
  return test(arr, n, false, false, 1);
}

[[clang::noinline]]
static int testFalse(bool *[[kitsune::mobile]] arr, long n) {
  for (long i = 0; i < n; ++i)
    arr[i] = false;
  return test(arr, n, false, false, 2);
}

[[clang::noinline]]
static int testSome(bool *[[kitsune::mobile]] arr, long n) {
  for (long i = 0; i < n; ++i)
    arr[i] = false;
  arr[n / 2] = true;
  return test(arr, n, true, false, 3);
}

int main(int argc, char *argv[]) {
  long n = 1024;
  if (argc > 1)
    n = atol(argv[1]);

  bool *[[kitsune::mobile]] arr =
      (bool *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(bool));

  int err = 0;
  err += testTrue(arr, n);
  err += testFalse(arr, n);
  err += testSome(arr, n);

  kitsune_mobile_free(arr);

  if (!err)
    printf("PASS\n");
  return err ? 1 : 0;
}
