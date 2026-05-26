// Simple reduction test with loop of depth 1 (operator: *)

#include <math.h>
#include <stdio.h>

#include <kitsune.h>

static const double epsilon = 1e-14;

[[clang::noinline]]
static int check(double v, long n, int label) {
  double e = 1;
  for (long i = 0; i < n; ++i)
    e *= sin(i + 1);

  if (fabs(e - v) > epsilon) {
    printf("ERROR in test %d\n", label);
    return label;
  }
  return 0;
}

[[clang::noinline]]
static int test(long n, int label) {
  double r = 1;

  // clang-format off
  forall (long i = 0; i < n; ++i) {
    __kitsune_reduce(&r, KIT_PROD, sin(i + 1));
  }
  // clang-format on

  return check(r, n, label);
}

int main(int argc, char *argv[]) {
  long n = 1024;
  if (argc > 1)
    n = atol(argv[1]);

  int err = 0;
  err += test(n, 1);
  if (argc == 1) {
    err += test(0, 2);
    err += test(1, 3);
    err += test(7, 4);
    err += test(8, 5);
    err += test(9, 6);
    err += test(256, 7);
    err += test(513, 8);
  }

  if (!err)
    printf("PASS\n");
  return err ? 1 : 0;
}
