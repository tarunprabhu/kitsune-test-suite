// Simple reduction test with loop of depth 1 (operator: +)

#include <stdio.h>

#include <kitsune.h>

[[clang::noinline]]
static int check(long v, long n, int label) {
  if (v != n * (n + 1) / 2) {
    printf("ERROR in test %d\n", label);
    return label;
  }
  return 0;
}

[[clang::noinline]]
static int test(long n, int label) {
  long r = 0;

  // clang-format off
  forall (long i = 0; i < n; ++i) {
    __kitsune_reduce(&r, KIT_SUM, i + 1);
  }
  // clang-format on

  return check(r, n, label);
}

int main(int argc, char *argv[]) {
  long n = 4096;
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
