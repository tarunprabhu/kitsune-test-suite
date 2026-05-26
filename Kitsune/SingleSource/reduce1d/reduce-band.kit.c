// Simple reduction test with loop of depth 1 (operator: &)

#include <limits.h>
#include <stdio.h>

#include <kitsune.h>

[[clang::noinline]]
static int check(uint64_t v, uint64_t n, int label) {
  uint64_t expected = n == 0 ? UINT64_MAX : 1;
  if (v != expected) {
    printf("ERROR in test %d\n", label);
    return label;
  }
  return 0;
}

[[clang::noinline]]
static int test(uint64_t n, int label) {
  uint64_t r = UINT64_MAX;

  // clang-format off
  forall (uint64_t i = 0; i < n; ++i) {
    __kitsune_reduce(&r, KIT_BAND, 2 * i + 1);
  }
  // clang-format on

  return check(r, n, label);
}

int main(int argc, char *argv[]) {
  uint64_t n = 1024;
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
