// Check that reading from and writing to global arrays works as expected. This
// mainly checks the handling of global arrays in the GPU-centric tapir targets.

#include <cstdio>

#include <kitsune.h>

static constexpr long N = 2048;

static long gdst[N];
static const long gsrc[] = {0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};

[[clang::noinline]]
static long check(long n) {
  for (long i = 0; i < n; ++i)
    if (gdst[i] != gsrc[i % 16])
      return i + 1;
  return 0;
}

[[clang::noinline]]
void test(long n) {
  // clang-format off
  forall (long i = 0; i < n; ++i)
    gdst[i] = gsrc[i % 16];
  // clang-format on
}

int main(int argc, char *argv[]) {
  test(N);
  long err = check(N);

  if (err)
    printf("FAIL: Error at index %ld\n", err - 1);
  else
    printf("PASS\n");
  return err ? 1 : 0;
}
