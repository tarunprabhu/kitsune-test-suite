// Check that parallel for loops in a nest of depth 3 operate over the expected
// set of indices.

#include "indices.inc"

[[clang::noinline]]
static void test(long m, long n, long p) {
  // clang-format off
  forall (long i = 0; i < m; ++i)
    forall (long j = 0; j < n; ++j)
      forall (long k = 0; k < p; ++k)
        printf("%ld %ld %ld\n", i, j, k);
  // clang-format on

  fflush(stdout);
}

int main(int argc, char *argv[]) {
  long m = 9;
  long n = 11;
  long p = 10;
  char *buf = nullptr;

  std::vector<std::vector<int>> expected;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < p; ++k)
        expected.push_back({i, j, k});

  buf = setup(8192);
  test(9, 11, 10);
  bool err = check(buf, expected);
  teardown(buf);

  return report(err);
}
