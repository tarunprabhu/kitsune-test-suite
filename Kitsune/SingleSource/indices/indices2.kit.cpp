// Check that parallel for loops in a nest of depth 2 operate over the expected
// set of indices.

#include "indices.inc"

[[clang::noinline]]
static void test(long m, long n) {
  // clang-format off
  forall (long i = 0; i < m; ++i)
    forall (long j = 0; j < n; ++j)
      printf("%ld %ld\n", i, j);
  // clang-format on

  fflush(stdout);
}

int main(int argc, char *argv[]) {
  long m = 34;
  long n = 33;
  char *buf = nullptr;

  std::vector<std::vector<int>> expected;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      expected.push_back({i, j});

  buf = setup(8192);
  test(34, 33);
  bool err = check(buf, expected);
  teardown(buf);

  return report(err);
}
