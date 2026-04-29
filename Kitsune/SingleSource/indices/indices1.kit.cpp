// Check that a parallel for loop operates over the expected set of indices.

#include "indices.inc"

[[clang::noinline]]
static void test(long n) {
  // clang-format off
  forall (long i = 0; i < n; ++i)
    printf("%ld\n", i);
  // clang-format on

  fflush(stdout);
}

int main(int argc, char *argv[]) {
  long n = 66;
  char *buf = nullptr;

  std::vector<std::vector<int>> expected;
  for (int i = 0; i < n; ++i)
    expected.push_back({i});

  buf = setup(1024);
  test(66);
  bool err = check(buf, expected);
  teardown(buf);

  return report(err);
}
