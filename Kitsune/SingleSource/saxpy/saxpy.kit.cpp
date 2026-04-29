// Simple saxpy test. This mostly tests that global variables are handled
// correctly. We could probably do something better than this for globals
// though

#include "saxpy.inc"

[[clang::noinline]]
static void test(ElemT *[[kitsune::mobile]] r,
                 const ElemT *[[kitsune::mobile]] x,
                 const ElemT *[[kitsune::mobile]] y, long n) {
  // clang-format off
  forall (long i = 0; i < n; ++i)
    r[i] = A * x[i] + y[i];
  // clang-format on
}

int main(int argc, char *argv[]) {
  long err;
  ElemT *[[kitsune::mobile]] r = nullptr;
  ElemT *[[kitsune::mobile]] x = nullptr;
  ElemT *[[kitsune::mobile]] y = nullptr;
  long n = 2048;
  if (argc > 1)
    n = atol(argv[1]);

  setup(r, x, y, n);
  test(r, x, y, n);
  err = check(r, x, y, n);
  teardown(r, x, y);

  return report(err);
}
