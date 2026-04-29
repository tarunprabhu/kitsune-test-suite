#include <cstdio>
#include <cstdlib>
#include <limits>

#include <kitsune.h>

using ElemT = float;

ElemT A = ElemT(3.14159);

static constexpr ElemT epsilon = 1e-6;
static bool ok(ElemT actual, ElemT expected, ElemT epsilon) {
  return std::abs((expected - actual) / expected) < epsilon;
}

[[clang::noinline]]
static void setup(ElemT *[[kitsune::mobile]] & r,
                  ElemT *[[kitsune::mobile]] & x,
                  ElemT *[[kitsune::mobile]] & y, long n) {
  r = (ElemT *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(ElemT));
  x = (ElemT *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(ElemT));
  y = (ElemT *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(ElemT));

  srand(7);
  for (long i = 0; i < n; ++i) {
    x[i] = rand() / (ElemT)RAND_MAX;
    y[i] = rand() / (ElemT)RAND_MAX;
    r[i] = 0;
  }
}

[[clang::noinline]]
static void teardown(ElemT *[[kitsune::mobile]] r, ElemT *[[kitsune::mobile]] x,
                     ElemT *[[kitsune::mobile]] y) {
  kitsune_mobile_free(r);
  kitsune_mobile_free(x);
  kitsune_mobile_free(y);
}

[[clang::noinline]]
static long check(const ElemT *[[kitsune::mobile]] r,
                  const ElemT *[[kitsune::mobile]] x,
                  const ElemT *[[kitsune::mobile]] y, long n) {
  for (long i = 0; i < n; ++i)
    if (!ok(r[i], A * x[i] + y[i], epsilon)) {
      ElemT a = r[i];
      ElemT e = A * x[i] + y[i];
      ElemT rel = (e - a) / e;
      printf("At index %ld: a=%.6f e=%.6f, rel=%.6f\n", i, a, e, rel);
      return i + 1;
    }
  return 0;
}

[[clang::noinline]]
static int report(long err) {
  if (err)
    printf("ERROR at index %ld\n", err - 1);
  else
    printf("PASS\n");
  return err ? 1 : 0;
}
