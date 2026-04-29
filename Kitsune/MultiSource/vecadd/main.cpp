// Multi-file vector addition benchmark. This is purely to test LTO.

#include <cstdio>
#include <cstdlib>

#include <kitsune.h>

void vecadd(int *[[kitsune::mobile]] c, const int *[[kitsune::mobile]] a,
            const int *[[kitsune::mobile]], long n);

[[clang::noinline]]
static void setup(int *[[kitsune::mobile]] & a, int *[[kitsune::mobile]] & b,
                  int *[[kitsune::mobile]] & c, long n) {
  a = (int *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(int));
  b = (int *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(int));
  c = (int *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(int));

  srand(7);
  for (long i = 0; i < n; ++i) {
    a[i] = rand();
    b[i] = rand();
    c[i] = 0;
  }
}

[[clang::noinline]]
static void teardown(int *[[kitsune::mobile]] a, int *[[kitsune::mobile]] b,
                     int *[[kitsune::mobile]] c) {
  kitsune_mobile_free(a);
  kitsune_mobile_free(b);
  kitsune_mobile_free(c);
}

[[clang::noinline]]
static long check(const int *[[kitsune::mobile]] c,
                  const int *[[kitsune::mobile]] a,
                  const int *[[kitsune::mobile]] b, long n) {
  for (long i = 0; i < n; ++i)
    if (c[i] != a[i] + b[i])
      return i + 1;
  return 0;
}

int main(int argc, char *argv[]) {
  long err;
  int *[[kitsune::mobile]] a = NULL;
  int *[[kitsune::mobile]] b = NULL;
  int *[[kitsune::mobile]] c = NULL;
  long n = 2048;
  if (argc > 1)
    n = atol(argv[1]);

  setup(a, b, c, n);
  vecadd(c, a, b, n);
  err = check(c, a, b, n);
  teardown(a, b, c);

  if (err)
    printf("ERROR at index %ld\n", err - 1);
  else
    printf("PASS\n");
  return err ? 1 : 0;
}
