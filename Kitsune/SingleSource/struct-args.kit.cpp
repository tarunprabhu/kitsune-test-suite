// Test that passing structs to, and returning structs from, device functions
// on GPUs works as expected

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <kitsune.h>

struct Vec {
  float x, y;
};

[[clang::noinline]]
static void setup(Vec *[[kitsune::mobile]] & a, Vec *[[kitsune::mobile]] & b,
                  Vec *[[kitsune::mobile]] & c, long n) {
  a = (Vec *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(Vec));
  b = (Vec *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(Vec));
  c = (Vec *[[kitsune::mobile]])kitsune_mobile_alloc(n * sizeof(Vec));

  for (size_t i = 0; i < n; ++i) {
    a[i].x = rand() / float(RAND_MAX);
    a[i].y = rand() / float(RAND_MAX);
    b[i].x = rand() / float(RAND_MAX);
    b[i].y = rand() / float(RAND_MAX);
    c[i].x = 0;
    c[i].y = 0;
  }
}

[[clang::noinline]]
static void teardown(Vec *[[kitsune::mobile]] & a, Vec *[[kitsune::mobile]] & b,
                     Vec *[[kitsune::mobile]] & c) {
  kitsune_mobile_free(a);
  kitsune_mobile_free(b);
  kitsune_mobile_free(c);
}

[[clang::noinline]]
static long check(const Vec *[[kitsune::mobile]] c,
                  const Vec *[[kitsune::mobile]] a,
                  const Vec *[[kitsune::mobile]] b, long n) {
  for (long i = 0; i < n; ++i)
    if (c[i].x != a[i].x + b[i].x || c[i].y != a[i].y + b[i].y)
      return i + 1;
  return 0;
}

static Vec sum(const Vec &a, const Vec &b) {
  Vec sum;
  sum.x = a.x + b.x;
  sum.y = a.y + b.y;
  return sum;
}

static void test(Vec *c, const Vec *a, const Vec *b, long n) {
  // clang-format off
  forall(long i = 0; i < n; i++) {
    c[i] = sum(a[i], b[i]);
  }
  // clang-format on
}

int main(int argc, char **argv) {
  Vec *[[kitsune::mobile]] a = nullptr;
  Vec *[[kitsune::mobile]] b = nullptr;
  Vec *[[kitsune::mobile]] c = nullptr;
  size_t n = 2048;
  if (argc > 1)
    n = atol(argv[1]);

  setup(a, b, c, n);
  test((Vec*)c, (Vec*)a, (Vec*)b, n);
  long err = check(c, a, b, n);
  teardown(a, b, c);

  if (err)
    printf("FAIL: Error at index %ld\n", err - 1);
  else
    printf("PASS\n");
  return err ? 1 : 0;
}
