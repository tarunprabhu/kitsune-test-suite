// Test that calls to math functions in forall loops work correctly. This is
// only really relevant for the GPU-centric tapir targets.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <kitsune.h>

static float math_call1(float);
static float math_call2(float);

struct testit {
  float a, b;
};

[[clang::noinline]]
static void setup(kitsune::mobile_ptr<float> &dst,
                  kitsune::mobile_ptr<float> &src, long n) {
  dst.alloc(n);
  src.alloc(n);

  for (long i = 0; i < n; ++i) {
    src[i] = rand() / float(RAND_MAX);
    dst[i] = 0;
  }
}

[[clang::noinline]]
static void teardown(kitsune::mobile_ptr<float> &dst,
                     kitsune::mobile_ptr<float> &src) {
  dst.free();
  src.free();
}

static constexpr float epsilon = 1e-6;

static float relErr(float actual, float expected) {
  return std::abs((expected - actual) / expected);
}

[[clang::noinline]]
static long check(const kitsune::mobile_ptr<float> &dst,
                  const kitsune::mobile_ptr<float> &src, long n) {
  for (long i = 0; i < n; ++i)
    if (relErr(dst[i], math_call1(src[i]) + math_call2(src[i])) > epsilon)
      return i + 1;
  return 0;
}

static float math_call1(float value) { return fminf(value, 0.1234); }

static float math_call2(float value) { return sqrtf(value); }

[[clang::noinline]]
static void test(kitsune::mobile_ptr<float> &dst,
                 const kitsune::mobile_ptr<float> &src, long n) {
  // clang-format off
  forall(long i = 0; i < n; ++i) {
    dst[i] = math_call1(src[i]) + math_call2(src[i]);
  }
  // clang-format on
}

int main(int argc, char **argv) {
  kitsune::mobile_ptr<float> dst;
  kitsune::mobile_ptr<float> src;
  long n = 2048;
  if (argc > 1)
    n = atol(argv[1]);

  setup(dst, src, n);
  test(dst, src, n);
  long err = check(dst, src, n);
  teardown(dst, src);

  if (err)
    printf("FAIL: Error at index %ld\n", err - 1);
  else
    printf("PASS\n");
  return err ? 1 : 0;
}
