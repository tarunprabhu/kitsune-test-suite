// Test that calls to math functions in forall loops work correctly. This is
// only really relevant for GPU tapir targets

#include <cmath>
#include <iostream>
#include <kitsune.h>

using namespace kitsune;

struct testit {
  float a, b;
};

template <typename T> static void random_fill(mobile_ptr<T> data, size_t n) {
  for (size_t i = 0; i < n; ++i)
    data[i] = (2.0 * 3.142) * (rand() / (T)RAND_MAX);
}

__attribute__((always_inline)) void struct_test(testit *ti) {
  ti->a = 4;
  ti->b = ti->a + 4;
}

template <typename T> __attribute__((always_inline)) T math_call1(T value) {
  testit t;
  struct_test(&t);
  return fminf(value, 1234.56 - t.a + t.b);
}

template <typename T> __attribute__((always_inline)) T math_call2(T value) {
  return sqrtf(value);
}

static size_t check(const mobile_ptr<float> dst, const mobile_ptr<float> src,
                    size_t n) {
  float epsilon = 1e-12;
  size_t errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (fabs(dst[i] - fminf(math_call1(src[i]) + math_call2(src[i]), 100.0)) >
        epsilon)
      errors += 1;
  return errors;
}

int main(int argc, char **argv) {
  size_t size = 1024 * 1024;
  if (argc > 1)
    size = atol(argv[1]);

  mobile_ptr<float> dst(size);
  mobile_ptr<float> src(size);

  random_fill(dst, size);

  // clang-format off
  forall(size_t i = 0; i < size; ++i) {
    dst[i] = fminf(math_call1(src[i]) + math_call2(src[i]), 100.0);
  }
  // clang-format on

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(dst, src, size);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  dst.free();
  src.free();

  return errors;
}
