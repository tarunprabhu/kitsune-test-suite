// Test that passing structs to, and returning structs from, device functions
// on GPUs works as expected

#include <cmath>
#include <cstring>
#include <iostream>
#include <kitsune.h>

using namespace kitsune;

struct Vec {
  float x, y;
};

static void random_fill(mobile_ptr<Vec> data, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    data[i].x = rand() / (float)RAND_MAX;
    data[i].y = rand() / (float)RAND_MAX;
  }
}

static Vec vec_sum(const Vec &a, const Vec &b) {
  Vec sum;
  sum.x = a.x + b.x;
  sum.y = a.y + b.y;
  return sum;
}

static size_t check(const mobile_ptr<Vec> dst, const mobile_ptr<Vec> src,
                    const mobile_ptr<Vec> cpy, size_t n) {
  size_t errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (dst[i].x != src[i].x + cpy[i].x || dst[i].y != src[i].y + cpy[i].y)
      errors += 1;
  return errors;
}

int main(int argc, char **argv) {
  size_t size = 1024 * 1024;
  if (argc > 1)
    size = atol(argv[1]);

  mobile_ptr<Vec> dst(size);
  mobile_ptr<Vec> src(size);
  mobile_ptr<Vec> cpy(size);

  random_fill(dst, size);
  random_fill(src, size);
  std::memcpy((Vec*)cpy.get(), (Vec*)dst.get(), size * sizeof(Vec));

  // clang-format off
  forall(size_t i = 0; i < size; i++) {
    Vec sum = vec_sum(dst[i], src[i]);
    dst[i].x = sum.x;
    dst[i].y = sum.y;
  }
  // clang-format on

  std::cout << "\n  Checking final result..." << std::flush;
  size_t errors = check(dst, src, cpy, size);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  dst.free();
  src.free();
  cpy.free();

  return errors;
}
