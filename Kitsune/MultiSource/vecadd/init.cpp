#include <cstdlib>
#include <kitsune.h>

using namespace kitsune;

void fill(mobile_ptr<float> data, size_t n) {
  float base = rand() / (float)RAND_MAX;
  // clang-format off
  forall(size_t i = 0; i < n; ++i) {
    data[i] = base + i;
  }
  // clang-format on
}
