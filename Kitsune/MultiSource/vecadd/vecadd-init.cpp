#include <cstdlib>
#include <kitsune.h>

using namespace kitsune;

extern "C" {

void fill(mobile_ptr<float> data, uint64_t N) {
  float base_value = rand() / (float)RAND_MAX;
  // clang-format off
  forall(size_t i = 0; i < N; ++i) {
    data[i] = base_value + i;
  }
  // clang-format on
}

} // extern "C"
