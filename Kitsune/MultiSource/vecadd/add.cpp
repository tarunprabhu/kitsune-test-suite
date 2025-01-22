#include <cstdlib>
#include <kitsune.h>

using namespace kitsune;

void vec_add(const mobile_ptr<float> a, const mobile_ptr<float> b,
             mobile_ptr<float> c, size_t n) {
  // clang-format off
  forall(size_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
  // clang-format on
}
