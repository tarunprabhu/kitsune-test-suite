#include <cstdlib>
#include <kitsune.h>

using namespace kitsune;

extern "C" {

void vec_add(const mobile_ptr<float> A, const mobile_ptr<float> B,
             mobile_ptr<float> C, uint64_t N) {
  // clang-format off
  forall(size_t i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
  // clang-format on
}

} // extern "C"
