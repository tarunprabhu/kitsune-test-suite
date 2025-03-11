#include <kitsune.h>

#include "../../Benchmarks/vecadd/types.h"

void vecadd(kitsune::mobile_ptr<ElementType> c,
            const kitsune::mobile_ptr<ElementType> a,
            const kitsune::mobile_ptr<ElementType> b, size_t n) {
  // clang-format off
  forall(size_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
  // clang-format on
}
