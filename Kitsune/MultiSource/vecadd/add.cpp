#include <kitsune.h>

#include "../../Benchmarks/vecadd/types.h"

[[kitsune::device]]
ElementType add(ElementType a, ElementType b) {
  return a + b;
}
