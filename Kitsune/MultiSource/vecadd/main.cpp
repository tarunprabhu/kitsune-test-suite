// Multi-file vector addition benchmark. This is purely to test LTO.

#include <iostream>
#include <kitsune.h>

#include "fpcmp.h"
#include "timing.h"

#include "../../Benchmarks/vecadd/vecadd.inc"

void vecadd(kitsune::mobile_ptr<ElementType> c,
            const kitsune::mobile_ptr<ElementType> a,
            const kitsune::mobile_ptr<ElementType> b, size_t n);

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  kitsune::mobile_ptr<ElementType> a;
  kitsune::mobile_ptr<ElementType> b;
  kitsune::mobile_ptr<ElementType> c;

  TimerGroup tg("vecadd");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations);
  header("forall", a, b, c, n);

  for (unsigned t = 0; t < iterations; t++) {
    total.start();
    vecadd(c, a, b, n);
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, a, b, c, n);
  return errors;
}
