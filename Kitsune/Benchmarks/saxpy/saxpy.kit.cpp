// Simple saxpy benchmark

#include <iostream>
#include <kitsune.h>

#include "fpcmp.h"
#include "timing.h"

#include "saxpy.inc"

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  kitsune::mobile_ptr<ElementType> x;
  kitsune::mobile_ptr<ElementType> y;
  kitsune::mobile_ptr<ElementType> r;

  TimerGroup tg("saxpy");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations);
  header("forall", x, y, r, n);

  for (unsigned t = 0; t < iterations; t++) {
    total.start();
    // clang-format off
    forall(size_t i = 0; i < n; i++) {
      r[i] = A * x[i] + y[i];
    }
    // clang-format on
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, x, y, r, n);
  return errors;
}
