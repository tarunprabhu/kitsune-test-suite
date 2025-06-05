// Simple vector scale benchmark

#include <iostream>
#include <kitsune.h>

#include "fpcmp.h"
#include "timing.h"

#include "vecscale.inc"

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  kitsune::mobile_ptr<ElementType> a;
  kitsune::mobile_ptr<ElementType> b;

  TimerGroup tg("vecscale");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations);
  header("forall", a, b, n);

  for (unsigned t = 0; t < iterations; t++) {
    total.start();
    // clang-format off
    forall(int i = 0; i < n; i++) {
      b[i] = a[i] * 65;
    }
    // clang-format on
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, a, b, n);
  return errors;
}
