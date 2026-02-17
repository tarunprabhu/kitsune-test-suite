// Simple saxpy benchmark

#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#include "saxpy.inc"

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  ElementType* x;
  ElementType* y;
  ElementType* r;

  TimerGroup tg("saxpy");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations);
  header("for", x, y, r, n);

  for (unsigned t = 0; t < iterations; t++) {
    total.start();
    // clang-format off
    for (size_t i = 0; i < n; i++) {
      r[i] = A * x[i] + y[i];
    }
    // clang-format on
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  bool hasErrors = footer(tg, x, y, r, n);
  return hasErrors;
}
