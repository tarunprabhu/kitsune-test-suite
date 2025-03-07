// Straightforward memory copy

#include <iostream>
#include <kitsune.h>

#include "timing.h"

#include "copy.inc"

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  kitsune::mobile_ptr<ElementType> dst;
  kitsune::mobile_ptr<ElementType> src;

  TimerGroup tg("copy");
  Timer &total = tg.add("total", "Total");

  parseCommandLineInto(argc, argv, n, iterations);
  header("forall", dst, src, n);

  for (unsigned t = 0; t < iterations; t++) {
    total.start();
    // clang-format off
    forall(size_t i = 0; i < n; i++) {
      dst[i] = src[i];
    }
    uint64_t us = total.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, dst, src, n);
  return errors;
}
