// Simple vector addition benchmark

#include <iostream>
#include <kitsune.h>
#include <timing.h>

using ElementType = float;
using namespace kitsune;

#include "vecadd.inc"

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  mobile_ptr<ElementType> a;
  mobile_ptr<ElementType> b;
  mobile_ptr<ElementType> c;
  TimerGroup tg("vecadd");
  Timer &timer = tg.add("vecadd");

  parseCommandLineInto(argc, argv, n, iterations);
  header("forall", a, b, c, n);

  for (unsigned t = 0; t < iterations; t++) {
    timer.start();
    // clang-format off
    forall(int i = 0; i < n; i++) {
      c[i] = a[i] + b[i];
    }
    // clang-format on
    uint64_t us = timer.stop();
    std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
  }

  size_t errors = footer(tg, a, b, c, n);
  return errors;
}
