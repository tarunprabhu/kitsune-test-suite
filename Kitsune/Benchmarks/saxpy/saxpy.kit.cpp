// Simple saxpy benchmark

#include <cmath>
#include <iostream>
#include <kitsune.h>
#include <timing.h>

using ElementType = float;
using namespace kitsune;

#include "saxpy.inc"

const ElementType DEFAULT_X_VALUE = rand() % 1000000;
const ElementType DEFAULT_Y_VALUE = rand() % 1000000;
const ElementType DEFAULT_A_VALUE = rand() % 1000000;

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  mobile_ptr<ElementType> x;
  mobile_ptr<ElementType> y;
  mobile_ptr<ElementType> r;
  TimerGroup tg("saxpy");
  Timer &init = tg.add("init");
  Timer &saxpy = tg.add("saxpy");

  parseCommandLineInto(argc, argv, n, iterations);
  header("forall", DEFAULT_A_VALUE, x, y, r, n);

  for (unsigned t = 0; t < iterations; t++) {
    init.start();
    // clang-format off
    forall(size_t i = 0; i < n; i++) {
      x[i] = DEFAULT_X_VALUE;
      y[i] = DEFAULT_Y_VALUE;
    }
    // clang-format on
    uint64_t usInit = init.stop();

    saxpy.start();
    // clang-format off
    forall(size_t i = 0; i < n; i++) {
      y[i] = DEFAULT_A_VALUE * x[i] + y[i];
    }
    // clang-format on
    uint64_t usSaxpy = saxpy.stop();
    std::cout << "\t" << t
              << ". iteration time: " << Timer::secs(usInit + usSaxpy) << "\n";
  }

  size_t errors = footer(tg, DEFAULT_A_VALUE, x, y, r, n);
  return errors;
}
