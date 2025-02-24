// Straightforward memory copy

#include "Kokkos_Core.hpp"
#include <iostream>
#include <kitsune.h>
#include <timing.h>

using ElementType = float;
using namespace kitsune;

#include "copy.inc"

int main(int argc, char *argv[]) {
  size_t n;
  unsigned iterations;
  mobile_ptr<ElementType> dst;
  mobile_ptr<ElementType> src;
  TimerGroup tg("copy");
  Timer &timer = tg.add("copy");
  size_t errors = 0;

  parseCommandLineInto(argc, argv, n, iterations);
  Kokkos::initialize(argc, argv);
  {
    header("kokkos", dst, src, n);

    // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
    // We could try to find a way to make that type Kokkos-friendly, or just
    // wait until the [[kitsune::mobile_ptr]] attribute is implemented which
    // should make Kokkos happy.
    ElementType *[[kitsune::mobile]] bufd = dst.get();
    ElementType *[[kitsune::mobile]] bufs = src.get();

    for (unsigned t = 0; t < iterations; t++) {
      timer.start();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        bufd[i] = bufs[i];
      });
      // clang-format on
      uint64_t us = timer.stop();
      std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
    }

    errors = footer(tg, dst, src, n);
  }
  Kokkos::finalize();

  return errors;
}
