// Straightforward memory copy

#include <Kokkos_Core.hpp>
#include <iostream>
#include <kitsune.h>

#include "timing.h"

#include "copy.inc"

int main(int argc, char *argv[]) {
  size_t errors = 0;
  Kokkos::initialize(argc, argv);
  {
    size_t n;
    unsigned iterations;
    kitsune::mobile_ptr<ElementType> dst;
    kitsune::mobile_ptr<ElementType> src;

    TimerGroup tg("copy");
    Timer &total = tg.add("total", "Total");

    parseCommandLineInto(argc, argv, n, iterations);
    header("kokkos", dst, src, n);

    // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
    // We could try to find a way to make that type Kokkos-friendly, or just
    // wait until the [[kitsune::mobile_ptr]] attribute is implemented which
    // should make Kokkos happy.
    ElementType *[[kitsune::mobile]] bufd = dst.get();
    ElementType *[[kitsune::mobile]] bufs = src.get();

    for (unsigned t = 0; t < iterations; t++) {
      total.start();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        bufd[i] = bufs[i];
      });
      // clang-format on
      uint64_t us = total.stop();
      std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
    }

    errors = footer(tg, dst, src, n);
  }
  Kokkos::finalize();

  return errors;
}
