// Simple saxpy benchmark

#include <Kokkos_Core.hpp>
#include <iostream>
#include <kitsune.h>

#include "fpcmp.h"
#include "timing.h"

#include "saxpy.inc"

int main(int argc, char *argv[]) {
  size_t errors = 0;
  Kokkos::initialize(argc, argv);
  {
    size_t n;
    unsigned iterations;
    kitsune::mobile_ptr<ElementType> x;
    kitsune::mobile_ptr<ElementType> y;
    kitsune::mobile_ptr<ElementType> r;

    TimerGroup tg("saxpy");
    Timer &total = tg.add("total", "Total");

    parseCommandLineInto(argc, argv, n, iterations);
    header("kokkos", x, y, r, n);

    // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
    // We could try to find a way to make that type Kokkos-friendly, or just
    // wait until the [[kitsune::mobile_ptr]] attribute is implemented which
    // should make Kokkos happy.
    ElementType *[[kitsune::mobile]] x_p = x.get();
    ElementType *[[kitsune::mobile]] y_p = y.get();
    ElementType *[[kitsune::mobile]] r_p = r.get();

    for (unsigned t = 0; t < iterations; t++) {
      total.start();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        r_p[i] = A * x_p[i] + y_p[i];
      });
      // clang-format on
      uint64_t us = total.stop();
      std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
    }

    errors = footer(tg, x, y, r, n);
  }
  Kokkos::finalize();

  return errors;
}
