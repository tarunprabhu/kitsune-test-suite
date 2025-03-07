// Simple vector addition benchmark

#include <Kokkos_Core.hpp>
#include <iostream>
#include <kitsune.h>

#include "fpcmp.h"
#include "timing.h"

#include "vecadd.inc"

int main(int argc, char *argv[]) {
  size_t errors = 0;
  Kokkos::initialize(argc, argv);
  {
    size_t n;
    unsigned iterations;
    kitsune::mobile_ptr<ElementType> a;
    kitsune::mobile_ptr<ElementType> b;
    kitsune::mobile_ptr<ElementType> c;

    TimerGroup tg("vecadd");
    Timer &total = tg.add("total", "Total");

    parseCommandLineInto(argc, argv, n, iterations);
    header("kokkos", a, b, c, n);

    // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
    // We could try to find a way to make that type Kokkos-friendly, or just
    // wait until the [[kitsune::mobile_ptr]] attribute is implemented which
    // should make Kokkos happy.
    ElementType *[[kitsune::mobile]] bufa = a.get();
    ElementType *[[kitsune::mobile]] bufb = b.get();
    ElementType *[[kitsune::mobile]] bufc = c.get();

    for (unsigned t = 0; t < iterations; t++) {
      total.start();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        bufc[i] = bufa[i] + bufb[i];
      });
      // clang-format on
      uint64_t us = total.stop();
      std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
    }

    errors = footer(tg, a, b, c, n);
  }
  Kokkos::finalize();

  return errors;
}
