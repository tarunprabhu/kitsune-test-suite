// Simple saxpy benchmark

#include "Kokkos_Core.hpp"
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
  size_t errors = 0;

  parseCommandLineInto(argc, argv, n, iterations);
  Kokkos::initialize(argc, argv);
  {
    header("kokkos", DEFAULT_A_VALUE, x, y, r, n);

    // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons.
    // We could try to find a way to make that type Kokkos-friendly, or just
    // wait until the [[kitsune::mobile_ptr]] attribute is implemented which
    // should make Kokkos happy.
    ElementType *[[kitsune::mobile]] bufx = x.get();
    ElementType *[[kitsune::mobile]] bufy = y.get();
    ElementType *[[kitsune::mobile]] bufr = r.get();

    init.start();
    // clang-format off
    // This has been disabled because there is a bug in Kitsune that triggers an
    // ICE (Internal Compiler Error) if there are two Kokkos::parallel_for's in
    // the same function. When that is fixed, this can be re-enabled - although
    // it is probably best to just leave this as a for loop and make the same
    // change in the other implementations. The more interesting timing here
    // is that of the actual saxpy loop.
    for (size_t i = 0; i < n; ++i) {
      bufx[i] = DEFAULT_X_VALUE;
      bufy[i] = DEFAULT_Y_VALUE;
    }
    // Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
    //    bufx[i] = DEFAULT_X_VALUE;
    //    bufy[i] = DEFAULT_Y_VALUE;
    // });
    // clang-format on
    uint64_t usInit = init.stop();

    for (unsigned t = 0; t < iterations; t++) {
      saxpy.start();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        bufr[i] = DEFAULT_A_VALUE * bufx[i] + bufy[i];
      });
      // clang-format on
      uint64_t usSaxpy = saxpy.stop();
      std::cout << "\t" << t
                << ". iteration time: " << Timer::secs(usInit + usSaxpy) << "\n";
    }

    errors = footer(tg, DEFAULT_A_VALUE, x, y, r, n);
  }
  Kokkos::finalize();

  return errors;
}
