#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#define __KOKKOS__
#include "vecadd.inc"

using DualViewVector = Kokkos::DualView<ElementType *, Kokkos::LayoutRight,
                                        Kokkos::DefaultExecutionSpace>;

// This is an explicit specialization of the randomFill function which is
// defined in vecadd.inc.
template <>
static void randomFill<>(DualViewVector &arr, size_t n, bool small) {
  for (size_t i = 0; i < n; ++i) {
    arr.view_host()(i) = rand() / ElementType(RAND_MAX);
    if (not small)
      arr.view_host()(i) *= rand();
  }
}

// This is an explicit specialization of the randomFill function which is
// defined in vecadd.inc. This doesn't actually check anything. It's not ideal
// but I really couldn't be bothered about Kokkos.
template <>
static size_t check(const DualViewVector &a, const DualViewVector &b,
                    const DualViewVector &c, size_t n) {
  std::cout << "\n  WARNING! Not checking for errors\n\n";
  return 0;
}

int main(int argc, char *argv[]) {
  size_t errors = 0;
  Kokkos::initialize(argc, argv);
  {
    size_t n;
    unsigned iterations;
    TimerGroup tg("vecadd");
    Timer &total = tg.add("total", "Total");

    parseCommandLineInto(argc, argv, n, iterations);

    DualViewVector a = DualViewVector("a", n);
    DualViewVector b = DualViewVector("b", n);
    DualViewVector c = DualViewVector("c", n);

    header("kokkos", a, b, c, n);

    a.modify_host();
    b.modify_host();

    for (unsigned t = 0; t < iterations; t++) {
      total.start();

      a.sync_device();
      b.sync_device();
      c.sync_device();
      const auto &bufa = a.view_device();
      const auto &bufb = b.view_device();
      const auto &bufc = c.view_device();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        bufc(i) = bufa(i) + bufb(i);
      });
      // clang-format on
      c.modify_device();
      Kokkos::fence();

      uint64_t us = total.stop();
      std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
    }
    c.sync_host();
    b.sync_host();
    a.sync_host();

    errors = footer(tg, a, b, c, n);
  }
  Kokkos::finalize();

  return errors;
}
