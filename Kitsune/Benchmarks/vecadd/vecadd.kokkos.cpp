#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#define __KOKKOS__
#include "vecadd.inc"

using DualView = Kokkos::DualView<ElementType *, Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace>;

template <> void randomFill<>(DualView &vwa, size_t n, bool small) {
  const auto &arr = vwa.view_host();
  for (size_t i = 0; i < n; ++i) {
    arr(i) = rand() / ElementType(RAND_MAX);
    if (not small)
      arr(i) *= rand();
  }
}

template <>
size_t check(const DualView &vwa, const DualView &vwb, const DualView &vwc,
             size_t n) {
  const auto &a = vwa.view_host();
  const auto &b = vwb.view_host();
  const auto &c = vwc.view_host();

  size_t errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (checkRelErr(c(i), a(i) + b(i), epsilon))
      errors++;
  return errors;
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

    DualView a = DualView("a", n);
    DualView b = DualView("b", n);
    DualView c = DualView("c", n);

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
