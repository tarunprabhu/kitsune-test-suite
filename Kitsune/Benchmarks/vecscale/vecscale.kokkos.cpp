#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#define __KOKKOS__
#include "vecscale.inc"

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
size_t check(const DualView &vwa, const DualView &vwb, size_t n) {
  const auto &a = vwa.view_host();
  const auto &b = vwb.view_host();

  size_t errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (checkRelErr(b(i), a(i) * 65, epsilon))
      errors++;
  return errors;
}

int main(int argc, char *argv[]) {
  size_t errors = 0;
  Kokkos::initialize(argc, argv);
  {
    size_t n;
    unsigned iterations;
    TimerGroup tg("vecscale");
    Timer &total = tg.add("total", "Total");

    parseCommandLineInto(argc, argv, n, iterations);

    DualView a = DualView("a", n);
    DualView b = DualView("b", n);

    header("kokkos", a, b, n);

    a.modify_host();

    for (unsigned t = 0; t < iterations; t++) {
      total.start();

      a.sync_device();
      b.sync_device();
      const auto &bufa = a.view_device();
      const auto &bufb = b.view_device();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        bufb(i) = bufa(i) * 65;
      });
      // clang-format on
      b.modify_device();
      Kokkos::fence();

      uint64_t us = total.stop();
      std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
    }
    b.sync_host();
    a.sync_host();

    errors = footer(tg, a, b, n);
  }
  Kokkos::finalize();

  return errors;
}
