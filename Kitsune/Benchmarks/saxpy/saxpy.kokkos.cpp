#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#define __KOKKOS__
#include "saxpy.inc"

using DualView = Kokkos::DualView<ElementType *, Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace>;

template <> void randomFill<>(DualView &vwa, size_t n) {
  const auto &arr = vwa.view_host();
  for (size_t i = 0; i < n; ++i) {
    arr(i) = rand() / ElementType(RAND_MAX);
  }
}

template <>
size_t check(const DualView &vwx, const DualView &vwy, const DualView &vwr,
             size_t n) {
  const auto &r = vwr.view_host();
  const auto &x = vwx.view_host();
  const auto &y = vwy.view_host();
  size_t errors = 0;
  for (size_t i = 0; i < n; i++)
    if (checkRelErr(A * x[i] + y[i], r[i], epsilon))
      errors++;
  return errors;
}

int main(int argc, char *argv[]) {
  size_t errors = 0;
  Kokkos::initialize(argc, argv);
  {
    size_t n;
    unsigned iterations;
    TimerGroup tg("saxpy");
    Timer &total = tg.add("total", "Total");

    parseCommandLineInto(argc, argv, n, iterations);

    DualView x = DualView("x", n);
    DualView y = DualView("y", n);
    DualView r = DualView("r", n);

    header("kokkos", x, y, r, n);

    x.modify_host();
    y.modify_host();

    for (unsigned t = 0; t < iterations; t++) {
      total.start();

      x.sync_device();
      y.sync_device();
      r.sync_device();
      const auto &bufx = x.view_device();
      const auto &bufy = y.view_device();
      const auto &bufr = r.view_device();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        bufr(i) = A * bufx(i) + bufy(i);
      });
      // clang-format on
      r.modify_device();
      Kokkos::fence();

      uint64_t us = total.stop();
      std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
    }
    r.sync_host();
    y.sync_host();
    x.sync_host();

    errors = footer(tg, x, y, r, n);
  }
  Kokkos::finalize();

  return errors;
}
