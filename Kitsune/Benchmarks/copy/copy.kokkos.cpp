#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#define __KOKKOS__
#include "copy.inc"

using DualView = Kokkos::DualView<ElementType *, Kokkos::LayoutRight,
                                  Kokkos::DefaultExecutionSpace>;

template <> void randomFill<>(DualView &vwa, size_t n) {
  const auto &arr = vwa.view_host();
  for (size_t i = 0; i < n; ++i) {
    arr(i) = rand() / ElementType(RAND_MAX);
  }
}

template <> size_t check(const DualView &vwd, const DualView &vws, size_t n) {
  const auto &dst = vwd.view_host();
  const auto &src = vws.view_host();
  size_t errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (dst(i) != src(i))
      errors++;
  return errors;
}

int main(int argc, char *argv[]) {
  size_t errors = 0;
  Kokkos::initialize(argc, argv);
  {
    size_t n;
    unsigned iterations;
    TimerGroup tg("copy");
    Timer &total = tg.add("total", "Total");

    parseCommandLineInto(argc, argv, n, iterations);

    DualView dst = DualView("dst", n);
    DualView src = DualView("src", n);

    header("kokkos", dst, src, n);

    src.modify_host();

    for (unsigned t = 0; t < iterations; t++) {
      total.start();

      dst.sync_device();
      src.sync_device();
      const auto &bufs = src.view_device();
      const auto &bufd = dst.view_device();
      // clang-format off
      Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) {
        bufd(i) = bufs(i);
      });
      // clang-format on
      dst.modify_device();
      Kokkos::fence();

      uint64_t us = total.stop();
      std::cout << "\t" << t << ". iteration time: " << Timer::secs(us) << "\n";
    }
    dst.sync_host();
    src.sync_host();

    errors = footer(tg, dst, src, n);
  }
  Kokkos::finalize();

  return errors;
}
