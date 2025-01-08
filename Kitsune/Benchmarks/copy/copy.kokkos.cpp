// Straightforward memory copy

#include "Kokkos_Core.hpp"

#include <iostream>
#include <kitsune.h>
#include <timing.h>

using namespace kitsune;

template <typename T> static void random_fill(mobile_ptr<T> arr, size_t n) {
  for (size_t i = 0; i < n; ++i)
    arr[i] = rand() / (T)RAND_MAX;
}

template <typename T>
static size_t check(const mobile_ptr<T> src, const mobile_ptr<T> dst,
                    size_t n) {
  size_t errors;
  for (size_t i = 0; i < n; ++i) {
    if (src[i] != dst[i])
      errors += 1;
  }
  return errors;
}

int main(int argc, char **argv) {
  size_t size = 1024 * 1024 * 256;
  unsigned int iterations = 10;
  if (argc > 1)
    size = atol(argv[1]);
  if (argc > 2)
    iterations = atoi(argv[2]);
  Timer timer("copy");
  int retcode = 0;

  std::cout << "\n";
  std::cout << "---- Simple copy benchmark (forall) ----\n"
            << "  Array size: " << size << "\n"
            << "  Iterations: " << iterations << "\n\n";
  std::cout << "Allocating arrays and filling with random values..."
            << std::flush;
  Kokkos::initialize(argc, argv);
  {
    mobile_ptr<float> src(size);
    mobile_ptr<float> dst(size);
    random_fill(src, size);
    std::cout << "  done.\n\n";

    // FIXME: Simply using dst and src in the Kokkos lambda does not compile.
    // The Kokkos are, unsurprisingly, utterly useless. For now, this works,
    // so we will leave it like this. This should eventually be changed to use
    // attributed pointers instead, which should work like the regular
    // pointers and, hopefully, keep Kokkos happy.
    float* bufs = src.get();
    float* bufd = dst.get();

    std::cout << "Starting benchmark...\n";

    for (int t = 0; t < iterations; t++) {
      timer.start();
      // clang-format off
      Kokkos::parallel_for(size, KOKKOS_LAMBDA(const int i) {
        bufd[i] = bufs[i];
      });
      // clang-format on
      uint64_t us = timer.stop();
      std::cout << "\t" << t << ". iteration time: " << us << " us\n";
    }

    std::cout << "\n  Checking final result..." << std::flush;
    size_t errors = check(src, dst, size);
    if (errors)
      std::cout << "  FAIL! (" << errors << " errors found)\n\n";
    else
      std::cout << "  pass\n\n";

    json(std::cout, {timer});

    src.free();
    dst.free();

    retcode = errors;
  }
  Kokkos::finalize();

  return retcode;
}
