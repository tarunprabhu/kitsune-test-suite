// saxpy

#include <cmath>
#include <iostream>
#include <kitsune.h>
#include <timing.h>

using namespace kitsune;

const float DEFAULT_X_VALUE = rand() % 1000000;
const float DEFAULT_Y_VALUE = rand() % 1000000;
const float DEFAULT_A_VALUE = rand() % 1000000;

static bool check(const mobile_ptr<float> v, size_t n) {
  float err = 0.0f;
  for (size_t i = 0; i < n; i++) {
    err = err +
          fabs(v[i] - (DEFAULT_A_VALUE * DEFAULT_X_VALUE + DEFAULT_Y_VALUE));
  }
  return err != 0.0f;
}

int main(int argc, char *argv[]) {
  size_t size = 1 << 28;
  unsigned int iterations = 10;
  if (argc > 3) {
    std::cout << "usage: saxpy [size] [iterations]\n";
    return 1;
  }
  if (argc > 2)
    iterations = atoi(argv[2]);
  if (argc > 1)
    size = atol(argv[1]);
  Timer init("init");
  Timer saxpy("saxpy");

  std::cout << "\n";
  std::cout << "---- saxpy benchmark (forall) ----\n"
            << "  Problem size: " << size << " elements.\n\n";
  std::cout << "  Allocating arrays..." << std::flush;
  mobile_ptr<float> x(size);
  mobile_ptr<float> y(size);
  std::cout << "  done.\n\n";

  std::cout << "  Starting benchmark...\n" << std::flush;

  for (unsigned int t = 0; t < iterations; t++) {
    init.start();
    // clang-format off
    forall(size_t i = 0; i < size; i++) {
      x[i] = DEFAULT_X_VALUE;
      y[i] = DEFAULT_Y_VALUE;
    }
    // clang-format on
    uint64_t msInit = init.stop();

    saxpy.start();
    // clang-format off
    forall(size_t i = 0; i < size; i++) {
      y[i] = DEFAULT_A_VALUE * x[i] + y[i];
    }
    // clang-format on
    uint64_t msSaxpy = saxpy.stop();
    std::cout << "\t" << t << ". iteration time: " << (msInit + msSaxpy)
              << " ms\n";
  }

  std::cout << "\n  Checking final result..." << std::flush;
  bool error = check(y, size);
  if (error) {
    std::cout << "  FAIL!\n\n";
  } else {
    std::cout << "  pass\n\n";
  }

  json(std::cout, {init, saxpy});

  x.free();
  y.free();

  return error;
}
