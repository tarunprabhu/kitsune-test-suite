// See the README file for details.
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <kitsune.h>

using namespace kitsune;
using namespace std;

const size_t DEFAULT_ARRAY_SIZE = 1024 * 1024;

struct Vec {
  float x, y;
};

void random_fill(mobile_ptr<Vec> data, size_t N) {
  for (size_t i = 0; i < N; ++i) {
    data[i].x = rand() / (float)RAND_MAX;
    data[i].y = rand() / (float)RAND_MAX;
  }
}

__attribute__((noinline)) Vec vec_sum(const Vec &a, const Vec &b) {
  Vec sum;
  sum.x = a.x + b.x;
  sum.y = a.y + b.y;
  return sum;
}

__attribute__((noinline)) void parallel_work(mobile_ptr<Vec> dst,
                                             const mobile_ptr<Vec> src, int N) {
  forall(size_t i = 0; i < N; i++) {
    Vec sum = vec_sum(dst[i], src[i]);
    dst[i].x = sum.x;
    dst[i].y = sum.y;
  }
}

__attribute__((noinline)) void print(mobile_ptr<Vec> data, size_t N) {
  for (size_t i = 0; i < N; i++)
    printf("(%f %f)", data[i].x, data[i].y);
  printf("\n\n");
}

int main(int argc, char **argv) {
  size_t array_size = DEFAULT_ARRAY_SIZE;
  if (argc >= 2)
    array_size = atol(argv[1]);
  fprintf(stdout, "array size: %ld\n", array_size);

  mobile_ptr<Vec> data0(array_size);
  mobile_ptr<Vec> data1(array_size);
  random_fill(data0, array_size);
  random_fill(data1, array_size);

  print(data0, 10);
  auto start = chrono::steady_clock::now();
  parallel_work(data1, data0, array_size);
  print(data1, 10);
  auto end = chrono::steady_clock::now();
  cout << "Execution time: " << chrono::duration<double>(end - start).count()
       << endl;

  data0.free();
  data1.free();

  return 0;
}
