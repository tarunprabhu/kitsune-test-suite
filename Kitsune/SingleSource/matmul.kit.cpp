#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <kitsune.h>
#include <vector>

using namespace kitsune;

void matrix_multiplication(mobile_ptr<double> A, mobile_ptr<double> B,
                           mobile_ptr<double> C, size_t M, size_t N, size_t K) {
  forall(size_t tid = 0; tid < M * N; tid++) {
    size_t m = tid / N;
    size_t n = tid % N;
    double sum = 0.0;
    for (size_t k = 0; k < K; k++) {
      sum += A[m * K + k] * B[n * K + k];
    }
    C[tid] = sum;
  }
}

int main(int argc, char *argv[]) {
  using namespace std;

  size_t m = 128, n = 128, k = 32;
  unsigned iterations = 10;

  if (argc > 5) {
    std::cerr << "Usage: " << argv[0] << " M N K [iteration]\n\n";
    return 1;
  }
  if (argc > 4)
    iterations = atoi(argv[4]);
  if (argc > 3)
    k = atoi(argv[3]);
  if (argc > 2)
    n = atoi(argv[2]);
  if (argc > 1)
    m = atoi(argv[1]);

  cout << setprecision(5);
  cout << "\n";
  cout << "---- Matrix multiplication with transposed B benchmark (forall) "
          "----\n"
       << "  Matrix size: " << m << " x " << n << " x " << k << ".\n\n";
  cout << "  Allocating matrices..." << std::flush;

  mobile_ptr<double> A(m * k);
  mobile_ptr<double> B(n * k);
  mobile_ptr<double> C(m * n);

  // Assuming A, B, and C are already initialized and B is already transposed
  cout << "  done.\n\n";

  double elapsed_time;
  double avg_time = 0.0;
  double min_time = 100000.0;
  double max_time = 0.0;

  for (unsigned t = 0; t < iterations; t++) {
    auto start_time = chrono::steady_clock::now();
    // Use transposed B for multiplication
    // matrix_multiplication(A, B, C, m);

    // Use transposed B for multiplication
    matrix_multiplication(A, B, C, m, n, k);
    auto end_time = chrono::steady_clock::now();
    elapsed_time = chrono::duration<double>(end_time - start_time).count();
    if (elapsed_time < min_time)
      min_time = elapsed_time;
    if (elapsed_time > max_time)
      max_time = elapsed_time;
    cout << "\t" << t << ". iteration time: " << elapsed_time << " seconds.\n";
    if (t)
      avg_time += elapsed_time;
  }
  avg_time = avg_time / (iterations - 1);
  cout << "  Total time: " << avg_time << " seconds.\n\n";
  A.free();
  B.free();
  C.free();
  return 0;
}
