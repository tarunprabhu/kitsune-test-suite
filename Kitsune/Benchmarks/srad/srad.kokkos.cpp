#include <Kokkos_Core.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <kitsune.h>

#include "fpcmp.h"
#include "timing.h"

#include "srad.inc"

int main(int argc, char *argv[]) {
  int mismatch = 0;
  Kokkos::initialize(argc, argv);
  {
    kitsune::mobile_ptr<float> I, J;
    kitsune::mobile_ptr<int> iN, iS, jE, jW;
    kitsune::mobile_ptr<float> dN, dS, dW, dE;
    kitsune::mobile_ptr<float> c;
    int rows, cols, size_I, size_R, niter;
    float q0sqr, tmp, meanROI, varROI;
    int r1, r2, c1, c2;
    float lambda;
    std::string cpuRefFile, gpuRefFile;
    std::string outFile;

    TimerGroup tg("srad");
    Timer &total = tg.add("total", "Total");
    Timer &init = tg.add("init", "Init");
    Timer &iters = tg.add("iters", "Compute");
    Timer &loop1 = tg.add("loop1", "Loop 1");
    Timer &loop2 = tg.add("loop2", "Loop 2");

    parseCommandLineInto(argc, argv, niter, rows, cols, r1, r2, c1, c2, lambda,
                         outFile, cpuRefFile, gpuRefFile);
    header("kokkos", I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, size_I, size_R,
           rows, cols, r1, r2, c1, c2, niter);

    // FIXME: Kokkos cannot deal with the mobile_ptr type for ... reasons. We
    // could try to find a way to make that type Kokkos-friendly, or just wait
    // until the [mobile_ptr] attribute is implemented which should make
    // Kokkos happy.
    float *[[kitsune::mobile]] I_p = I.get();
    float *[[kitsune::mobile]] J_p = J.get();
    int *[[kitsune::mobile]] iN_p = iN.get();
    int *[[kitsune::mobile]] iS_p = iS.get();
    int *[[kitsune::mobile]] jE_p = jE.get();
    int *[[kitsune::mobile]] jW_p = jW.get();
    float *[[kitsune::mobile]] dN_p = dN.get();
    float *[[kitsune::mobile]] dS_p = dS.get();
    float *[[kitsune::mobile]] dW_p = dW.get();
    float *[[kitsune::mobile]] dE_p = dE.get();
    float *[[kitsune::mobile]] c_p = c.get();

    total.start();
    init.start();
    // clang-format off
    Kokkos::parallel_for(rows, KOKKOS_LAMBDA(const int i) {
      iN_p[i] = i - 1;
      iS_p[i] = i + 1;
    });
    Kokkos::fence();
    // clang-format on

    // clang-format off
    Kokkos::parallel_for(cols, KOKKOS_LAMBDA(const int j) {
      jW_p[j] = j - 1;
      jE_p[j] = j + 1;
    });
    Kokkos::fence();
    // clang-format on

    iN[0] = 0;
    iS[rows - 1] = rows - 1;
    jW[0] = 0;
    jE[cols - 1] = cols - 1;

    // clang-format off
    Kokkos::parallel_for(size_I, KOKKOS_LAMBDA(const int k) {
      J_p[k] = std::exp(I_p[k]);
    });
    Kokkos::fence();
    // clang-format on
    init.stop();

    iters.start();
    for (int iter = 0; iter < niter; iter++) {
      float sum = 0, sum2 = 0;

      for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
          tmp = J[i * cols + j];
          sum += tmp;
          sum2 += tmp * tmp;
        }
      }
      meanROI = sum / size_R;
      varROI = (sum2 / size_R) - meanROI * meanROI;
      q0sqr = varROI / (meanROI * meanROI);

      loop1.start();
      // clang-format off
      Kokkos::parallel_for(rows, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < cols; j++) {
          int k = i * cols + j;
          float Jc = J_p[k];

          // directional derivatives
          dN_p[k] = J_p[iN_p[i] * cols + j] - Jc;
          dS_p[k] = J_p[iS_p[i] * cols + j] - Jc;
          dE_p[k] = J_p[i * cols + jE_p[j]] - Jc;
          dW_p[k] = J_p[i * cols + jW_p[j]] - Jc;

          float G2 = (dN_p[k] * dN_p[k] + dS_p[k] * dS_p[k] +
                      dW_p[k] * dW_p[k] + dE_p[k] * dE_p[k]) /
            (Jc * Jc);
          float L = (dN_p[k] + dS_p[k] + dW_p[k] + dE_p[k]) / Jc;
          float num = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
          float den = 1 + (.25f * L);
          float qsqr = num / (den * den);

          // diffusion coefficient (equ 33)
          den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
          c_p[k] = 1.0f / (1.0f + den);

          // saturate diffusion coefficient
          if (c_p[k] < 0)
            c_p[k] = 0.0;
          else if (c_p[k] > 1)
            c_p[k] = 1.0;
        }
      });
      // clang-format on
      Kokkos::fence();
      loop1.stop();

      loop2.start();
      // clang-format off
      Kokkos::parallel_for(rows, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < cols; j++) {
          // current index
          int k = i * cols + j;
          // diffusion coefficient
          float cN = c_p[k];
          float cS = c_p[iS_p[i] * cols + j];
          float cW = c_p[k];
          float cE = c_p[i * cols + jE_p[j]];
          // divergence (equ 58)
          float D = cN * dN_p[k] + cS * dS_p[k] + cW * dW_p[k] + cE * dE_p[k];
          // image update (equ 61)
          J_p[k] = J_p[k] + 0.25 * lambda * D;
        }
      });
      // clang-format on
      loop2.stop();
    }
    iters.stop();
    total.stop();

    mismatch = footer(tg, I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, rows, cols,
                      outFile, cpuRefFile, gpuRefFile);
  }
  return mismatch;
}
