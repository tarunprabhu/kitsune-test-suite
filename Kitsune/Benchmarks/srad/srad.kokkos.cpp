#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#define __KOKKOS__
#include "srad.inc"

using DualViewFloat = Kokkos::DualView<float *, Kokkos::LayoutRight,
                                       Kokkos::DefaultExecutionSpace>;

using DualViewInt =
    Kokkos::DualView<int *, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace>;

template <> void randomFill(DualViewFloat &vwI, int rows, int cols) {
  const auto &I = vwI.view_host();
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I(i * cols + j) = (rand() / float(RAND_MAX));
    }
  }
}

int main(int argc, char *argv[]) {
  int mismatch = 0;
  Kokkos::initialize(argc, argv);
  {
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

    DualViewFloat I("I", rows * cols);
    DualViewFloat J("J", rows * cols);
    DualViewFloat c("c", rows * cols);
    DualViewInt iN("iN", rows);
    DualViewInt iS("iS", rows);
    DualViewInt jW("jW", cols);
    DualViewInt jE("jE", cols);
    DualViewFloat dN("dN", rows * cols);
    DualViewFloat dS("dS", rows * cols);
    DualViewFloat dW("dW", rows * cols);
    DualViewFloat dE("dE", rows * cols);

    header("kokkos", I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, size_I, size_R,
           rows, cols, r1, r2, c1, c2, niter);

    const auto &I_p = I.view_device();
    const auto &J_p = J.view_device();
    const auto &iN_p = iN.view_device();
    const auto &iS_p = iS.view_device();
    const auto &jE_p = jE.view_device();
    const auto &jW_p = jW.view_device();
    const auto &dN_p = dN.view_device();
    const auto &dS_p = dS.view_device();
    const auto &dW_p = dW.view_device();
    const auto &dE_p = dE.view_device();
    const auto &c_p = c.view_device();

    total.start();
    init.start();

    iN.modify_device();
    iS.modify_device();
    // clang-format off
    Kokkos::parallel_for(rows, KOKKOS_LAMBDA(const int i) {
      iN_p(i) = i - 1;
      iS_p(i) = i + 1;
    });
    Kokkos::fence();
    // clang-format on

    jW.modify_device();
    jE.modify_device();
    // clang-format off
    Kokkos::parallel_for(cols, KOKKOS_LAMBDA(const int j) {
      jW_p(j) = j - 1;
      jE_p(j) = j + 1;
    });
    Kokkos::fence();
    // clang-format on

    iN.sync_host();
    iN.modify_host();
    iN.view_host()(0) = 0;

    iS.sync_host();
    iS.modify_host();
    iS.view_host()(rows - 1) = rows - 1;

    jW.sync_host();
    jW.modify_host();
    jW.view_host()(0) = 0;

    jE.sync_host();
    jE.modify_host();
    jE.view_host()(cols - 1) = cols - 1;

    I.sync_device();
    J.modify_device();
    // clang-format off
    Kokkos::parallel_for(size_I, KOKKOS_LAMBDA(const int k) {
      J_p(k) = std::exp(I_p(k));
    });
    Kokkos::fence();
    // clang-format on
    init.stop();

    iN.sync_device();
    iS.sync_device();
    jE.sync_device();
    jW.sync_device();
    iters.start();
    for (int iter = 0; iter < niter; iter++) {
      float sum = 0, sum2 = 0;

      J.sync_host();
      for (int i = r1; i <= r2; i++) {
        for (int j = c1; j <= c2; j++) {
          tmp = J.view_host()(i * cols + j);
          sum += tmp;
          sum2 += tmp * tmp;
        }
      }
      meanROI = sum / size_R;
      varROI = (sum2 / size_R) - meanROI * meanROI;
      q0sqr = varROI / (meanROI * meanROI);

      loop1.start();
      c.modify_device();
      // clang-format off
      Kokkos::parallel_for(rows, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < cols; j++) {
          int k = i * cols + j;
          float Jc = J_p(k);

          // directional derivatives
          dN_p(k) = J_p(iN_p(i) * cols + j) - Jc;
          dS_p(k) = J_p(iS_p(i) * cols + j) - Jc;
          dE_p(k) = J_p(i * cols + jE_p(j)) - Jc;
          dW_p(k) = J_p(i * cols + jW_p(j)) - Jc;

          float G2 = (dN_p(k) * dN_p(k) + dS_p(k) * dS_p(k) +
                      dW_p(k) * dW_p(k) + dE_p(k) * dE_p(k)) /
            (Jc * Jc);
          float L = (dN_p(k) + dS_p(k) + dW_p(k) + dE_p(k)) / Jc;
          float num = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
          float den = 1 + (.25f * L);
          float qsqr = num / (den * den);

          // diffusion coefficient (equ 33)
          den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
          c_p[k] = 1.0f / (1.0f + den);

          // saturate diffusion coefficient
          if (c_p(k) < 0)
            c_p(k) = 0.0;
          else if (c_p(k) > 1)
            c_p(k) = 1.0;
        }
      });
      // clang-format on
      Kokkos::fence();
      loop1.stop();

      loop2.start();
      J.modify_device();
      // clang-format off
      Kokkos::parallel_for(rows, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j < cols; j++) {
          // current index
          int k = i * cols + j;
          // diffusion coefficient
          float cN = c_p(k);
          float cS = c_p(iS_p(i) * cols + j);
          float cW = c_p(k);
          float cE = c_p(i * cols + jE_p(j));
          // divergence (equ 58)
          float D = cN * dN_p(k) + cS * dS_p(k) + cW * dW_p(k) + cE * dE_p(k);
          // image update (equ 61)
          J_p(k) = J_p(k) + 0.25 * lambda * D;
        }
      });
      // clang-format on
      Kokkos::fence();
      loop2.stop();
    }
    J.sync_host();
    iters.stop();
    total.stop();

    mismatch = footer(tg, I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, rows, cols,
                      outFile, cpuRefFile, gpuRefFile);
  }
  return mismatch;
}
