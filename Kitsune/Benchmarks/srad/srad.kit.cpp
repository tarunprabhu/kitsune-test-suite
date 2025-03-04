#include <cmath>
#include <filesystem>
#include <iostream>
#include <kitsune.h>

#include "timing.h"

namespace fs = std::filesystem;
using namespace kitsune;

#include "srad.inc"

int main(int argc, char *argv[]) {
  mobile_ptr<float> I, J;
  mobile_ptr<int> iN, iS, jE, jW;
  mobile_ptr<float> dN, dS, dW, dE;
  mobile_ptr<float> c;
  int rows, cols, size_I, size_R, niter;
  float q0sqr, tmp, meanROI, varROI;
  int r1, r2, c1, c2;
  float lambda;
  std::string cpuRefFile, gpuRefFile;
  std::string outFile;

  TimerGroup tg("srad");
  Timer &main = tg.add("main", "Total");
  Timer &init = tg.add("init", "Init");
  Timer &iters = tg.add("iters", "Compute");
  Timer &loop1 = tg.add("loop1", "Loop 1");
  Timer &loop2 = tg.add("loop2", "Loop 2");

  parseCommandLineInto(argc, argv, niter, rows, cols, r1, r2, c1, c2, lambda,
                       outFile, cpuRefFile, gpuRefFile);
  header("forall", I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, size_I, size_R,
         rows, cols, r1, r2, c1, c2, niter);

  main.start();
  init.start();
  // clang-format off
  forall(int i = 0; i < rows; i++) {
    iN[i] = i - 1;
    iS[i] = i + 1;
  }
  // clang-format on

  // clang-format off
  forall(int j = 0; j < cols; j++) {
    jW[j] = j - 1;
    jE[j] = j + 1;
  }
  // clang-format on

  iN[0] = 0;
  iS[rows - 1] = rows - 1;
  jW[0] = 0;
  jE[cols - 1] = cols - 1;

  // clang-format off
  forall(int k = 0; k < size_I; k++) {
    J[k] = (float)exp(I[k]);
  }
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
    forall(int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        int k = i * cols + j;
        float Jc = J[k];

        // directional derivatives
        dN[k] = J[iN[i] * cols + j] - Jc;
        dS[k] = J[iS[i] * cols + j] - Jc;
        dE[k] = J[i * cols + jE[j]] - Jc;
        dW[k] = J[i * cols + jW[j]] - Jc;

        float G2 =
            (dN[k] * dN[k] + dS[k] * dS[k] + dW[k] * dW[k] + dE[k] * dE[k]) /
            (Jc * Jc);
        float L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;
        float num = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
        float den = 1 + (.25f * L);
        float qsqr = num / (den * den);

        // diffusion coefficient (equ 33)
        den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
        c[k] = 1.0f / (1.0f + den);

        // saturate diffusion coefficient
        if (c[k] < 0)
          c[k] = 0.0;
        else if (c[k] > 1)
          c[k] = 1.0;
      }
    }
    loop1.stop();

    loop2.start();
    forall(int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        // current index
        int k = i * cols + j;
        // diffusion coefficient
        float cN = c[k];
        float cS = c[iS[i] * cols + j];
        float cW = c[k];
        float cE = c[i * cols + jE[j]];
        // divergence (equ 58)
        float D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];
        // image update (equ 61)
        J[k] = J[k] + 0.25 * lambda * D;
      }
    }
    loop2.stop();
  }
  iters.stop();
  main.stop();

  int mismatch = footer(tg, I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, rows, cols,
                        outFile, cpuRefFile, gpuRefFile);
  return mismatch;
}
