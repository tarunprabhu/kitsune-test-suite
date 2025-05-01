#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdlib.h>

#include "fpcmp.h"
#include "timing.h"

#include "srad.inc"

__global__ void initNS(int *iN, int *iS, int rows) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < rows) {
    iN[i] = i - 1;
    iS[i] = i + 1;
  }
}

__global__ void initWE(int *jW, int *jE, int cols) {
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  if (j < cols) {
    jW[j] = j - 1;
    jE[j] = j + 1;
  }
}

__global__ void initJ(float *J, float *I, int size_I) {
  int k = blockDim.x * blockIdx.x + threadIdx.x;
  if (k < size_I)
    J[k] = std::exp(I[k]);
}

__global__ void loop1(int rows, int cols, float *J, int *iN, int *iS, int *jE,
                      int *jW, float *dN, float *dS, float *dE, float *dW,
                      float q0sqr, float *c) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < rows) {
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
}

__global__ void loop2(int rows, int cols, float *J, float *c, int *iS, int *jE,
                      float *dN, float *dS, float *dE, float *dW,
                      float lambda) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < rows) {
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
}

int main(int argc, char *argv[]) {
  float *I, *J;
  int *iN, *iS, *jE, *jW;
  float *dN, *dS, *dW, *dE;
  float *c;
  int rows, cols, size_I, size_R, niter;
  float q0sqr, tmp, meanROI, varROI;
  int r1, r2, c1, c2;
  float lambda;
  std::string cpuRefFile, gpuRefFile;
  std::string outFile;
  unsigned threadsPerBlock;

  TimerGroup tg("srad");
  Timer &total = tg.add("total", "Total");
  Timer &init = tg.add("init", "Init");
  Timer &iters = tg.add("iters", "Compute");
  Timer &tl1 = tg.add("loop1", "Loop 1");
  Timer &tl2 = tg.add("loop2", "Loop 2");

  parseCommandLineInto(argc, argv, niter, rows, cols, r1, r2, c1, c2, lambda,
                       outFile, cpuRefFile, gpuRefFile, &threadsPerBlock);

  header("cuda", I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, size_I, size_R, rows,
         cols, r1, r2, c1, c2, niter);

  unsigned bpgRows = (rows + threadsPerBlock - 1) / threadsPerBlock;
  unsigned bpgCols = (cols + threadsPerBlock - 1) / threadsPerBlock;
  unsigned bpgSize = (size_I + threadsPerBlock - 1) / threadsPerBlock;

  total.start();
  init.start();
  initNS<<<bpgRows, threadsPerBlock>>>(iN, iS, rows);
  cudaDeviceSynchronize();

  initWE<<<bpgCols, threadsPerBlock>>>(jW, jE, cols);
  cudaDeviceSynchronize();

  iN[0] = 0;
  iS[rows - 1] = rows - 1;
  jW[0] = 0;
  jE[cols - 1] = cols - 1;

  initJ<<<bpgSize, threadsPerBlock>>>(J, I, size_I);
  cudaDeviceSynchronize();
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

    tl1.start();
    loop1<<<bpgRows, threadsPerBlock>>>(rows, cols, J, iN, iS, jE, jW, dN, dS,
                                        dE, dW, q0sqr, c);
    cudaDeviceSynchronize();
    tl1.stop();

    tl2.start();
    loop2<<<bpgRows, threadsPerBlock>>>(rows, cols, J, c, iS, jE, dN, dS, dE,
                                        dW, lambda);
    cudaDeviceSynchronize();
    tl2.stop();
  }
  iters.stop();
  total.stop();

  int mismatch = footer(tg, I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, rows, cols,
                        outFile, cpuRefFile, gpuRefFile);
  return mismatch;
}
