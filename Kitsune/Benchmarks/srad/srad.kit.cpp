#include <cmath>
#include <iostream>
#include <kitsune.h>
#include <timing.h>

using namespace kitsune;

static void random_matrix(mobile_ptr<float> I, unsigned int rows,
                          unsigned int cols) {
  srand(7);
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      I[i * cols + j] = rand() / (float)RAND_MAX;
    }
  }

  std::cout << "  initial input data:\n";
  for (unsigned int i = 0; i < 10; i++) {
    std::cout << "   ";
    for (unsigned int j = 0; j < 10; j++)
      std::cout << I[i * cols + j] << " ";
    std::cout << "...\n";
  }
  std::cout << "   ...\n";
}

[[noreturn]]
static void usage(int argc, char **argv) {
  std::cerr << "Usage: " << argv[0]
            << " <rows> <cols> <y1> <y2> <x1> <x2> <lambda> <no. of iter>\n";
  std::cerr << "\t<rows>        - number of rows\n";
  std::cerr << "\t<cols>        - number of cols\n";
  std::cerr << "\t<y1>          - y1 value of the speckle\n";
  std::cerr << "\t<y2>          - y2 value of the speckle\n";
  std::cerr << "\t<x1>          - x1 value of the speckle\n";
  std::cerr << "\t<x2>          - x2 value of the speckle\n";
  std::cerr << "\t<lambda>      - lambda (0,1\n";
  std::cerr << "\t<no. of iter> - number of iterations\n";
  exit(1);
}

int main(int argc, char *argv[]) {
  int rows, cols, size_I, size_R, niter;
  mobile_ptr<float> I, J;
  float q0sqr, tmp, meanROI, varROI;
  float Jc, G2, L, num, den, qsqr;
  mobile_ptr<int> iN, iS, jE, jW;
  mobile_ptr<float> dN, dS, dW, dE;
  int r1, r2, c1, c2;
  float cN, cS, cW, cE;
  float D;
  mobile_ptr<float> c;
  float lambda;
  Timer main("main");
  Timer init("init");
  Timer iters("iters");
  Timer loop1("loop1");
  Timer loop2("loop2");

  if (argc == 9) {
    rows = atoi(argv[1]);   // number of rows in the domain
    cols = atoi(argv[2]);   // number of cols in the domain
    r1 = atoi(argv[3]);     // y1 position of the speckle
    r2 = atoi(argv[4]);     // y2 position of the speckle
    c1 = atoi(argv[5]);     // x1 position of the speckle
    c2 = atoi(argv[6]);     // x2 position of the speckle
    lambda = atof(argv[7]); // Lambda value
    niter = atoi(argv[8]);  // number of iterations
  } else if (argc == 1) {
    // run with a default configuration.
    rows = 6400;
    cols = 6400;
    r1 = 0;
    r2 = 127;
    c1 = 0;
    c2 = 127;
    lambda = 0.5;
    niter = 2000;
  } else {
    usage(argc, argv);
  }

  if ((rows % 16 != 0) || (cols % 16 != 0)) {
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }

  std::cout << "\n";
  std::cout << "---- srad benchmark (forall) ----\n"
            << "  Row size    : " << rows << ".\n"
            << "  Column size : " << cols << ".\n"
            << "  Iterations  : " << niter << ".\n\n";

  std::cout << "  Allocating arrays and building random matrix..."
            << std::flush;

  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  I.alloc(size_I);
  J.alloc(size_I);
  c.alloc(size_I);
  iN.alloc(rows);
  iS.alloc(rows);
  jW.alloc(cols);
  jE.alloc(cols);
  dN.alloc(size_I);
  dS.alloc(size_I);
  dW.alloc(size_I);
  dE.alloc(size_I);

  random_matrix(I, rows, cols);

  std::cout << "  Running benchmark...\n" << std::flush;

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

  // TODO: Actually check that the output is correct.
  size_t errors = 0;

  FILE *fp = fopen("srad-forall-output.dat", "wb");
  if (fp != NULL) {
    fwrite((void *)J.get(), sizeof(float), size_I, fp);
    fclose(fp);
  }

  json(std::cout, "srad", {main, init, iters, loop1, loop2});

  return 0;
}
