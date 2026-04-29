// SRAD (Speckle Reducing Anisotropic Diffusion) benchmark from the Rodinia
// suite

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>

#include <kitsune.h>

static const char *usage =
    "srad [OPTIONS] [iters] [rows] [cols] [y1] [y2] [x1] [x2] [lambda]\n"
    "Try `srad -h` for more information";

// clang-format off
static const char *help =
  "SRAD (Speckle Reducing Anisotropic Diffusion) from the Rodinia suite\n"
  "\n"
  "    srad [OPTIONS] [iters] [rows] [cols] [y1] [y2] [x1] [x2] [lambda]\n"
  "\n"
  "OPTIONS\n"
  "\n"
  "    -c <FILE>  Path to the reference output file\n"
  "    -h         Print help and exit\n"
  "\n"
  "ARGUMENTS\n"
  "\n"
  "    iters      Number of iterations           [100]\n"
  "    rows       Number of rows                 [320]\n"
  "    cols       Number of columns              [320]\n"
  "    y1         Y1 value of the speckle        [0]\n"
  "    y2         Y2 value of the speckle        [127]\n"
  "    x1         X1 value of the speckle        [0]\n"
  "    x2         X2 value of the speckle        [127]\n"
  "    lambda     lambda. Value must be in [0,1) [0.5]\n";
// clang-format on

static float relErr(float actual, float expected) {
  return std::abs(actual - expected) / (std::abs(expected) + 1);
}

static bool checkRelErr(float actual, float expected, float epsilon) {
  return relErr(actual, expected) > epsilon;
}

static size_t checkRelErr(float *actual, float *expected, size_t n,
                          float epsilon) {
  size_t errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (checkRelErr(actual[i], expected[i], epsilon))
      ++errors;
  return errors;
}

[[clang::noinline]]
static std::string getOutFile(char *argv[]) {
  return std::filesystem::path(argv[0]).filename().string() + ".dat";
}

[[clang::noinline]]
static void
setup(float *[[kitsune::mobile]] & I, float *[[kitsune::mobile]] & J,
      float *[[kitsune::mobile]] & c, int *[[kitsune::mobile]] & iN,
      int *[[kitsune::mobile]] & iS, int *[[kitsune::mobile]] & jW,
      int *[[kitsune::mobile]] & jE, float *[[kitsune::mobile]] & dN,
      float *[[kitsune::mobile]] & dS, float *[[kitsune::mobile]] & dW,
      float *[[kitsune::mobile]] & dE, int size_I, int rows, int cols) {
  I = (float *[[kitsune::mobile]])kitsune_mobile_alloc(size_I * sizeof(float));
  J = (float *[[kitsune::mobile]])kitsune_mobile_alloc(size_I * sizeof(float));
  c = (float *[[kitsune::mobile]])kitsune_mobile_alloc(size_I * sizeof(float));
  iN = (int *[[kitsune::mobile]])kitsune_mobile_alloc(rows * sizeof(int));
  iS = (int *[[kitsune::mobile]])kitsune_mobile_alloc(rows * sizeof(int));
  jW = (int *[[kitsune::mobile]])kitsune_mobile_alloc(cols * sizeof(int));
  jE = (int *[[kitsune::mobile]])kitsune_mobile_alloc(cols * sizeof(int));
  dN = (float *[[kitsune::mobile]])kitsune_mobile_alloc(size_I * sizeof(float));
  dS = (float *[[kitsune::mobile]])kitsune_mobile_alloc(size_I * sizeof(float));
  dW = (float *[[kitsune::mobile]])kitsune_mobile_alloc(size_I * sizeof(float));
  dE = (float *[[kitsune::mobile]])kitsune_mobile_alloc(size_I * sizeof(float));

  srand(7);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < cols; ++j)
      I[i * cols + j] = rand() / float(RAND_MAX);

  for (int i = 0; i < rows; i++) {
    iN[i] = i - 1;
    iS[i] = i + 1;
  }

  for (int j = 0; j < cols; j++) {
    jW[j] = j - 1;
    jE[j] = j + 1;
  }

  iN[0] = 0;
  iS[rows - 1] = rows - 1;
  jW[0] = 0;
  jE[cols - 1] = cols - 1;

  for (int k = 0; k < size_I; k++)
    J[k] = std::exp(I[k]);
}

[[clang::noinline]]
static void teardown(float *[[kitsune::mobile]] I, float *[[kitsune::mobile]] J,
                     float *[[kitsune::mobile]] c, int *[[kitsune::mobile]] iN,
                     int *[[kitsune::mobile]] iS, int *[[kitsune::mobile]] jE,
                     int *[[kitsune::mobile]] jW, float *[[kitsune::mobile]] dN,
                     float *[[kitsune::mobile]] dS,
                     float *[[kitsune::mobile]] dW,
                     float *[[kitsune::mobile]] dE) {
  kitsune_mobile_free(I);
  kitsune_mobile_free(J);
  kitsune_mobile_free(c);
  kitsune_mobile_free(iN);
  kitsune_mobile_free(iS);
  kitsune_mobile_free(jW);
  kitsune_mobile_free(jE);
  kitsune_mobile_free(dN);
  kitsune_mobile_free(dS);
  kitsune_mobile_free(dW);
  kitsune_mobile_free(dE);
}

[[clang::noinline]]
static void save(const std::string &outFile, const float *J, int size_I) {
  FILE *fp = fopen(outFile.c_str(), "wb");
  fwrite(J, sizeof(float), size_I, fp);
  fclose(fp);
}

[[clang::noinline]]
static int report(long errs) {
  if (errs)
    printf("FAIL: %ld errors\n", errs);
  else
    printf("PASS\n");
  return errs ? 1 : 0;
}

// This cannot currently be 1e-6 since the hip version will fail. At some point,
// we should try to sort out at least some of the floating point issues so we
// can make this a bit tighter.
static constexpr float epsilon = 1e-5;

[[clang::noinline]]
static long check(const std::string &outFile, const std::string &checkFile,
                  int size_I) {
  FILE *fa = nullptr;
  FILE *fe = nullptr;
  float *ev = nullptr;
  float *av = nullptr;
  int ne = 0;
  int na = 0;
  size_t errors = 0;

  fa = fopen(outFile.c_str(), "rb");
  if (!fa) {
    errors = -2;
    goto cleanup;
  }

  fe = fopen(checkFile.c_str(), "rb");
  if (!fe) {
    errors = -1;
    goto cleanup;
  }

  ev = (float *)malloc(size_I * sizeof(float));
  av = (float *)malloc(size_I * sizeof(float));

  ne = fread(ev, sizeof(float), size_I, fe);
  na = fread(av, sizeof(float), size_I, fa);
  if (ne != na && na != size_I * sizeof(float)) {
    errors = -1;
    goto cleanup;
  }

  if (size_t errs = checkRelErr(av, ev, size_I, epsilon)) {
    errors = errs;
    goto cleanup;
  }

  if (fgetc(fe) != EOF or fgetc(fa) != EOF) {
    errors = -2;
    goto cleanup;
  }

cleanup:
  free(ev);
  free(av);
  if (fe)
    fclose(fe);
  if (fa)
    fclose(fa);

  return errors;
}

[[clang::noinline]]
static void test(float *[[kitsune::mobile]] J, float *[[kitsune::mobile]] c,
                 int *[[kitsune::mobile]] iN, int *[[kitsune::mobile]] iS,
                 int *[[kitsune::mobile]] jW, int *[[kitsune::mobile]] jE,
                 float *[[kitsune::mobile]] dN, float *[[kitsune::mobile]] dS,
                 float *[[kitsune::mobile]] dW, float *[[kitsune::mobile]] dE,
                 float lambda, int niter, int size_R, int rows, int cols,
                 int r1, int r2, int c1, int c2) {
  for (int iter = 0; iter < niter; iter++) {
    float sum = 0, sum2 = 0;
    for (int i = r1; i <= r2; i++) {
      for (int j = c1; j <= c2; j++) {
        float tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }

    float meanROI = sum / size_R;
    float varROI = (sum2 / size_R) - meanROI * meanROI;
    float q0sqr = varROI / (meanROI * meanROI);

    // clang-format off
    forall(int i = 0; i < rows; i++) {
      forall (int j = 0; j < cols; j++) {
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
    // clang-format on

    // clang-format off
    forall (int i = 0; i < rows; i++) {
      forall (int j = 0; j < cols; j++) {
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
    // clang-format on
  }
}

int main(int argc, char *argv[]) {
  float *[[kitsune::mobile]] I = nullptr;
  float *[[kitsune::mobile]] J = nullptr;
  float *[[kitsune::mobile]] c = nullptr;
  int *[[kitsune::mobile]] iN = nullptr;
  int *[[kitsune::mobile]] iS = nullptr;
  int *[[kitsune::mobile]] jW = nullptr;
  int *[[kitsune::mobile]] jE = nullptr;
  float *[[kitsune::mobile]] dN = nullptr;
  float *[[kitsune::mobile]] dS = nullptr;
  float *[[kitsune::mobile]] dW = nullptr;
  float *[[kitsune::mobile]] dE = nullptr;
  int rows, cols, size_I, size_R, niter;
  int r1, r2, c1, c2;
  float lambda;
  std::string checkFile, outFile;

  niter = 100;
  rows = 320;
  cols = 320;
  r1 = 0;
  r2 = 127;
  c1 = 0;
  c2 = 127;
  lambda = 0.5;
  outFile = getOutFile(argv);

  int flag;
  while ((flag = getopt(argc, argv, "c:h")) != -1) {
    switch (flag) {
    case 'c':
      checkFile = optarg;
      break;
    case 'h':
      printf("%s\n", help);
      exit(0);
    default:
      printf("ERROR: Unknown option '%c'\n\n", optopt);
      printf("%s\n", usage);
      exit(1);
    }
  }

  char **args = &argv[optind];
  int argn = argc - optind;
  if (argn > 0)
    niter = atoi(args[0]);
  if (argn > 1)
    rows = atoi(args[1]);
  if (argn > 2)
    cols = atoi(args[2]);
  if (argn > 3)
    r1 = atoi(args[3]);
  if (argn > 4)
    r2 = atoi(args[4]);
  if (argn > 5)
    c1 = atoi(args[5]);
  if (argn > 6)
    c2 = atoi(args[6]);
  if (argn > 7)
    lambda = atof(args[7]);
  if (argn > 8) {
    printf("%s\n", usage);
    exit(1);
  }

  if (rows % 16) {
    printf("ERROR: rows must be a multiple of 16\n");
    exit(1);
  } else if (cols % 16) {
    printf("ERROR: cols must be a multiple of 16\n");
    exit(1);
  }

  size_I = rows * cols;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  long errs = 0;
  setup(I, J, c, iN, iS, jW, jE, dN, dS, dW, dE, size_I, rows, cols);
  test(J, c, iN, iS, jW, jE, dN, dS, dW, dE, lambda, niter, size_R, rows, cols,
       r1, r2, c1, c2);
  save(outFile, (float *)J, size_I);
  if (checkFile.size())
    errs = check(outFile, checkFile, size_I);
  teardown(I, J, c, iN, iS, jW, jE, dN, dS, dW, dE);

  return report(errs);
}
