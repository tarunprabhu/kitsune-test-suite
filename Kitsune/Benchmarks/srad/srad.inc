// Declarations for getopt. We could just include unistd.h, but we want to keep
// the main test code simple and not include the header there. We could include
// it here, but since including a file such as this is already rather ugly, I
// don't see why it should be made even worse.
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int getopt(int argc, char *argv[], const char *optstring);

extern char *optarg;
extern int optind, opterr, optopt;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

static const char *usage =
    "srad [OPTIONS] [iters] [rows] [cols] [y1] [y2] [x1] [x2] [lambda]\n"
    "Try `srad -h` for more information";

// clang-format off
static const char *help =
  "<<<<Short description of srad here>>>>\n"
  "\n"
  "    srad [OPTIONS] [iters] [rows] [cols] [y1] [y2] [x1] [x2] [lambda]\n"
  "\n"
  "OPTIONS\n"
  "\n"
  "    -c <FILE>  The path to the reference output file on a CPU. If srad was\n"
  "               built with a GPU tapir target, this option is ignored\n"
  "    -g <FILE>  The path to the reference output file on a GPU. If srad was\n"
  "               built with a CPU tapir target, this option is ignored\n"
  "    -h         Print help and exit\n"
  "    -j         Print the recorded times in JSON\n"
  "    -t <N>     Number of threads per block on the GPU. This is only used\n"
  "               by the .cu and .hip implementations of this benchmark [256]\n"
  "\n"
  "ARGUMENTS\n"
  "\n"
  "    iters      Number of iterations           [2000]\n"
  "    rows       Number of rows                 [6400]\n"
  "    cols       Number of columns              [6400]\n"
  "    y1         Y1 value of the speckle        [0]\n"
  "    y2         Y2 value of the speckle        [127]\n"
  "    x1         X1 value of the speckle        [0]\n"
  "    x2         X2 value of the speckle        [127]\n"
  "    lambda     lambda. Value must be in [0,1) [0.5]\n";
// clang-format on

// Show the times in JSON. Yes, it's global. Too bad.
static bool showJSON = false;

static void parseCommandLineInto(int argc, char *argv[], int &niter, int &rows,
                                 int &cols, int &r1, int &r2, int &c1, int &c2,
                                 float &lambda, std::string &outFile,
                                 std::string &cpuRefFile,
                                 std::string &gpuRefFile,
                                 unsigned *threadsPerBlock = nullptr) {
  niter = 2000;
  rows = 6400;
  cols = 6400;
  r1 = 0;
  r2 = 127;
  c1 = 0;
  c2 = 127;
  lambda = 0.5;
  cpuRefFile = "";
  gpuRefFile = "";
  if (threadsPerBlock) {
    // This is loosely for consistency with the launch parameters from kitsune.
    *threadsPerBlock = 256;
  }
  outFile = std::filesystem::path(argv[0]).filename().string() + ".dat";

  int flag;
  while ((flag = getopt(argc, argv, "c:g:hjt:")) != -1) {
    switch (flag) {
    case 'c':
      cpuRefFile = optarg;
      break;
    case 'g':
      gpuRefFile = optarg;
      break;
    case 'h':
      std::cout << help << std::endl;
      exit(0);
    case 'j':
      showJSON = true;
      break;
    case 't':
      if (threadsPerBlock) {
        *threadsPerBlock = std::stoi(optarg);
      } else {
        std::cout << "-------------------------------------------------\n";
        std::cout << " WARNING: Ignoring threads-per-block option (-t) \n";
        std::cout << "-------------------------------------------------\n";
      }
      break;
    default:
      std::cout << "ERROR: Unknown option '" << optopt << "'\n";
      std::cout << "\n";
      std::cout << usage << "\n";
      exit(1);
    }
  }

  if (threadsPerBlock) {
    std::cout << "Threads per block = " << *threadsPerBlock << "\n";
  }

  char **args = &argv[optind];
  int argn = argc - optind;
  if (argn == 0)
    return;
  if (argn > 0)
    niter = std::stoi(args[0]);
  if (argn > 1)
    rows = std::stoi(args[1]);
  if (argn > 2)
    cols = std::stoi(args[2]);
  if (argn > 3)
    r1 = std::stoi(args[3]);
  if (argn > 4)
    r2 = std::stoi(args[4]);
  if (argn > 5)
    c1 = std::stoi(args[5]);
  if (argn > 6)
    c2 = std::stoi(args[6]);
  if (argn > 7)
    lambda = std::stof(args[7]);
  if (argn > 8) {
    std::cout << usage << std::endl;
    exit(1);
  }

  if (rows % 16) {
    std::cout << "ERROR: rows must be a multiple of 16\n";
    exit(1);
  } else if (cols % 16) {
    std::cout << "ERROR: cols must be a multiple of 16\n";
    exit(1);
  }
}

template <typename Arr> static void randomFill(Arr &I, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I[i * cols + j] = (rand() / float(RAND_MAX));
    }
  }
}

template <typename FArr, typename IArr>
static void dealloc(FArr &I, FArr &J, FArr &c, IArr &iN, IArr &iS, IArr &jE,
                    IArr &jW, FArr &dN, FArr &dS, FArr &dW, FArr &dE) {
#if defined(__KOKKOS__)
#elif defined(__CUDACC__)
  cudaFree(I);
  cudaFree(J);
  cudaFree(c);
  cudaFree(iN);
  cudaFree(iS);
  cudaFree(jW);
  cudaFree(jE);
  cudaFree(dN);
  cudaFree(dS);
  cudaFree(dW);
  cudaFree(dE);
#elif defined(__HIP__)
  (void)hipFree(I);
  (void)hipFree(J);
  (void)hipFree(c);
  (void)hipFree(iN);
  (void)hipFree(iS);
  (void)hipFree(jW);
  (void)hipFree(jE);
  (void)hipFree(dN);
  (void)hipFree(dS);
  (void)hipFree(dW);
  (void)hipFree(dE);
#else
  I.free();
  J.free();
  c.free();
  iN.free();
  iS.free();
  jW.free();
  jE.free();
  dN.free();
  dS.free();
  dW.free();
  dE.free();
#endif // CPU
}

template <typename FArr, typename IArr>
static void alloc(FArr &I, FArr &J, FArr &c, IArr &iN, IArr &iS, IArr &jE,
                  IArr &jW, FArr &dN, FArr &dS, FArr &dW, FArr &dE, int rows,
                  int cols) {
  int size_I = rows * cols;
#if defined(__KOKKOS__)
#elif defined(__CUDACC__)
  cudaMallocManaged(&I, size_I * sizeof(float));
  cudaMallocManaged(&J, size_I * sizeof(float));
  cudaMallocManaged(&c, size_I * sizeof(float));
  cudaMallocManaged(&iN, rows * sizeof(int));
  cudaMallocManaged(&iS, rows * sizeof(int));
  cudaMallocManaged(&jW, cols * sizeof(int));
  cudaMallocManaged(&jE, cols * sizeof(int));
  cudaMallocManaged(&dN, size_I * sizeof(float));
  cudaMallocManaged(&dS, size_I * sizeof(float));
  cudaMallocManaged(&dW, size_I * sizeof(float));
  cudaMallocManaged(&dE, size_I * sizeof(float));
#elif defined(__HIP__)
  (void)hipMallocManaged(&I, size_I * sizeof(float));
  (void)hipMallocManaged(&J, size_I * sizeof(float));
  (void)hipMallocManaged(&c, size_I * sizeof(float));
  (void)hipMallocManaged(&iN, rows * sizeof(int));
  (void)hipMallocManaged(&iS, rows * sizeof(int));
  (void)hipMallocManaged(&jW, cols * sizeof(int));
  (void)hipMallocManaged(&jE, cols * sizeof(int));
  (void)hipMallocManaged(&dN, size_I * sizeof(float));
  (void)hipMallocManaged(&dS, size_I * sizeof(float));
  (void)hipMallocManaged(&dW, size_I * sizeof(float));
  (void)hipMallocManaged(&dE, size_I * sizeof(float));
#else
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
#endif // CPU
}

template <typename FArr, typename IArr>
static void header(const std::string &label, FArr &I, FArr &J, FArr &c,
                   IArr &iN, IArr &iS, IArr &jE, IArr &jW, FArr &dN, FArr &dS,
                   FArr &dW, FArr &dE, int &size_I, int &size_R, int rows,
                   int cols, int r1, int r2, int c1, int c2, int niter) {
  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  srand(7);
  std::cout << "\n";
  std::cout << "---- srad benchmark (" << label << ") ----\n"
            << "  Rows        " << rows << "\n"
            << "  Columns     " << cols << "\n"
            << "  Iterations  " << niter << "\n\n";

  std::cout << "  Allocating arrays and building random matrix..."
            << std::flush;

  alloc(I, J, c, iN, iS, jE, jW, dN, dS, dW, dE, rows, cols);
  randomFill(I, rows, cols);

  std::cout << "  done.\n\n";
  std::cout << "  Starting benchmark ... " << std::flush;
}

// FIXME: We may be able to lower the tolerance once Kitsune correctly handles
// the fp-contract option.
static constexpr float epsilon = 1e-4;

// Returns -1 if the sizes of the files mismatch. If the files are identical,
// returns 0. Otherwise returns the number of mismatches.
static size_t check(const std::string &outFile, const std::string &checkFile,
                    int n) {
  size_t errors = 0;

  FILE *fa = fopen(outFile.c_str(), "rb");
  FILE *fe = fopen(checkFile.c_str(), "rb");
  float *ev = (float *)malloc(n * sizeof(float));
  float *av = (float *)malloc(n * sizeof(float));

  int ne = fread(ev, sizeof(float), n, fe);
  int na = fread(av, sizeof(float), n, fa);
  if (ne != na && na != n * sizeof(float)) {
    errors = -1;
    goto cleanup;
  }

  if (size_t errs = checkRelErr(av, ev, n, epsilon)) {
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
  fclose(fe);
  fclose(fa);

  return errors;
}

static int check(const std::string &outFile, const std::string &checkFile,
                 int rows, int cols) {
  int mismatch = 0;
  if (not checkFile.size()) {
    std::cout << "--------------------------------------------------\n";
    std::cout << " WARNING: Not checking output (no reference file) \n";
    std::cout << "--------------------------------------------------\n";
  } else {
    char buf[16];
    snprintf(buf, 16, "%.6g", epsilon);
    std::cout << "\n  Checking with epsilon = " << buf << " ..." << std::flush;
    mismatch = check(outFile, checkFile, rows * cols);
    if (mismatch)
      std::cout << "  FAIL! Output mismatch at byte " << mismatch << "\n\n";
    else
      std::cout << "  pass\n\n";
  }
  return mismatch;
}

template <typename FArr, typename IArr>
static size_t footer(TimerGroup &tg, FArr &I, FArr &J, FArr &c, IArr &iN,
                     IArr &iS, IArr &jE, IArr &jW, FArr &dN, FArr &dS, FArr &dW,
                     FArr &dE, int rows, int cols, const std::string &outFile,
                     const std::string &cpuRefFile,
                     const std::string &gpuRefFile) {
  std::cout << "done\n\n";
  tg.prettyTimes(std::cout, 4);
  std::cout << "----\n\n";

  FILE *fp = fopen(outFile.c_str(), "wb");
#if defined(__KOKKOS__)
  for (size_t i = 0; i < rows * cols; ++i) {
    float v = J.view_host()(i);
    fwrite(&v, sizeof(float), 1, fp);
  }
#else
  fwrite(&*J, sizeof(float), rows * cols, fp);
#endif
  fclose(fp);

  std::string checkFile = cpuRefFile;
#if defined(__CUDACC__) || defined(__HIP__) || defined(__KOKKOS__)
  checkFile = gpuRefFile;
#else
  std::string tt = __kitsune_tt__;
  if (tt == "cuda" or tt == "hip")
    checkFile = gpuRefFile;
#endif // !__CUDACC__ && !__HIP__

  size_t errors = check(outFile, checkFile, rows, cols);
  dealloc(I, J, c, iN, iS, jE, jW, dN, dS, dW, dE);
  if (showJSON)
    tg.json(std::cout);

  return errors;
}
