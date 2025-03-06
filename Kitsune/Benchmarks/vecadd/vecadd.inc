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

static const char *usage = "vecadd [OPTIONS] [n] [iterations]\n"
                           "Try `vecadd -h` for more information";

// clang-format off
static const char *help =
  "Simple vector addition benchmark\n"
  "\n"
  "    vecadd [OPTIONS] [n] [iters]\n"
  "\n"
  "OPTIONS\n"
  "\n"
  "    -h       Print help and exit\n"
  "    -j       Print the recorded times in JSON\n"
  "    -t <N>   Number of threads per block on the GPU. This is only relevant\n"
  "             for the .cu and .hip implementations of this benchmark [256]\n"
  "\n"
  "ARGUMENTS\n"
  "\n"
  "    n        Elements in the arrays [268435456]\n"
  "    iters    Number of iterations   [10]\n";
// clang-format on

// Show the times in JSON. Yes, it's global. Too bad.
static bool showJSON = false;

static void parseCommandLineInto(int argc, char *argv[], size_t &n,
                                 unsigned &iterations,
                                 unsigned *tpb = nullptr) {
  n = 1024 * 1024 * 256;
  iterations = 10;
  if (tpb) {
    // This is loosely for consistency with the launch parameters from kitsune.
    *tpb = 256;
  }

  char flag;
  while ((flag = getopt(argc, argv, "hjt:")) != -1) {
    switch (flag) {
    case 'h':
      std::cout << help << std::endl;
      exit(0);
    case 'j':
      showJSON = true;
      break;
    case 't':
      if (tpb) {
        *tpb = std::stoi(optarg);
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

  char **args = &argv[optind];
  int argn = argc - optind;
  if (argn == 0)
    return;
  if (argn > 0)
    n = std::stol(args[0]);
  if (argn > 1)
    iterations = std::stoi(args[1]);
  if (argn > 2) {
    std::cout << usage << std::endl;
    exit(1);
  }
}

template <typename Arr> static void randomFill(Arr &arr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    ElementType r1 = rand();
    ElementType r2 = rand();
    arr[i] = (r1 / float(RAND_MAX)) * r2;
  }
}

template <typename Arr>
static size_t check(const Arr &a, const Arr &b, const Arr &c, size_t n) {
  std::cout << "\n  Checking final result..." << std::flush;

  size_t errors = 0;
  for (size_t i = 0; i < n; i++)
    if (c[i] != a[i] + b[i])
      errors++;

  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  return errors;
}

template <typename Arr> static void alloc(Arr &a, Arr &b, Arr &c, size_t n) {
#if defined(IS_CUDA)
  cudaMallocManaged(&a, n * sizeof(ElementType));
  cudaMallocManaged(&b, n * sizeof(ElementType));
  cudaMallocManaged(&c, n * sizeof(ElementType));
#elif defined(IS_HIP)
  hipMallocManaged(&a, n * sizeof(ElementType));
  hipMallocManaged(&b, n * sizeof(ElementType));
  hipMallocManaged(&c, n * sizeof(ElementType));
#else
  a.alloc(n);
  b.alloc(n);
  c.alloc(n);
#endif // CPU
}

template <typename Arr>
static void header(const std::string &label, Arr &a, Arr &b, Arr &c, size_t n) {
  srand(7);
  std::cout << "\n";
  std::cout << "---- vector addition benchmark (" << label << ") ----\n"
            << "  Vector size: " << n << " elements.\n\n";
  std::cout << "  Allocating arrays and filling with random values..."
            << std::flush;

  alloc(a, b, c, n);
  randomFill(a, n);
  randomFill(b, n);

  std::cout << "  done.\n\n";
  std::cout << "  Starting benchmark...\n\n";
}

template <typename Arr> static void dealloc(Arr &a, Arr &b, Arr &c) {
#if defined(sIS_CUDA)
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
#elif defined(IS_HIP)
  hipFree(a);
  hipFree(b);
  hipFree(c);
#else  // CPU
  a.free();
  b.free();
  c.free();
#endif // CPU
}

template <typename Arr>
static size_t footer(TimerGroup &tg, Arr &a, Arr &b, Arr &c, size_t n) {
  size_t errors = check(a, b, c, n);
  dealloc(a, b, c);
  if (showJSON)
    tg.json(std::cout);

  return errors;
}
