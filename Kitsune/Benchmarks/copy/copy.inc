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

static const char *usage = "copy [OPTIONS] [n] [iterations]\n"
                           "Try `copy -h` for more information";

// clang-format off
static const char *help =
  "Simple memory copy benchmark\n"
  "\n"
  "    copy [OPTIONS] [n] [iters]\n"
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

  int flag;
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

using ElementType = float;

template <typename Arr> static void randomFill(Arr &arr, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    ElementType r1 = rand();
    ElementType r2 = rand();
    arr[i] = (r1 / ElementType(RAND_MAX)) * r2;
  }
}

template <typename Arr>
static size_t check(const Arr &dst, const Arr &src, size_t n) {
  size_t errors = 0;
  for (size_t i = 0; i < n; i++)
    if (dst[i] != src[i])
      errors++;
  return errors;
}

template <typename Arr> static void alloc(Arr &dst, Arr &src, size_t n) {
#if defined(__KOKKOS__)
#elif defined(__CUDACC__)
  cudaMallocManaged(&dst, n * sizeof(ElementType));
  cudaMallocManaged(&src, n * sizeof(ElementType));
#elif defined(__HIP__)
  (void)hipMallocManaged(&dst, n * sizeof(ElementType));
  (void)hipMallocManaged(&src, n * sizeof(ElementType));
#else
  dst.alloc(n);
  src.alloc(n);
#endif // CPU
}

template <typename Arr>
static void header(const std::string &label, Arr &dst, Arr &src, size_t n) {
  srand(7);
  std::cout << "\n";
  std::cout << "---- copy benchmark (" << label << ")" << "\n"
            << "  Array size: " << n << " elements.\n\n";
  std::cout << "  Allocating arrays and filling with random values..."
            << std::flush;

  alloc(dst, src, n);
  randomFill(src, n);

  std::cout << "  done.\n\n";
  std::cout << "  Starting benchmark...\n\n";
}

template <typename Arr> static void dealloc(Arr &dst, Arr &src) {
#if defined(__KOKKOS__)
#elif defined(__CUDACC__)
  cudaFree(dst);
  cudaFree(src);
#elif defined(__HIP__)
  (void)hipFree(dst);
  (void)hipFree(src);
#else  // CPU
  dst.free();
  src.free();
#endif // CPU
}

template <typename Arr>
static size_t footer(TimerGroup &tg, Arr &dst, Arr &src, size_t n) {
  std::cout << "\n  Checking with epsilon = 0 ..." << std::flush;
  size_t errors = check(dst, src, n);
  if (errors)
    std::cout << "  FAIL! (" << errors << " errors found)\n\n";
  else
    std::cout << "  pass\n\n";

  dealloc(dst, src);
  if (showJSON)
    tg.json(std::cout);

  return errors;
}
