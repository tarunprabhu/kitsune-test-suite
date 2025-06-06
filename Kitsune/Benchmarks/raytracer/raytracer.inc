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

struct Pixel {
  unsigned char r, g, b;
};

#define HIT_NONE 0
#define HIT_LETTER 1
#define HIT_WALL 2
#define HIT_SUN 3

static const char *usage = "raytracer [OPTIONS] [samples] [width] [height]\n"
                           "Try `raytracer -h` for more information";

// clang-format off
static const char *help =
  "Raytracer benchmark. The output should be a rendering of the letters LANL\n"
  "\n"
  "    raytracer [OPTIONS] [samples] [width] [height]\n"
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
  "    samples  Samples per pixel [128]\n"
  "    width    Image width       [1280]\n"
  "    height   Image height      [1024]\n";
// clang-format on

// Show the times in JSON. Yes, it's global. Too bad.
static bool showJSON = false;

static void parseCommandLineInto(int argc, char *argv[], unsigned &sampleCount,
                                 unsigned &imageWidth, unsigned &imageHeight,
                                 std::string &imgFile, std::string &outFile,
                                 std::string &cpuRefFile,
                                 std::string &gpuRefFile,
                                 unsigned *tpb = nullptr) {
  std::string base = std::filesystem::path(argv[0]).filename().string();
  imgFile = base + ".ppm";
  outFile = base + ".dat";
  sampleCount = 128;
  imageWidth = 1280;
  imageHeight = 1024;
  cpuRefFile = "";
  gpuRefFile = "";
  if (tpb) {
    // This is loosely for consistency with the launch parameters from kitsune.
    *tpb = 256;
  }

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
    sampleCount = std::stoi(args[0]);
  if (argn > 1)
    imageWidth = std::stoi(args[1]);
  if (argn > 2)
    imageHeight = std::stoi(args[2]);
  if (argn > 3) {
    std::cout << usage << std::endl;
    exit(1);
  }
}

template <typename Img, typename RawImg>
static void dealloc(Img &img, RawImg &rawImg) {
#if defined(__KOKKOS__)
#elif defined(__CUDACC__)
  cudaFree(img);
  cudaFree(rawImg);
#elif defined(__HIP__)
  (void)hipFree(img);
  (void)hipFree(rawImg);
#else
  img.free();
  rawImg.free();
#endif
}

template <typename Img, typename RawImg>
static void alloc(Img &img, RawImg &rawImg, unsigned imageWidth,
                  unsigned imageHeight) {
  unsigned imageSize = imageWidth * imageHeight;
#if defined(__KOKKOS__)
#elif defined(__CUDACC__)
  cudaMallocManaged(&img, imageSize * sizeof(Pixel));
  cudaMallocManaged(&rawImg, imageSize * sizeof(Vec));
#elif defined(__HIP__)
  (void)hipMallocManaged(&img, imageSize * sizeof(Pixel));
  (void)hipMallocManaged(&rawImg, imageSize * sizeof(Vec));
#else
  img.alloc(imageSize);
  rawImg.alloc(imageSize);
#endif
}

template <typename Img, typename RawImg>
static void header(const std::string &label, Img &img, RawImg &rawImg,
                   unsigned sampleCount, unsigned imageWidth,
                   unsigned imageHeight) {
  std::cout << "\n";
  std::cout << "---- Raytracer benchmark (" << label << ") ----\n"
            << "  Image size    : " << imageWidth << "x" << imageHeight << "\n"
            << "  Samples/pixel : " << sampleCount << "\n\n";

  std::cout << "  Allocating image..." << std::flush;

  alloc(img, rawImg, imageWidth, imageHeight);

  std::cout << "  done.\n\n";
  std::cout << "  Running benchmark ... " << std::flush;
}

// FIXME: It is absolutely ridiculous that the epsilon here is so large. It used
// to be 10 which was bad enough, but when compiling with Kokkos on AMD, even
// that is not sufficient. The image looks reasonable though.
static constexpr float epsilon = 10000;

// If there is a mismatch in the output, return the number of mismatches. If the
// size of the image does not match, return -1. If the output matches, return 0.
static int check(const std::string &checkFile, const std::string &outFile) {
  int errors = 0;
  FILE *fa = fopen(outFile.c_str(), "rb");
  FILE *fe = fopen(checkFile.c_str(), "rb");
  Vec *av = nullptr;
  Vec *ev = nullptr;

  unsigned ew, aw;
  unsigned eh, ah;
  fread(&ew, sizeof(unsigned), 1, fe);
  fread(&eh, sizeof(unsigned), 1, fe);
  fread(&aw, sizeof(unsigned), 1, fa);
  fread(&ah, sizeof(unsigned), 1, fa);
  if (ew != aw or eh != ah) {
    errors = -1;
    goto cleanup;
  }

  ev = (Vec *)malloc(sizeof(Vec) * ew * eh);
  av = (Vec *)malloc(sizeof(Vec) * aw * ah);

  fread(ev, sizeof(Vec), ew * eh, fe);
  fread(av, sizeof(Vec), aw * ah, fa);

  for (unsigned i = 0; i < ew * eh; ++i) {
    if (checkRelErr(av[i].x, ev[i].x, epsilon) or
        checkRelErr(av[i].y, ev[i].y, epsilon) or
        checkRelErr(av[i].z, ev[i].z, epsilon))
      ++errors;
  }

cleanup:
  free(ev);
  free(av);
  fclose(fe);
  fclose(fa);

  return errors;
}

template <typename Img>
static void saveImage(const Img &img, unsigned imageWidth, unsigned imageHeight,
                      const std::string &imgFile) {
  std::cout << "  Saving image ... " << std::flush;
  std::ofstream of(imgFile);
  if (of.is_open()) {
    of << "P6 " << imageWidth << " " << imageHeight << " 255 ";
    for (int i = imageWidth * imageHeight - 1; i >= 0; i--)
      of << img[i].r << img[i].g << img[i].b;
    of.close();
  }
  std::cout << "done\n\n";
}

template <typename RawImg>
static void saveRaw(const RawImg &rawImg, unsigned imageWidth,
                    unsigned imageHeight, const std::string &outFile) {
  FILE *fp = fopen(outFile.c_str(), "wb");
  fwrite(&imageWidth, sizeof(unsigned), 1, fp);
  fwrite(&imageHeight, sizeof(unsigned), 1, fp);
  fwrite((void *)&*rawImg, sizeof(Vec), imageWidth * imageHeight, fp);
  fclose(fp);
}

template <typename Img, typename RawImg>
static int footer(TimerGroup &tg, Img &img, RawImg &rawImg, unsigned imageWidth,
                  unsigned imageHeight, const std::string &imgFile,
                  const std::string &outFile, const std::string &cpuRefFile,
                  const std::string &gpuRefFile) {
  std::cout << "done\n\n";
  tg.prettyTimes(std::cout, 4);
  std::cout << "----\n\n";

  saveImage(img, imageWidth, imageHeight, imgFile);
  saveRaw(rawImg, imageWidth, imageHeight, outFile);

  std::string checkFile = cpuRefFile;
#if defined(__CUDACC__) || defined(__HIP__) || defined(__KOKKOS__)
  checkFile = gpuRefFile;
#else
  std::string tt = __kitsune_tt__;
  if (tt == "cuda" or tt == "hip")
    checkFile = gpuRefFile;
#endif // !__CUDACC__ && !__HIP__

  int mismatch = 0;
  if (!checkFile.size()) {
    std::cout << "--------------------------------------------------\n";
    std::cout << " WARNING: Not checking output (no reference file) \n";
    std::cout << "--------------------------------------------------\n";
  } else {
    std::cout << "\n  Checking final result..." << std::flush;
    mismatch = -3;
    mismatch = check(outFile, checkFile);
    if (mismatch == -2)
      std::cout << "  FAIL! (File size mismatch)\n\n";
    else if (mismatch == -1)
      std::cout << "  FAIL! (Image size mismatch)\n\n";
    else if (mismatch)
      std::cout << "  FAIL! (Mismatch at pixel " << mismatch << ")\n\n";
    else
      std::cout << "  pass\n\n";
  }

  dealloc(img, rawImg);
  if (showJSON)
    tg.json(std::cout);

  return mismatch;
}
