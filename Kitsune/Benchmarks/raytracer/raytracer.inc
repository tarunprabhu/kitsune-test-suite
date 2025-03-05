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
  std::string base = fs::path(argv[0]).filename().string();
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

  char flag;
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
static void alloc(Img &img, RawImg &rawImg, unsigned imageWidth,
                  unsigned imageHeight) {
  unsigned imageSize = imageWidth * imageHeight;
#if defined(IS_CUDA)
  cudaMallocManaged(&img, imageSize * sizeof(Pixel));
  cudaMallocManaged(&rawImg, imageSize * sizeof(Vec));
#elif defined(IS_HIP)
  hipMallocManaged(&img, imageSize * sizeof(Pixel));
  hipMallocManaged(&rawImg, imageSize * sizeof(Vec));
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

template <typename Img, typename RawImg>
static void dealloc(Img &img, RawImg &rawImg) {
#if defined(IS_CUDA)
  cudaFree(img);
  cudaFree(rawImg);
#elif defined(IS_HIP)
  hipFree(img);
  hipFree(rawImg);
#else
  img.free();
  rawImg.free();
#endif
}

// If there is a mismatch in the output, return the pixel at which the output
// did not match. If the size of the image does not match, return -1. If the
// output matches, return 0.
static int check(const std::string &checkFile, const std::string &outFile) {
  const float epsilon = 10;

  FILE *fa = fopen(outFile.c_str(), "rb");
  FILE *fe = fopen(checkFile.c_str(), "rb");

  unsigned ew, aw;
  unsigned eh, ah;
  fread(&ew, sizeof(unsigned), 1, fe);
  fread(&eh, sizeof(unsigned), 1, fe);
  fread(&aw, sizeof(unsigned), 1, fa);
  fread(&ah, sizeof(unsigned), 1, fa);
  if (ew != aw or eh != ah)
    return -1;

  Vec *ev = (Vec *)malloc(sizeof(Vec) * ew * eh);
  Vec *av = (Vec *)malloc(sizeof(Vec) * aw * ah);

  size_t re = fread(ev, sizeof(Vec), ew * eh, fe);
  size_t ra = fread(av, sizeof(Vec), aw * ah, fa);
  if (re != ra)
    return -2;

  for (unsigned i = 0; i < ew * eh; ++i) {
    if (fabs(ev[i].x - av[i].x) > epsilon or
        fabs(ev[i].y - av[i].y) > epsilon or
        fabs(ev[i].z - av[i].z) > epsilon) {
      return i + 1;
    }
  }

  free(ev);
  free(av);
  fclose(fe);
  fclose(fa);

  return 0;
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
#if defined(IS_CUDA) || defined(IS_HIP)
  checkFile = gpuRefFile;
#else
  std::string tt = __kitsune_tt__;
  if (tt == "cuda" or tt == "hip")
    checkFile = gpuRefFile;
#endif // !IS_CUDA && !IS_HIP

  int mismatch = -3;
  if (!checkFile.size()) {
    std::cout << "--------------------------------------------------\n";
    std::cout << " WARNING: Not checking output (no reference file) \n";
    std::cout << "--------------------------------------------------\n";
  } else {
    std::cout << "\n  Checking final result..." << std::flush;
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
