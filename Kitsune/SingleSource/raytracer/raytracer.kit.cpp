// Raytracer. The output image should be a rendering of the letters LANL.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <unistd.h>

#include <kitsune.h>

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
  "    -c <FILE>  Path to the reference output file\n"
  "    -h         Print help and exit\n"
  "\n"
  "ARGUMENTS\n"
  "\n"
  "    samples  Samples per pixel [8]\n"
  "    width    Image width       [160]\n"
  "    height   Image height      [100]\n";
// clang-format on

struct Pixel {
  unsigned char r, g, b;
};

struct Vec {
  float x, y, z;

  [[clang::always_inline]]
  Vec(float v = 0) {
    x = y = z = v;
  }

  [[clang::always_inline]] Vec(float a, float b, float c = 0.0f) {
    x = a;
    y = b;
    z = c;
  }

  [[clang::always_inline]] Vec operator+(const Vec r) const {
    return Vec(x + r.x, y + r.y, z + r.z);
  }

  [[clang::always_inline]] Vec operator*(const Vec r) const {
    return Vec(x * r.x, y * r.y, z * r.z);
  }

  [[clang::always_inline]] float operator%(const Vec r) const {
    return x * r.x + y * r.y + z * r.z;
  }

  [[clang::always_inline]] Vec operator!() {
    return *this * (1.0 / sqrtf(*this % *this));
  }
};

static float relErr(float actual, float expected) {
  return std::abs(actual - expected) / (std::abs(expected) + 1);
}

static bool checkRelErr(float actual, float expected, float epsilon) {
  return relErr(actual, expected) > epsilon;
}

[[clang::noinline]]
static std::string getFileName(char *argv[], const std::string &ext) {
  std::stringstream ss;
  ss << std::filesystem::path(argv[0]).filename().string() << "." << ext;
  return ss.str();
}

[[clang::noinline]]
static void setup(Pixel *[[kitsune::mobile]] & img,
                  Vec *[[kitsune::mobile]] & rawImg, unsigned width,
                  unsigned height) {
  unsigned size = width * height;

  img = (Pixel *[[kitsune::mobile]])kitsune_mobile_alloc(size * sizeof(Pixel));
  rawImg = (Vec *[[kitsune::mobile]])kitsune_mobile_alloc(size * sizeof(Vec));
}

[[clang::noinline]]
static void teardown(Pixel *[[kitsune::mobile]] img,
                     Vec *[[kitsune::mobile]] rawImg) {
  kitsune_mobile_free(img);
  kitsune_mobile_free(rawImg);
}

[[clang::noinline]]
static void saveImage(const Pixel *[[kitsune::mobile]] img, unsigned width,
                      unsigned height, const std::string &imgFile) {

  if (FILE *fp = fopen(imgFile.c_str(), "wb")) {
    fprintf(fp, "P6 %d %d 255 ", width, height);
    for (int i = width * height - 1; i >= 0; --i)
      fwrite((void *)&img[i], sizeof(unsigned char), 3, fp);
    fclose(fp);
  }
}

[[clang::noinline]]
static void saveRaw(const Vec *[[kitsune::mobile]] rawImg, unsigned width,
                    unsigned height, const std::string &outFile) {
  FILE *fp = fopen(outFile.c_str(), "wb");
  fwrite(&width, sizeof(unsigned), 1, fp);
  fwrite(&height, sizeof(unsigned), 1, fp);
  fwrite((void *)rawImg, sizeof(Vec), width * height, fp);
  fclose(fp);
}

[[clang::noinline]]
static void save(const Pixel *[[kitsune::mobile]] img,
                 const Vec *[[kitsune::mobile]] rawImg, unsigned width,
                 unsigned height, const std::string &imgFile,
                 const std::string &outFile) {
  saveImage(img, width, height, imgFile);
  saveRaw(rawImg, width, height, outFile);
}

// FIXME: This epsilon is larger than I would like, but it is not clear how we
// can make this smaller. This code is a nightmare to work with.
static constexpr float epsilon = 1;

// If there is a mismatch in the output, return the number of mismatches. If the
// size of the image does not match, return -1. If the output matches, return 0.
[[clang::noinline]]
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
        checkRelErr(av[i].z, ev[i].z, epsilon)) {
      printf("%d: (%.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f)\n", i, ev[i].x,
             ev[i].y, ev[i].z, av[i].x, av[i].y, av[i].z);
      ++errors;
    }
  }

cleanup:
  free(ev);
  free(av);
  fclose(fe);
  fclose(fa);

  return errors;
}

[[clang::noinline]]
static int report(int mismatch) {
  if (mismatch == -2)
    printf("FAIL! (File size mismatch)\n");
  else if (mismatch == -1)
    printf("FAIL! (Image size mismatch)\n");
  else if (mismatch)
    printf("FAIL! (%d %s)\n", mismatch,
           mismatch == 1 ? "mismatch" : "mismatches");
  else
    printf("PASS\n");
  return mismatch ? 1 : 0;
}

#define HIT_NONE 0
#define HIT_LETTER 1
#define HIT_WALL 2
#define HIT_SUN 3

static float randomVal(unsigned int &x) {
  x = (214013 * x + 2531011);
  return ((x >> 16) & 0x7FFF) / 66635.0f;
}

// Rectangle CSG equation. Returns minimum signed distance from
// space carved bylowerLeft vertex and opposite rectangle vertex
// upperRight.
static float boxTest(const Vec &position, Vec lowerLeft, Vec upperRight) {
  lowerLeft = position + lowerLeft * -1.0f;
  upperRight = upperRight + position * -1.0f;
  return -fminf(
      fminf(fminf(lowerLeft.x, upperRight.x), fminf(lowerLeft.y, upperRight.y)),
      fminf(lowerLeft.z, upperRight.z));
}

// Sample the world using Signed Distance Fields.
static float queryDatabase(const Vec &position, int &hitType) {
  float distance = 1e9; // FLT_MAX;
  Vec f = position;     // Flattened position (z=0)
  f.z = 0;
  const float lines[10 * 4] = {
      -20.0f, 0.0f,  -20.0f, 16.0f, -20.0f, 0.0f,  -14.0f, 0.0f,  -11.0f, 0.0f,
      -7.0f,  16.0f, -3.0f,  0.0f,  -7.0f,  16.0f, -5.5f,  5.0f,  -9.5f,  5.0f,
      0.0f,   0.0f,  0.0f,   16.0f, 6.0f,   0.0f,  6.0f,   16.0f, 0.0f,   16.0f,
      6.0f,   0.0f,  9.0f,   0.0f,  9.0f,   16.0f, 9.0f,   0.0f,  15.0f,  0.0f};

  for (unsigned i = 0; i < sizeof(lines) / sizeof(float); i += sizeof(float)) {
    Vec begin = Vec(lines[i], lines[i + 1]) * 0.5f;
    Vec e = Vec(lines[i + 2], lines[i + 3]) * 0.5f + begin * -1.0f;
    Vec o = f + (begin +
                 e * fminf(-fminf((((begin + f * -1) % e) / (e % e)), 0), 1)) *
                    -1.0f;
    distance = fminf(distance, o % o); // compare squared distance.
  }

  distance = sqrtf(distance); // Get real distance, not square distance.
  distance = powf(powf(distance, 8.0f) + powf(position.z, 8.0f), 0.125f) - 0.5f;
  hitType = HIT_LETTER;

  float roomDist;
  roomDist =
      fminf(-fminf(boxTest(position, Vec(-30.0f, -0.5f, -30.0f),
                           Vec(30.0f, 18.0f, 30.0f)),
                   boxTest(position, Vec(-25.0f, 17.0f, -25.0f),
                           Vec(25.0f, 20.0f, 25.0f))),
            boxTest( // Ceiling "planks" spaced 8 units apart.
                Vec(fmodf(fabsf(position.x), 8.0f), position.y, position.z),
                Vec(1.5f, 18.5f, -25.0f), Vec(6.5f, 20.0f, 25.0f)));
  if (roomDist < distance) {
    distance = roomDist;
    hitType = HIT_WALL;
  }
  float sun = 19.9f - position.y; // Everything above 19.9 is light source.
  if (sun < distance) {
    distance = sun;
    hitType = HIT_SUN;
  }
  return distance;
}

// Perform signed sphere marching
// Returns hitType 0, 1, 2, or 3 and update hit position/normal
static int rayMarching(const Vec &origin, const Vec &direction, Vec &hitPos,
                       Vec &hitNorm) {
  int hitType = HIT_NONE;
  int noHitCount = 0;

  // Signed distance marching
  float d; // distance from closest object in world.
  for (float total_d = 0.0f; total_d < 100.0f; total_d += d) {
    d = queryDatabase(hitPos = origin + direction * total_d, hitType);
    if (d < .01f || ++noHitCount > 99) {
      hitNorm = !Vec(
          queryDatabase(hitPos + Vec(0.01f, 0.00f), noHitCount) - d,
          queryDatabase(hitPos + Vec(0.00f, 0.01f), noHitCount) - d,
          queryDatabase(hitPos + Vec(0.00f, 0.00f, 0.01f), noHitCount) - d);
      return hitType;
    }
  }
  return HIT_NONE;
}

static Vec trace(Vec origin, Vec direction, unsigned int &rn) {
  Vec sampledPosition;
  Vec normal;
  Vec color(0.0f, 0.0f, 0.0f);
  Vec attenuation(1.0f);
  Vec lightDirection(!Vec(0.6f, 0.6f, 1.0f)); // Directional light

  for (int bounceCount = 8; bounceCount--;) {
    int hitType = rayMarching(origin, direction, sampledPosition, normal);
    if (hitType == HIT_NONE)
      break; // No hit, return color.
    else if (hitType == HIT_LETTER) {
      // Specular bounce on a letter. No color acc.
      direction = direction + normal * (normal % direction * -2.0f);
      origin = sampledPosition + direction * 0.1f;
      attenuation = attenuation * 0.2f; // Attenuation via distance traveled.
    } else if (hitType == HIT_WALL) {   // Wall hit uses color yellow?
      float incidence = normal % lightDirection;
      float p = 6.283185f * randomVal(rn);
      float c = randomVal(rn);
      float s = sqrtf(1.0f - c);
      float g = normal.z < 0.0f ? -1.0f : 1.0f;
      float u = (-1.0f / (g + normal.z));
      float v = normal.x * normal.y * u;
      float cosp = cosf(p);
      float sinp = sinf(p);
      // sincosf(p, &sinp, &cosp);
      direction = Vec(v, g + normal.y * normal.y * u, -normal.y) * (cosp * s) +
                  Vec(1 + g * normal.x * normal.x * u, g * v, -g * normal.x) *
                      (sinp * s) +
                  normal * sqrtf(c);
      origin = sampledPosition + direction * 0.1f;
      attenuation = attenuation * 0.2f;

      if (incidence > 0.0f &&
          rayMarching(sampledPosition + normal * 0.1f, lightDirection,
                      sampledPosition, normal) == HIT_SUN)
        color = color + attenuation * Vec(500, 400, 100) * incidence;
    } else if (hitType == HIT_SUN) { //
      color = color + attenuation * Vec(50, 80, 100);
      break; // Sun Color
    }
  }
  return color;
}

[[clang::noinline]]
static void test(Pixel *[[kitsune::mobile]] img,
                 Vec *[[kitsune::mobile]] rawImg, unsigned sampleCount,
                 unsigned width, unsigned height) {
  forall(unsigned int i = 0; i < width * height; ++i) {
    int x = i % width;
    int y = i / width;

    const Vec position(-12.0f, 5.0f, 25.0f);
    const Vec goal = !(Vec(-3.0f, 4.0f, 0.0f) + position * -1.0f);
    const Vec left = !Vec(goal.z, 0, -goal.x) * (1.0f / width);

    // Cross-product to get the up vector
    const Vec up(goal.y * left.z - goal.z * left.y,
                 goal.z * left.x - goal.x * left.z,
                 goal.x * left.y - goal.y * left.x);
    Vec color;
    for (unsigned int p = sampleCount, v = i; p--;) {
      Vec rand_left = Vec(randomVal(v), randomVal(v), randomVal(v)) * .001;
      float xf = x + randomVal(v);
      float yf = y + randomVal(v);
      color = color + trace(position,
                            !((goal + rand_left) +
                              left * ((xf - width / 2.0f) + randomVal(v)) +
                              up * ((yf - height / 2.0f) + randomVal(v))),
                            v);
    }
    // Reinhard tone mapping
    color = color * (1.0f / sampleCount) + 14.0f / 241.0f;

    // Save the pre-quantized image.
    rawImg[i] = color;

    Vec o = color + 1.0f;
    color = Vec(color.x / o.x, color.y / o.y, color.z / o.z) * 255.0f;
    img[i].r = (unsigned char)color.x;
    img[i].g = (unsigned char)color.y;
    img[i].b = (unsigned char)color.z;
  }
}

int main(int argc, char **argv) {
  Pixel *[[kitsune::mobile]] img = nullptr;
  Vec *[[kitsune::mobile]] rawImg = nullptr;
  unsigned sampleCount;
  unsigned width;
  unsigned height;
  std::string imgFile;
  std::string outFile;
  std::string checkFile;

  sampleCount = 8;
  width = 160;
  height = 100;
  checkFile = "";
  imgFile = getFileName(argv, "ppm");
  outFile = getFileName(argv, "dat");

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
      printf("ERROR: Unknown option: '%c'\n\n", optopt);
      printf("%s\n", usage);
      exit(1);
    }
  }

  char **args = &argv[optind];
  int argn = argc - optind;
  if (argn > 0)
    sampleCount = atoi(args[0]);
  if (argn > 1)
    width = atoi(args[1]);
  if (argn > 2)
    height = atoi(args[2]);
  if (argn > 3) {
    printf("%s\n", usage);
    exit(1);
  }

  setup(img, rawImg, width, height);
  test(img, rawImg, sampleCount, width, height);
  save(img, rawImg, width, height, imgFile, outFile);
  int mismatch = 0;
  if (checkFile.size())
    mismatch = check(checkFile, outFile);
  teardown(img, rawImg);

  return report(mismatch);
}
