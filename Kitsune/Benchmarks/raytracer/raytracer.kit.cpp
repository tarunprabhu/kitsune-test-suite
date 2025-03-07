// Raytracer. The output image should be a rendering of the letters LANL.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <kitsune.h>

#include "fpcmp.h"
#include "timing.h"

struct Vec {
  float x, y, z;
  __attribute__((always_inline)) Vec(float v = 0) { x = y = z = v; }
  __attribute__((always_inline)) Vec(float a, float b, float c = 0.0f) {
    x = a;
    y = b;
    z = c;
  }
  __attribute__((always_inline)) Vec operator+(const Vec r) const {
    return Vec(x + r.x, y + r.y, z + r.z);
  }
  __attribute__((always_inline)) Vec operator*(const Vec r) const {
    return Vec(x * r.x, y * r.y, z * r.z);
  }
  __attribute__((always_inline)) float operator%(const Vec r) const {
    return x * r.x + y * r.y + z * r.z;
  }
  __attribute__((always_inline)) Vec operator!() {
    return *this * (1.0 / sqrtf(*this % *this));
  }
};

#include "raytracer.inc"

inline __attribute__((always_inline)) static float randomVal(unsigned int &x) {
  x = (214013 * x + 2531011);
  return ((x >> 16) & 0x7FFF) / 66635.0f;
}

// Rectangle CSG equation. Returns minimum signed distance from
// space carved bylowerLeft vertex and opposite rectangle vertex
// upperRight.
inline __attribute__((always_inline)) static float
boxTest(const Vec &position, Vec lowerLeft, Vec upperRight) {
  lowerLeft = position + lowerLeft * -1.0f;
  upperRight = upperRight + position * -1.0f;
  return -fminf(
      fminf(fminf(lowerLeft.x, upperRight.x), fminf(lowerLeft.y, upperRight.y)),
      fminf(lowerLeft.z, upperRight.z));
}

// Sample the world using Signed Distance Fields.
inline __attribute__((always_inline)) static float
queryDatabase(const Vec &position, int &hitType) {
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
inline __attribute__((always_inline)) int rayMarching(const Vec &origin,
                                                      const Vec &direction,
                                                      Vec &hitPos,
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

inline __attribute__((always_inline)) Vec trace(Vec origin, Vec direction,
                                                unsigned int &rn) {
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

int main(int argc, char **argv) {
  unsigned sampleCount;
  unsigned imageWidth;
  unsigned imageHeight;
  std::string imgFile;
  std::string outFile;
  std::string cpuRefFile;
  std::string gpuRefFile;
  kitsune::mobile_ptr<Pixel> img;
  kitsune::mobile_ptr<Vec> rawImg;

  parseCommandLineInto(argc, argv, sampleCount, imageWidth, imageHeight,
                       imgFile, outFile, cpuRefFile, gpuRefFile);

  TimerGroup tg("raytracer");
  Timer &total = tg.add("total", "Total");

  header("forall", img, rawImg, sampleCount, imageWidth, imageHeight);

  total.start();
  forall(unsigned int i = 0; i < imageWidth * imageHeight; ++i) {
    int x = i % imageWidth;
    int y = i / imageWidth;

    const Vec position(-12.0f, 5.0f, 25.0f);
    const Vec goal = !(Vec(-3.0f, 4.0f, 0.0f) + position * -1.0f);
    const Vec left = !Vec(goal.z, 0, -goal.x) * (1.0f / imageWidth);
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
                              left * ((xf - imageWidth / 2.0f) + randomVal(v)) +
                              up * ((yf - imageHeight / 2.0f) + randomVal(v))),
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
  total.stop();

  int mismatch = footer(tg, img, rawImg, imageWidth, imageHeight, imgFile,
                        outFile, cpuRefFile, gpuRefFile);
  return mismatch;
}
