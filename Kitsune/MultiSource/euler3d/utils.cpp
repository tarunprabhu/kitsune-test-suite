#include <cmath>
#include <kitsune.h>

#include "../../Benchmarks/euler3d/common.h"

void cpy(kitsune::mobile_ptr<float> dst, kitsune::mobile_ptr<float> src,
         int n) {
  // clang-format off
  forall(unsigned int i = 0; i < n; i++) {
    dst[i] = src[i];
  }
  // clang-format on
}

[[kitsune::device]]
void compute_flux_contribution(const float density, const Float3 &momentum,
                               const float density_energy, const float pressure,
                               Float3 &velocity, Float3 &fc_momentum_x,
                               Float3 &fc_momentum_y, Float3 &fc_momentum_z,
                               Float3 &fc_density_energy) {
  fc_momentum_x.x = velocity.x * momentum.x + pressure;
  fc_momentum_x.y = velocity.x * momentum.y;
  fc_momentum_x.z = velocity.x * momentum.z;

  fc_momentum_y.x = fc_momentum_x.y;
  fc_momentum_y.y = velocity.y * momentum.y + pressure;
  fc_momentum_y.z = velocity.y * momentum.z;

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
  fc_momentum_z.z = velocity.z * momentum.z + pressure;

  float de_p = density_energy + pressure;
  fc_density_energy.x = velocity.x * de_p;
  fc_density_energy.y = velocity.y * de_p;
  fc_density_energy.z = velocity.z * de_p;
}

[[kitsune::device]]
void compute_velocity(float density, const Float3 &momentum, Float3 &velocity) {
  velocity.x = momentum.x / density;
  velocity.y = momentum.y / density;
  velocity.z = momentum.z / density;
}

[[kitsune::device]]
float compute_speed_sqd(const Float3 &velocity) {
  return velocity.x * velocity.x + velocity.y * velocity.y +
         velocity.z * velocity.z;
}

[[kitsune::device]]
float compute_pressure(float density, float density_energy, float speed_sqd) {
  return (float(GAMMA) - float(1.0f)) *
         (density_energy - float(0.5f) * density * speed_sqd);
}

[[kitsune::device]]
float compute_speed_of_sound(float density, float pressure) {
  return sqrtf(float(GAMMA) * pressure / density);
}
