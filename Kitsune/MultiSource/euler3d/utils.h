#ifndef EULER3D_UTILS_H
#define EULER3D_UTILS_H

#include <kitsune.h>

#include "../../Benchmarks/euler3d/common.h"

void cpy(kitsune::mobile_ptr<float> dst, kitsune::mobile_ptr<float> src, int n);

void compute_flux_contribution(const float density, const Float3 &momentum,
                               const float density_energy, const float pressure,
                               Float3 &velocity, Float3 &fc_momentum_x,
                               Float3 &fc_momentum_y, Float3 &fc_momentum_z,
                               Float3 &fc_density_energy);

[[kitsune::device]]
void compute_velocity(float density, const Float3 &momentum, Float3 &velocity);

[[kitsune::device]]
float compute_speed_sqd(const Float3 &velocity);

[[kitsune::device]]
float compute_pressure(float density, float density_energy, float speed_sqd);

[[kitsune::device]]
float compute_speed_of_sound(float density, float pressure);

#endif // EULER3D_UTILS_H
