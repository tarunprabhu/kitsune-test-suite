#ifndef EULER3D_UTILS_H
#define EULER3D_UTILS_H

#include <kitsune.h>

#include "../../SingleSource/euler3d/euler3d.h"

void cpy(float *[[kitsune::mobile]] dst, const float *[[kitsune::mobile]] src,
         int n);

void compute_flux_contribution(const float density, const Float3 &momentum,
                               const float density_energy, const float pressure,
                               Float3 &velocity, Float3 &fc_momentum_x,
                               Float3 &fc_momentum_y, Float3 &fc_momentum_z,
                               Float3 &fc_density_energy);

void compute_velocity(float density, const Float3 &momentum, Float3 &velocity);

float compute_speed_sqd(const Float3 &velocity);

float compute_pressure(float density, float density_energy, float speed_sqd);

float compute_speed_of_sound(float density, float pressure);

#endif // EULER3D_UTILS_H
