#ifndef EULER3D_COMMON_H
#define EULER3D_COMMON_H

// This contains types and constants that are to be shared. The sharing is only
// relevant for the multi-source euler3d benchmark where the code has been split
// into multiple files in order to test LTO.

struct Float3 {
  float x, y, z;
};

#define block_length 1

/*
 * Options
 *
 */
#define GAMMA 1.4
#define NDIM 3
#define NNB 4
#define RK 3 // 3rd order RK
#define ff_mach 1.2
#define deg_angle_of_attack 0.0f

/*
 * not options
 */
#define VAR_DENSITY 0
#define VAR_MOMENTUM 1
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM + NDIM)
#define NVAR (VAR_DENSITY_ENERGY + 1)

#endif // EULER3D_COMMON_H
