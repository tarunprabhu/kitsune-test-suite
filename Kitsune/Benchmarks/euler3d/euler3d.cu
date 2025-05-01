/// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "fpcmp.h"
#include "timing.h"

#include "euler3d.inc"

__global__ void cpy(float *dst, const float *src, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i];
}

__global__ void initialize_variables(int nelr, float *variables,
                                     float *ff_variable) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < nelr) {
    for (int j = 0; j < NVAR; j++)
      variables[i + j * nelr] = ff_variable[j];
  }
}

__host__ __device__ void compute_flux_contribution(
    const float density, const Float3 &momentum, const float density_energy,
    const float pressure, Float3 &velocity, Float3 &fc_momentum_x,
    Float3 &fc_momentum_y, Float3 &fc_momentum_z, Float3 &fc_density_energy) {
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

__device__ void compute_velocity(float density, const Float3 &momentum,
                                 Float3 &velocity) {
  velocity.x = momentum.x / density;
  velocity.y = momentum.y / density;
  velocity.z = momentum.z / density;
}

__device__ float compute_speed_sqd(const Float3 &velocity) {
  return velocity.x * velocity.x + velocity.y * velocity.y +
         velocity.z * velocity.z;
}

__device__ float compute_pressure(float density, float density_energy,
                                  float speed_sqd) {
  return (float(GAMMA) - float(1.0f)) *
         (density_energy - float(0.5f) * density * speed_sqd);
}

__device__ float compute_speed_of_sound(float density, float pressure) {
  return sqrtf(float(GAMMA) * pressure / density);
}

__global__ void compute_step_factor(int nelr, const float *variables,
                                    const float *areas, float *step_factors) {
  int blk = blockDim.x * blockIdx.x + threadIdx.x;
  if (blk < nelr / block_length) {
    int b_start = blk * block_length;
    int b_end =
        (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;

    for (int i = b_start; i < b_end; i++) {
      float density = variables[i + VAR_DENSITY * nelr];

      Float3 momentum;
      momentum.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
      momentum.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
      momentum.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

      float density_energy = variables[i + VAR_DENSITY_ENERGY * nelr];
      Float3 velocity;
      compute_velocity(density, momentum, velocity);
      float speed_sqd = compute_speed_sqd(velocity);
      float pressure = compute_pressure(density, density_energy, speed_sqd);
      float speed_of_sound = compute_speed_of_sound(density, pressure);

      // dt = float(0.5f) * sqrt(areas[i]) / (||v|| + c).... but
      // when we do time stepping, this later would need to be divided
      // by the area, so we just do it all at once
      step_factors[i] =
          float(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
    }
  }
}

__global__ void compute_flux(int nelr, int *elements_surrounding_elements,
                             float *normals, float *variables, float *fluxes,
                             const float *ff_variable,
                             const Float3 ff_flux_contribution_momentum_x,
                             const Float3 ff_flux_contribution_momentum_y,
                             const Float3 ff_flux_contribution_momentum_z,
                             const Float3 ff_flux_contribution_density_energy) {
  const float smoothing_coefficient = 0.2f;

  int blk = blockDim.x * blockIdx.x + threadIdx.x;
  if (blk < nelr / block_length) {
    unsigned int b_start = blk * block_length;
    unsigned int b_end =
        (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;

    for (unsigned int i = b_start; i < b_end; ++i) {
      float density_i = variables[i + VAR_DENSITY * nelr];
      Float3 momentum_i;
      momentum_i.x = variables[i + (VAR_MOMENTUM + 0) * nelr];
      momentum_i.y = variables[i + (VAR_MOMENTUM + 1) * nelr];
      momentum_i.z = variables[i + (VAR_MOMENTUM + 2) * nelr];

      float density_energy_i = variables[i + VAR_DENSITY_ENERGY * nelr];

      Float3 velocity_i;
      compute_velocity(density_i, momentum_i, velocity_i);
      float speed_sqd_i = compute_speed_sqd(velocity_i);
      float speed_i = sqrtf(speed_sqd_i);
      float pressure_i =
          compute_pressure(density_i, density_energy_i, speed_sqd_i);
      float speed_of_sound_i = compute_speed_of_sound(density_i, pressure_i);
      Float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
          flux_contribution_i_momentum_z;

      Float3 flux_contribution_i_density_energy;
      compute_flux_contribution(
          density_i, momentum_i, density_energy_i, pressure_i, velocity_i,
          flux_contribution_i_momentum_x, flux_contribution_i_momentum_y,
          flux_contribution_i_momentum_z, flux_contribution_i_density_energy);

      float flux_i_density = 0.0f;
      Float3 flux_i_momentum;
      flux_i_momentum.x = 0.0f;
      flux_i_momentum.y = 0.0f;
      flux_i_momentum.z = 0.0f;
      float flux_i_density_energy = 0.0f;

      Float3 velocity_nb;
      float density_nb, density_energy_nb;
      Float3 momentum_nb;
      Float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y,
          flux_contribution_nb_momentum_z;
      Float3 flux_contribution_nb_density_energy;
      float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

      for (int j = 0; j < NNB; j++) {
        Float3 normal;
        float normal_len;
        float factor;

        int nb = elements_surrounding_elements[i + j * nelr];
        normal.x = normals[i + (j + 0 * NNB) * nelr];
        normal.y = normals[i + (j + 1 * NNB) * nelr];
        normal.z = normals[i + (j + 2 * NNB) * nelr];
        normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y +
                           normal.z * normal.z);

        if (nb >= 0) { // a legitimate neighbor
          density_nb = variables[nb + VAR_DENSITY * nelr];
          momentum_nb.x = variables[nb + (VAR_MOMENTUM)*nelr];
          momentum_nb.y = variables[nb + (VAR_MOMENTUM + 1) * nelr];
          momentum_nb.z = variables[nb + (VAR_MOMENTUM + 2) * nelr];
          density_energy_nb = variables[nb + VAR_DENSITY_ENERGY * nelr];
          compute_velocity(density_nb, momentum_nb, velocity_nb);
          speed_sqd_nb = compute_speed_sqd(velocity_nb);
          pressure_nb =
              compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
          speed_of_sound_nb = compute_speed_of_sound(density_nb, pressure_nb);
          compute_flux_contribution(
              density_nb, momentum_nb, density_energy_nb, pressure_nb,
              velocity_nb, flux_contribution_nb_momentum_x,
              flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z,
              flux_contribution_nb_density_energy);

          // artificial viscosity
          factor = -normal_len * smoothing_coefficient * 0.5f *
                   (speed_i + sqrtf(speed_sqd_nb) + speed_of_sound_i +
                    speed_of_sound_nb);
          flux_i_density += factor * (density_i - density_nb);
          flux_i_density_energy +=
              factor * (density_energy_i - density_energy_nb);
          flux_i_momentum.x += factor * (momentum_i.x - momentum_nb.x);
          flux_i_momentum.y += factor * (momentum_i.y - momentum_nb.y);
          flux_i_momentum.z += factor * (momentum_i.z - momentum_nb.z);

          // accumulate cell-centered fluxes
          factor = 0.5f * normal.x;
          flux_i_density += factor * (momentum_nb.x + momentum_i.x);
          flux_i_density_energy +=
              factor * (flux_contribution_nb_density_energy.x +
                        flux_contribution_i_density_energy.x);
          flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.x +
                                         flux_contribution_i_momentum_x.x);
          flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.x +
                                         flux_contribution_i_momentum_y.x);
          flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.x +
                                         flux_contribution_i_momentum_z.x);

          factor = 0.5f * normal.y;
          flux_i_density += factor * (momentum_nb.y + momentum_i.y);
          flux_i_density_energy +=
              factor * (flux_contribution_nb_density_energy.y +
                        flux_contribution_i_density_energy.y);
          flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.y +
                                         flux_contribution_i_momentum_x.y);
          flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.y +
                                         flux_contribution_i_momentum_y.y);
          flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.y +
                                         flux_contribution_i_momentum_z.y);

          factor = 0.5f * normal.z;
          flux_i_density += factor * (momentum_nb.z + momentum_i.z);
          flux_i_density_energy +=
              factor * (flux_contribution_nb_density_energy.z +
                        flux_contribution_i_density_energy.z);
          flux_i_momentum.x += factor * (flux_contribution_nb_momentum_x.z +
                                         flux_contribution_i_momentum_x.z);
          flux_i_momentum.y += factor * (flux_contribution_nb_momentum_y.z +
                                         flux_contribution_i_momentum_y.z);
          flux_i_momentum.z += factor * (flux_contribution_nb_momentum_z.z +
                                         flux_contribution_i_momentum_z.z);
        } else if (nb == -1) { // a wing boundary
          flux_i_momentum.x += normal.x * pressure_i;
          flux_i_momentum.y += normal.y * pressure_i;
          flux_i_momentum.z += normal.z * pressure_i;
        } else if (nb == -2) { // a far field boundary
          factor = 0.5f * normal.x;
          flux_i_density += factor * (ff_variable[VAR_MOMENTUM] + momentum_i.x);
          flux_i_density_energy +=
              factor * (ff_flux_contribution_density_energy.x +
                        flux_contribution_i_density_energy.x);
          flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.x +
                                         flux_contribution_i_momentum_x.x);
          flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.x +
                                         flux_contribution_i_momentum_y.x);
          flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.x +
                                         flux_contribution_i_momentum_z.x);

          factor = float(0.5f) * normal.y;
          flux_i_density +=
              factor * (ff_variable[VAR_MOMENTUM + 1] + momentum_i.y);
          flux_i_density_energy +=
              factor * (ff_flux_contribution_density_energy.y +
                        flux_contribution_i_density_energy.y);
          flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.y +
                                         flux_contribution_i_momentum_x.y);
          flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.y +
                                         flux_contribution_i_momentum_y.y);
          flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.y +
                                         flux_contribution_i_momentum_z.y);

          factor = float(0.5f) * normal.z;
          flux_i_density +=
              factor * (ff_variable[VAR_MOMENTUM + 2] + momentum_i.z);
          flux_i_density_energy +=
              factor * (ff_flux_contribution_density_energy.z +
                        flux_contribution_i_density_energy.z);
          flux_i_momentum.x += factor * (ff_flux_contribution_momentum_x.z +
                                         flux_contribution_i_momentum_x.z);
          flux_i_momentum.y += factor * (ff_flux_contribution_momentum_y.z +
                                         flux_contribution_i_momentum_y.z);
          flux_i_momentum.z += factor * (ff_flux_contribution_momentum_z.z +
                                         flux_contribution_i_momentum_z.z);
        }
      }

      fluxes[i + VAR_DENSITY * nelr] = flux_i_density;
      fluxes[i + (VAR_MOMENTUM + 0) * nelr] = flux_i_momentum.x;
      fluxes[i + (VAR_MOMENTUM + 1) * nelr] = flux_i_momentum.y;
      fluxes[i + (VAR_MOMENTUM + 2) * nelr] = flux_i_momentum.z;
      fluxes[i + VAR_DENSITY_ENERGY * nelr] = flux_i_density_energy;
    }
  }
}

__global__ void time_step(int j, int nelr, float *old_variables,
                          float *variables, float *step_factors,
                          float *fluxes) {
  int blk = blockDim.x * blockIdx.x + threadIdx.x;
  if (blk < nelr / block_length) {
    int b_start = blk * block_length;
    int b_end =
        (blk + 1) * block_length > nelr ? nelr : (blk + 1) * block_length;
    for (int i = b_start; i < b_end; ++i) {
      float factor = step_factors[i] / float(RK + 1 - j);
      variables[i + VAR_DENSITY * nelr] =
          old_variables[i + VAR_DENSITY * nelr] +
          factor * fluxes[i + VAR_DENSITY * nelr];
      variables[i + (VAR_MOMENTUM + 0) * nelr] =
          old_variables[i + (VAR_MOMENTUM + 0) * nelr] +
          factor * fluxes[i + (VAR_MOMENTUM + 0) * nelr];
      variables[i + (VAR_MOMENTUM + 1) * nelr] =
          old_variables[i + (VAR_MOMENTUM + 1) * nelr] +
          factor * fluxes[i + (VAR_MOMENTUM + 1) * nelr];
      variables[i + (VAR_MOMENTUM + 2) * nelr] =
          old_variables[i + (VAR_MOMENTUM + 2) * nelr] +
          factor * fluxes[i + (VAR_MOMENTUM + 2) * nelr];
      variables[i + VAR_DENSITY_ENERGY * nelr] =
          old_variables[i + VAR_DENSITY_ENERGY * nelr] +
          factor * fluxes[i + VAR_DENSITY_ENERGY * nelr];
    }
  }
}

int main(int argc, char *argv[]) {
  float *ff_variable;
  float *areas;
  int *elements_surrounding_elements;
  float *normals;
  float *variables;
  float *old_variables;
  float *fluxes;
  float *step_factors;
  int iterations;
  int nel, nelr;
  std::string domainFile;
  std::string cpuRefFile, gpuRefFile;
  std::string outFile;
  Float3 ff_flux_contribution_momentum_x;
  Float3 ff_flux_contribution_momentum_y;
  Float3 ff_flux_contribution_momentum_z;
  Float3 ff_flux_contribution_density_energy;
  unsigned threadsPerBlock;

  parseCommandLineInto(argc, argv, domainFile, iterations, outFile, cpuRefFile,
                       gpuRefFile, &threadsPerBlock);

  TimerGroup tg("euler3d");
  Timer &total = tg.add("total", "Total");
  Timer &init = tg.add("init", "Init");
  Timer &iters = tg.add("iters", "Compute");
  Timer &copy = tg.add("copy", "Copy");
  Timer &sf = tg.add("step_factor", "Step factor");
  Timer &rk = tg.add("rk", "Runge-Kutta");

  header("forall", domainFile, iterations);

  // read in domain geometry
  read_domain(ff_variable, areas, elements_surrounding_elements, normals,
              variables, old_variables, fluxes, step_factors,
              ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,
              ff_flux_contribution_momentum_z,
              ff_flux_contribution_density_energy, nel, nelr, domainFile);
  unsigned bpgNelrNvar = (nelr * NVAR + threadsPerBlock - 1) / threadsPerBlock;
  unsigned bpgNelrAll = (nelr + threadsPerBlock - 1) / threadsPerBlock;
  unsigned bpgNelr =
      (nelr / block_length + threadsPerBlock - 1) / threadsPerBlock;

  // Create arrays and set initial conditions
  total.start();
  init.start();
  initialize_variables<<<bpgNelrAll, threadsPerBlock>>>(nelr, variables,
                                                        ff_variable);
  cudaDeviceSynchronize();
  init.stop();

  // Begin iterations
  iters.start();
  for (int i = 0; i < iterations; i++) {
    copy.start();
    cpy<<<bpgNelrNvar, threadsPerBlock>>>(old_variables, variables,
                                           nelr * NVAR);
    copy.stop();

    // for the first iteration we compute the time step
    sf.start();
    compute_step_factor<<<bpgNelr, threadsPerBlock>>>(nelr, variables, areas,
                                                      step_factors);
    cudaDeviceSynchronize();
    sf.stop();

    rk.start();
    for (int j = 0; j < RK; j++) {
      compute_flux<<<bpgNelr, threadsPerBlock>>>(
          nelr, elements_surrounding_elements, normals, variables, fluxes,
          ff_variable, ff_flux_contribution_momentum_x,
          ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z,
          ff_flux_contribution_density_energy);
      cudaDeviceSynchronize();
      time_step<<<bpgNelr, threadsPerBlock>>>(j, nelr, old_variables, variables,
                                              step_factors, fluxes);
      cudaDeviceSynchronize();
    }
    rk.stop();
  }
  iters.stop();
  total.stop();

  // ok will be true (non-zero) on success. But the OS needs 0 to indicate
  // success.
  bool ok = footer(tg, ff_variable, areas, elements_surrounding_elements,
                   normals, variables, old_variables, fluxes, step_factors, nel,
                   nelr, outFile, cpuRefFile, gpuRefFile);
  return !ok;
}
