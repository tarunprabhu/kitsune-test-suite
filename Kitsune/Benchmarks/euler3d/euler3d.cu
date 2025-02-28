/// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

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

void cpy(float *dst, const float *src, int N) {
  for (unsigned int i = 0; i < N; i++)
    dst[i] = src[i];
}

void dump(float *variables, int nel, int nelr, char *name) {
  std::string nameD = (std::string)name + "_density.dat";
  FILE *fpD = fopen(nameD.c_str(), "wb");
  fwrite(&variables[VAR_DENSITY * nelr], sizeof(float), nel, fpD);
  fclose(fpD);

  std::string nameM = (std::string)name + "_momentum.dat";
  FILE *fpM = fopen(nameM.c_str(), "wb");
  for (int j = 0; j < NDIM; j++)
    fwrite(&variables[(VAR_MOMENTUM + j) * nelr], sizeof(float), nel, fpM);
  fclose(fpM);

  std::string nameDE = (std::string)name + "_density_energy.dat";
  FILE *fpDE = fopen(nameDE.c_str(), "wb");
  fwrite(&variables[VAR_DENSITY_ENERGY * nelr], sizeof(float), nel, fpDE);
  fclose(fpDE);
}

void initialize_variables(int nelr, float *variables, float *ff_variable) {
  for (int i = 0; i < nelr; i++) {
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
  using namespace std;
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

/*
 * Main function
 */
int main(int argc, char **argv) {
  using namespace std;

  if (argc < 2) {
    cout << "specify data file name" << endl;
    return 0;
  }

  int iterations = 4000;
  if (argc > 2)
    iterations = atoi(argv[2]);

  const char *data_file_name = argv[1];

  cout << setprecision(5);
  cout << "\n";
  cout << "---- euler3d benchmark (cuda) ----\n\n"
       << "  Input file : " << data_file_name << "\n"
       << "  Iterations : " << iterations << ".\n\n";
  cout << "  Reading input data, allocating arrays, initializing data, etc..."
       << std::flush;
  auto total_start_time = chrono::steady_clock::now();

  // these need to be computed the first time in order to compute time step
  cudaError_t err = cudaSuccess;
  float *ff_variable;
  err = cudaMallocManaged(&ff_variable, NVAR * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  Float3 ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,
      ff_flux_contribution_momentum_z;
  Float3 ff_flux_contribution_density_energy;

  // set far field conditions
  const float angle_of_attack =
      float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

  ff_variable[VAR_DENSITY] = float(1.4);

  float ff_pressure = float(1.0f);
  float ff_speed_of_sound =
      sqrtf(GAMMA * ff_pressure / ff_variable[VAR_DENSITY]);
  float ff_speed = float(ff_mach) * ff_speed_of_sound;

  Float3 ff_velocity;
  ff_velocity.x = ff_speed * float(cosf((float)angle_of_attack));
  ff_velocity.y = ff_speed * float(sinf((float)angle_of_attack));
  ff_velocity.z = 0.0f;

  ff_variable[VAR_MOMENTUM + 0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
  ff_variable[VAR_MOMENTUM + 1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
  ff_variable[VAR_MOMENTUM + 2] = ff_variable[VAR_DENSITY] * ff_velocity.z;

  ff_variable[VAR_DENSITY_ENERGY] =
      ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed)) +
      (ff_pressure / float(GAMMA - 1.0f));

  Float3 ff_momentum;
  ff_momentum.x = *(ff_variable + VAR_MOMENTUM + 0);
  ff_momentum.y = *(ff_variable + VAR_MOMENTUM + 1);
  ff_momentum.z = *(ff_variable + VAR_MOMENTUM + 2);
  compute_flux_contribution(
      ff_variable[VAR_DENSITY], ff_momentum, ff_variable[VAR_DENSITY_ENERGY],
      ff_pressure, ff_velocity, ff_flux_contribution_momentum_x,
      ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z,
      ff_flux_contribution_density_energy);

  int nel;
  int nelr;

  // read in domain geometry
  float *areas;
  int *elements_surrounding_elements;
  float *normals;

  ifstream file(data_file_name);
  file >> nel;
  nelr = block_length * ((nel / block_length) + min(1, nel % block_length));

  err = cudaMallocManaged(&areas, nelr * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&elements_surrounding_elements,
                          nelr * NNB * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  err = cudaMallocManaged(&normals, NDIM * NNB * nelr * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }

  // read in data
  for (int i = 0; i < nel; i++) {
    file >> areas[i];
    for (int j = 0; j < NNB; j++) {
      file >> elements_surrounding_elements[i + j * nelr];
      if (elements_surrounding_elements[i + j * nelr] < 0)
        elements_surrounding_elements[i + j * nelr] = -1;
      // it's coming in with Fortran numbering
      elements_surrounding_elements[i + j * nelr]--;

      for (int k = 0; k < NDIM; k++) {
        file >> normals[i + (j + k * NNB) * nelr];
        normals[i + (j + k * NNB) * nelr] = -normals[i + (j + k * NNB) * nelr];
      }
    }
  }

  // fill in remaining data
  int last = nel - 1;
  for (int i = nel; i < nelr; i++) {
    areas[i] = areas[last];
    for (int j = 0; j < NNB; j++) {
      // duplicate the last element
      elements_surrounding_elements[i + j * nelr] =
          elements_surrounding_elements[last + j * nelr];
      for (int k = 0; k < NDIM; k++)
        normals[i + (j + k * NNB) * nelr] =
            normals[last + (j + k * NNB) * nelr];
    }
  }

  // Create arrays and set initial conditions
  float *variables;
  err = cudaMallocManaged(&variables, nelr * NVAR * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  cout << "  done.\n\n";

  cout << "  Starting benchmark...\n" << std::flush;
  auto start_time = chrono::steady_clock::now();
  int threadsPerBlock = 256;
  int blocksPerGrid =
      (nelr / block_length + threadsPerBlock - 1) / threadsPerBlock;

  initialize_variables(nelr, variables, ff_variable);
  float *old_variables;
  err = cudaMallocManaged(&old_variables, nelr * NVAR * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  float *fluxes;
  err = cudaMallocManaged(&fluxes, nelr * NVAR * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }
  float *step_factors;
  err = cudaMallocManaged(&step_factors, nelr * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "failed to allocate managed memory!\n");
    return 1;
  }

  double *rk_times = new double[iterations];

  // Begin iterations
  double copy_total = 0.0;
  double sf_total = 0.0;
  double rk_total = 0.0;

  for (int i = 0; i < iterations; i++) {
    auto copy_start = chrono::steady_clock::now();
    cpy(old_variables, variables, nelr * NVAR);
    auto copy_end = chrono::steady_clock::now();
    double time = chrono::duration<double>(copy_end - copy_start).count();
    copy_total += time;

    // for the first iteration we compute the time step
    auto sf_start = chrono::steady_clock::now();
    compute_step_factor<<<blocksPerGrid, threadsPerBlock>>>(
        nelr, variables, areas, step_factors);
    cudaDeviceSynchronize();
    auto sf_end = chrono::steady_clock::now();
    time = chrono::duration<double>(sf_end - sf_start).count();
    sf_total += time;

    auto rk_start = chrono::steady_clock::now();
    for (int j = 0; j < RK; j++) {
      compute_flux<<<blocksPerGrid, threadsPerBlock>>>(
          nelr, elements_surrounding_elements, normals, variables, fluxes,
          ff_variable, ff_flux_contribution_momentum_x,
          ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z,
          ff_flux_contribution_density_energy);
      cudaDeviceSynchronize();
      time_step<<<blocksPerGrid, threadsPerBlock>>>(
          j, nelr, old_variables, variables, step_factors, fluxes);
      cudaDeviceSynchronize();
    }
    auto rk_end = chrono::steady_clock::now();
    time = chrono::duration<double>(rk_end - rk_start).count();
    rk_times[i] = time;
    rk_total += time;
  }
  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>(end_time - start_time).count();

  dump(variables, nel, nelr, argv[0]);

  end_time = chrono::steady_clock::now();
  double total_time =
      chrono::duration<double>(end_time - total_start_time).count();
  double rk_mean = rk_total / (iterations - 1);
  double sum = 0.0;
  for (int i = 1; i < iterations; i++) {
    double dist = rk_times[i] - rk_mean;
    sum += dist * dist;
  }
  double rk_std_dev = sqrt(sum / iterations);

  cout << "\n"
       << "      Total time : " << total_time << " seconds.\n"
       << "    Compute time : " << elapsed_time << " seconds.\n"
       << "            copy : " << copy_total
       << " seconds (average: " << copy_total / iterations << " seconds).\n"
       << "              sf : " << sf_total
       << " seconds (average: " << sf_total / iterations << " seconds).\n"
       << "              rk : " << rk_total << " seconds (average: " << rk_mean
       << " seconds / std dev:" << rk_std_dev << ").\n"
       << "*** " << elapsed_time << ", " << elapsed_time << "\n"
       << "----\n\n";

  return 0;
}
