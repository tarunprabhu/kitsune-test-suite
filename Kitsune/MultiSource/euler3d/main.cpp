// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>

#include <kitsune.h>

#include "utils.h"

#include "../../SingleSource/euler3d/euler3d.h"
#include "../../SingleSource/euler3d/euler3d.inc"

void initialize_variables(int nelr, float *[[kitsune::mobile]] variables,
                          float *[[kitsune::mobile]] ff_variable);

void compute_flux(int nelr,
                  const int *[[kitsune::mobile]] elements_surrounding_elements,
                  const float *[[kitsune::mobile]] normals,
                  const float *[[kitsune::mobile]] variables,
                  float *[[kitsune::mobile]] fluxes,
                  const float *[[kitsune::mobile]] ff_variable,
                  const Float3 ff_flux_contribution_momentum_x,
                  const Float3 ff_flux_contribution_momentum_y,
                  const Float3 ff_flux_contribution_momentum_z,
                  const Float3 ff_flux_contribution_density_energy);

void time_step(int j, int nelr, float *[[kitsune::mobile]] old_variables,
               float *[[kitsune::mobile]] variables,
               float *[[kitsune::mobile]] step_factors,
               float *[[kitsune::mobile]] fluxes);

void compute_step_factor(int nelr, const float *[[kitsune::mobile]] variables,
                         const float *[[kitsune::mobile]] areas,
                         float *[[kitsune::mobile]] step_factors);

[[clang::noinline]]
static void
test(float *[[kitsune::mobile]] ff_variable, float *[[kitsune::mobile]] areas,
     int *[[kitsune::mobile]] elements_surrounding_elements,
     float *[[kitsune::mobile]] normals, float *[[kitsune::mobile]] variables,
     float *[[kitsune::mobile]] old_variables,
     float *[[kitsune::mobile]] fluxes, float *[[kitsune::mobile]] step_factors,
     Float3 &ff_flux_contribution_momentum_x,
     Float3 &ff_flux_contribution_momentum_y,
     Float3 &ff_flux_contribution_momentum_z,
     Float3 &ff_flux_contribution_density_energy, int nel, int nelr,
     int iterations) {
  // Initialize variables is part of test() because it uses forall's
  initialize_variables(nelr, variables, ff_variable);

  for (int i = 0; i < iterations; i++) {
    cpy(old_variables, variables, nelr * NVAR);

    // for the first iteration we compute the time step
    compute_step_factor(nelr, variables, areas, step_factors);

    for (int j = 0; j < RK; j++) {
      compute_flux(nelr, elements_surrounding_elements, normals, variables,
                   fluxes, ff_variable, ff_flux_contribution_momentum_x,
                   ff_flux_contribution_momentum_y,
                   ff_flux_contribution_momentum_z,
                   ff_flux_contribution_density_energy);
      time_step(j, nelr, old_variables, variables, step_factors, fluxes);
    }
  }
}

int main(int argc, char *argv[]) {
  float *[[kitsune::mobile]] ff_variable = nullptr;
  float *[[kitsune::mobile]] areas = nullptr;
  int *[[kitsune::mobile]] elements_surrounding_elements = nullptr;
  float *[[kitsune::mobile]] normals = nullptr;
  float *[[kitsune::mobile]] variables = nullptr;
  float *[[kitsune::mobile]] old_variables = nullptr;
  float *[[kitsune::mobile]] fluxes = nullptr;
  float *[[kitsune::mobile]] step_factors = nullptr;
  int iterations;
  int nel, nelr;
  std::string domainFile;
  std::string checkFile;
  std::string outFile;
  Float3 ff_flux_contribution_momentum_x;
  Float3 ff_flux_contribution_momentum_y;
  Float3 ff_flux_contribution_momentum_z;
  Float3 ff_flux_contribution_density_energy;

  parseCommandLineArgs(argc, argv, iterations, domainFile, checkFile, outFile);

  setup(ff_variable, areas, elements_surrounding_elements, normals, variables,
        old_variables, fluxes, step_factors, ff_flux_contribution_momentum_x,
        ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z,
        ff_flux_contribution_density_energy, nel, nelr, domainFile);

  test(ff_variable, areas, elements_surrounding_elements, normals, variables,
       old_variables, fluxes, step_factors, ff_flux_contribution_momentum_x,
       ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z,
       ff_flux_contribution_density_energy, nel, nelr, iterations);

  save(outFile, (float *)variables, nel, nelr);

  bool ok = true;
  if (checkFile.size())
    ok = check(outFile, checkFile);

  teardown(ff_variable, areas, elements_surrounding_elements, normals,
           variables, old_variables, fluxes, step_factors);

  return report(ok);
}
