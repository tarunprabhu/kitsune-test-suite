// Copyright 2009, Andrew Corrigan, acorriga@gmu.edu
// This code is from the AIAA-2009-4001 paper

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <kitsune.h>

#include "fpcmp.h"
#include "timing.h"
#include "utils.h"

#include "../../Benchmarks/euler3d/euler3d.inc"

void initialize_variables(int nelr, kitsune::mobile_ptr<float> variables,
                          kitsune::mobile_ptr<float> ff_variable);

void compute_flux(int nelr,
                  const kitsune::mobile_ptr<int> elements_surrounding_elements,
                  const kitsune::mobile_ptr<float> normals,
                  const kitsune::mobile_ptr<float> variables,
                  kitsune::mobile_ptr<float> fluxes,
                  const kitsune::mobile_ptr<float> ff_variable,
                  const Float3 ff_flux_contribution_momentum_x,
                  const Float3 ff_flux_contribution_momentum_y,
                  const Float3 ff_flux_contribution_momentum_z,
                  const Float3 ff_flux_contribution_density_energy);

void time_step(int j, int nelr, kitsune::mobile_ptr<float> old_variables,
               kitsune::mobile_ptr<float> variables,
               kitsune::mobile_ptr<float> step_factors,
               kitsune::mobile_ptr<float> fluxes);

void compute_step_factor(int nelr, const kitsune::mobile_ptr<float> variables,
                         const kitsune::mobile_ptr<float> areas,
                         kitsune::mobile_ptr<float> step_factors);

int main(int argc, char *argv[]) {
  kitsune::mobile_ptr<float> ff_variable;
  kitsune::mobile_ptr<float> areas;
  kitsune::mobile_ptr<int> elements_surrounding_elements;
  kitsune::mobile_ptr<float> normals;
  kitsune::mobile_ptr<float> variables;
  kitsune::mobile_ptr<float> old_variables;
  kitsune::mobile_ptr<float> fluxes;
  kitsune::mobile_ptr<float> step_factors;
  int iterations;
  int nel, nelr;
  std::string domainFile;
  std::string cpuRefFile, gpuRefFile;
  std::string outFile;
  Float3 ff_flux_contribution_momentum_x;
  Float3 ff_flux_contribution_momentum_y;
  Float3 ff_flux_contribution_momentum_z;
  Float3 ff_flux_contribution_density_energy;

  parseCommandLineInto(argc, argv, domainFile, iterations, outFile, cpuRefFile,
                       gpuRefFile);

  TimerGroup tg("euler3d");
  Timer &total = tg.add("total", "Total");
  Timer &init = tg.add("init", "Init");
  Timer &iters = tg.add("iters", "Compute");
  Timer &copy = tg.add("copy", "Copy");
  Timer &sf = tg.add("step_factor", "Step factor");
  Timer &rk = tg.add("rk", "Runge-Kutta");

  header("forall", domainFile, iterations);

  // read in domain geometry and create arrays
  read_domain(ff_variable, areas, elements_surrounding_elements, normals,
              variables, old_variables, fluxes, step_factors,
              ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,
              ff_flux_contribution_momentum_z,
              ff_flux_contribution_density_energy, nel, nelr, domainFile);

  total.start();
  init.start();
  initialize_variables(nelr, variables, ff_variable);
  init.stop();

  // Begin iterations
  iters.start();
  for (int i = 0; i < iterations; i++) {
    copy.start();
    cpy(old_variables, variables, nelr * NVAR);
    copy.stop();

    // for the first iteration we compute the time step
    sf.start();
    compute_step_factor(nelr, variables, areas, step_factors);
    sf.stop();

    rk.start();
    for (int j = 0; j < RK; j++) {
      compute_flux(nelr, elements_surrounding_elements, normals, variables,
                   fluxes, ff_variable, ff_flux_contribution_momentum_x,
                   ff_flux_contribution_momentum_y,
                   ff_flux_contribution_momentum_z,
                   ff_flux_contribution_density_energy);
      time_step(j, nelr, old_variables, variables, step_factors, fluxes);
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
