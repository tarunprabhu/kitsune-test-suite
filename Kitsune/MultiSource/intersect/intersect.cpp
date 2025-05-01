//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.    It is released under
// the LLVM license.
//
// Simple example of mesh intersection. Hopefully represents relevant memory
// access patterns.
//

// FIXME: This is now totally broken. _tapir_cuda_target will never be passed
// by Kitsune's frontend. In order to have a similar effect, one would need to
// rely on Kitsune's middle end passes correctly dealing with memory allocations
// (or we need to provide some hooks to do it manually). In either case, this
// test needs some significant reworking. But there are more serious issues
// with supporting multi-targets *cleanly* in the compiler, so it would all need
// to be handled with some more care.
#ifdef _tapir_cuda_target
constexpr bool HAVE_KITSUNE = true;
#else
constexpr bool HAVE_KITSUNE = false;
#endif

#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>

#include <kitsune.h>
#include "kitsune/timer.h"

#include "common.h"
#include "serial_algorithms.h"

const bool CHECK = true;

using namespace std;

int main(int argc, char *argv[]) {

  if constexpr (HAVE_KITSUNE){
    printf ("At runtime we HAVE_KITSUNE!\n");
    // __kitrt_cuInit();
  } else
    printf ("At runtime we do not HAVE_KITSUNE!\n");


  // for profiling, do all the cuda overhead now
  nvtxRangePushA("Kitsune/Cuda Initialization");
  kitsune::mobile<int> dum(16);
  dum.free(); // doesn't work ATM
  nvtxRangePop();

  ///////////////////////////////
  // simulation parameters
  ///////////////////////////////

  int nx = 3, ny = 3;
  const double x_max = 1., y_max = 1., shift_x = .0, shift_y = .0;
  bool shuffle = false, use_UVM = false;
  PrefetchKinds PFKind = NONE;

  parse_args(argc, argv, &nx, &ny, &shuffle, &PFKind);

  // derived parameters
  const int n_cells = nx * ny, n_nodes = (nx + 1) * (ny + 1);

  printf("x_max=%f, y_max=%f\nshift_x=%f, shift_y=%f\n", x_max, y_max, shift_x,
         shift_y);

  ///////////////////////////////
  // create the shuffle arrays if necessary
  ///////////////////////////////

  nvtxRangePushA("Create the entity permutations");
  // randomness for the permutations
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto DRE = std::default_random_engine(0);

  // create the source node and cell permutations, target node only

  std::vector<size_t> shuffle_source_nodes((nx + 1) * (ny + 1));
  std::vector<size_t> shuffle_source_cells(nx * ny);
  std::vector<size_t> shuffle_target_nodes((nx + 1) * (ny + 1));

  if (shuffle) {
    for (size_t i = 0; i < (nx + 1) * (ny + 1); ++i)
      shuffle_source_nodes[i] = i;
    std::shuffle(shuffle_source_nodes.begin(), shuffle_source_nodes.end(), DRE);

    for (size_t i = 0; i < nx * ny; ++i)
      shuffle_source_cells[i] = i;
    std::shuffle(shuffle_source_cells.begin(), shuffle_source_cells.end(), DRE);

    for (size_t i = 0; i < (nx + 1) * (ny + 1); ++i)
      shuffle_target_nodes[i] = i;
    std::shuffle(shuffle_target_nodes.begin(), shuffle_target_nodes.end(), DRE);
  }
  nvtxRangePop();


  nvtxRangePushA("Serial Work");

  ///////////////////////////////
  // create the mesh
  ///////////////////////////////

  nvtxRangePushA("Serial Mesh Creation");

  // declare source, target meshes, and intersection candidates
  double *source_coordinates, *target_coordinates;
  size_t *source_cell_nodes, *source_node_offsets;
  size_t *target_cell_nodes, *target_node_offsets;
  size_t *candidates, *candidate_offsets;

  create_meshes(HOST, nx, ny, x_max, y_max, shift_x, shift_y,
                source_coordinates, source_cell_nodes, source_node_offsets,
                target_coordinates, target_cell_nodes, target_node_offsets,
                candidates, candidate_offsets, shuffle, shuffle_source_nodes,
                shuffle_source_cells, shuffle_target_nodes);
  nvtxRangePop();

  ///////////////////////////////
  // do serial work
  ///////////////////////////////

  nvtxRangePushA("Serial Centroid Computation");
  double *source_centroids =
      allocate<double>(HOST, 2 * n_cells, "source centroids");
  nvtxMark("starting centroid computation");
  for (size_t i = 0; i < n_cells; ++i)
    serial::centroid(i, source_node_offsets, source_cell_nodes,
                     source_coordinates, source_centroids);
  if constexpr (LOG_LEVEL > 0) {
    printf("\nComputed centroids\n");
    for (size_t i = 0; i < n_cells; ++i)
      printf("cell %3zu centroid: %.4f, %.4f\n", i, source_centroids[2 * i],
             source_centroids[2 * i + 1]);
  }
  nvtxRangePop();
  nvtxRangePop();


  ///////////////////////////////
  // parallel setup
  ///////////////////////////////

  int threadsPerBlock = 256;
  int blocksPerGrid = (n_cells + threadsPerBlock - 1) / threadsPerBlock;
  fprintf(
      stderr,
      "\nKernel launch parameters: blocksPerGrid=%d, threadsPerBlock=%d\n\n",
      blocksPerGrid, threadsPerBlock);

  // declare the various arrays that we will reuse in the different execution graphs
  double *source_coordinates_kit;
  size_t *source_cell_nodes_kit;
  size_t *source_node_offsets_kit;

  double *target_coordinates_kit;
  size_t *target_cell_nodes_kit;
  size_t *target_node_offsets_kit;

  size_t *candidate_offsets_kit;
  size_t *candidates_kit;

  double *centroids_kit1;
  double *centroids_kit2;
  double *centroids_kit3;
  double *centroids_kit4;
  double *centroids_kit5;
  double *centroids_kit6;
  double *centroids_kit7;
  double *centroids_kit8;
  double *centroids_kit9;
  double *centroids_kit10;


  //////////////////////////////////
  // FIRST EXECUTION GRAPH, create the mesh in UVM directly
  //////////////////////////////////

  nvtxRangeId_t NVTX_PMC1 = nvtxRangeStartA("Parallel Mesh Creation 1 in UVM");

  nvtxRangeId_t NVTX_PR11 = nvtxRangeStartA("Parallel Region 1");

  spawn source_coordinates
  {
  nvtxRangeId_t NVTX_SC1 = nvtxRangeStartA("Source Coords");
  create_coordinates_gpu(source_coordinates_kit, nx, ny, x_max, y_max, 0., 0.,
                     "source_coordinates gpu",
                     shuffle ? &shuffle_source_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_SC1);
  }

  spawn target_coordinates
  {
  nvtxRangeId_t NVTX_TC1 = nvtxRangeStartA("Target Coords");
  create_coordinates_gpu(target_coordinates_kit, nx, ny, x_max, y_max, shift_x,
                     shift_y, "target_coordinates gpu",
                     shuffle ? &shuffle_target_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_TC1);
  }

  spawn source_cell_nodes
  {
  nvtxRangeId_t NVTX_SC2N1 = nvtxRangeStartA("Source C2N");
  create_cell_nodes_gpu(source_cell_nodes_kit, nx, ny, "source_cell_nodes",
                    shuffle ? &shuffle_source_nodes : nullptr,
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_SC2N1);
  }

  spawn target_cell_nodes
  {
  nvtxRangeId_t NVTX_TC2N1 = nvtxRangeStartA("Target C2N");
  // no need to shuffle the target cells
  create_cell_nodes_gpu(target_cell_nodes_kit, nx, ny, "target_cell_nodes",
                    shuffle ? &shuffle_target_nodes : nullptr, nullptr);
  nvtxRangeEnd(NVTX_TC2N1);
  }

  source_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "source_node_offsets");
  spawn source_node_offsets
  {
  nvtxRangeId_t NVTX_SNO1 = nvtxRangeStartA("Source Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    source_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_SNO1);
  }

  target_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "target node offsets");
  spawn target_node_offsets
  {
  nvtxRangeId_t NVTX_TNO1 = nvtxRangeStartA("Target Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    target_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_TNO1);
  }

  spawn create_candidate_offsets
  {
  nvtxRangeId_t NVTX_SCO1 = nvtxRangeStartA("Source Candidate Offsets");
  create_candidate_offsets(candidate_offsets_kit, nx, ny);
  nvtxRangeEnd(NVTX_SCO1);
  }

  sync source_coordinates;
  sync target_coordinates;
  sync source_cell_nodes;
  sync target_cell_nodes;
  sync source_node_offsets;
  sync target_node_offsets;
  sync create_candidate_offsets;

  nvtxRangeEnd(NVTX_PR11);//Parallel Region 1

  // At this point we know the number of candidates so we can allocate. All the syncs
  // needed to complete in create_meshes_gpu in order for us to do the allocation here.
  // This isn't really what we want, but is the best we can do at the moment.
  nvtxRangeId_t NVTX_PR21 = nvtxRangeStartA("Parallel Region 2");

  spawn allocate_and_compute_candidates {
  nvtxRangeId_t NVTX_CAN1 = nvtxRangeStartA("Computing candidates");
  create_candidates_gpu(candidates_kit, candidate_offsets_kit[n_cells], nx, ny, candidate_offsets_kit,
                    "create candidates ",
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_CAN1);

  }

  // compute centroids
  spawn cuda_centroids{
  nvtxRangeId_t NVTX_CEN1 = nvtxRangeStartA("Centroid on mesh defined with UVM");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  nvtxMark("Centroid Kitsune kernel start...");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument
  double *results = &source_coordinates_kit[2 * n_nodes];
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    &source_coordinates_kit[2 * n_nodes]);
  nvtxMark("Centroid kernel end");

  // check centroids
  if constexpr (LOG_LEVEL > 0)
    for (size_t i = 0; i < n_cells; ++i)
        printf("cell %3zu centroid: %.4f, %.4f\n", i, results[2 * i],
                results[2 * i + 1]);
  if constexpr (CHECK){
    printf("Graph 1 Centroids: ");
    check_equal(results, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN1);
  }

  sync cuda_centroids;
  sync allocate_and_compute_candidates;

  nvtxRangeEnd(NVTX_PR21); // Parallel Region 2

  ///////////////////////////////
  // free application heap memory
  ///////////////////////////////

  nvtxRangeId_t NVTX_FREE1 = nvtxRangeStartA("Free Resources");
  dealloc(source_coordinates_kit);
  dealloc(source_cell_nodes_kit);
  dealloc(source_node_offsets_kit);

  dealloc(target_coordinates_kit);
  dealloc(target_cell_nodes_kit);
  dealloc(target_node_offsets_kit);

  dealloc(candidate_offsets_kit);
  dealloc(candidates_kit);
  nvtxRangeEnd(NVTX_FREE1);

  nvtxRangeEnd(NVTX_PMC1); // Parallel Mesh Creation 1 in UVM


  ///////////////////////////////
  // SECOND EXECUTION GRAPH
  ///////////////////////////////

  nvtxRangeId_t NVTX_PMC2 = nvtxRangeStartA("Parallel Mesh Creation 2 in UVM");

  nvtxRangeId_t NVTX_PR12 = nvtxRangeStartA("Parallel Region 1");

  spawn source_coordinates
  {
  nvtxRangeId_t NVTX_SC2 = nvtxRangeStartA("Source Coords");
  create_coordinates_gpu(source_coordinates_kit, nx, ny, x_max, y_max, 0., 0.,
                     "source_coordinates gpu",
                     shuffle ? &shuffle_source_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_SC2);
  }

  spawn target_coordinates
  {
  nvtxRangeId_t NVTX_TC2 = nvtxRangeStartA("Target Coords");
  create_coordinates_gpu(target_coordinates_kit, nx, ny, x_max, y_max, shift_x,
                     shift_y, "target_coordinates gpu",
                     shuffle ? &shuffle_target_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_TC2);
  }

  spawn source_cell_nodes
  {
  nvtxRangeId_t NVTX_SC2N2 = nvtxRangeStartA("Source C2N");
  create_cell_nodes_gpu(source_cell_nodes_kit, nx, ny, "source_cell_nodes",
                    shuffle ? &shuffle_source_nodes : nullptr,
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_SC2N2);
  }

  spawn target_cell_nodes
  {
  nvtxRangeId_t NVTX_TC2N2 = nvtxRangeStartA("Target C2N");
  // no need to shuffle the target cells
  create_cell_nodes_gpu(target_cell_nodes_kit, nx, ny, "target_cell_nodes",
                    shuffle ? &shuffle_target_nodes : nullptr, nullptr);
  nvtxRangeEnd(NVTX_TC2N2);
  }

  source_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "source_node_offsets");
  spawn source_node_offsets
  {
  nvtxRangeId_t NVTX_SNO2 = nvtxRangeStartA("Source Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    source_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_SNO2);
  }

  target_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "target_node_offsets");
  spawn target_node_offsets
  {
  nvtxRangeId_t NVTX_TNO2 = nvtxRangeStartA("Target Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    target_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_TNO2);
  }

  spawn create_candidate_offsets
  {
  nvtxRangeId_t NVTX_CANDO2 = nvtxRangeStartA("Candidates and Offsets");
  nvtxRangeId_t NVTX_CO2 = nvtxRangeStartA("Candidate Offsets");
  create_candidate_offsets(candidate_offsets_kit, nx, ny);
  nvtxRangeEnd(NVTX_CO2);
  nvtxRangeId_t NVTX_CAN2 = nvtxRangeStartA("Candidates");
  create_candidates_gpu(candidates_kit, candidate_offsets_kit[n_cells], nx, ny, candidate_offsets_kit,
                    "create candidates ",
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_CAN2);
  nvtxRangeEnd(NVTX_CANDO2); // Candidates and offsets
  }

  sync source_coordinates;
  sync target_coordinates;
  sync source_cell_nodes;
  sync target_cell_nodes;
  sync source_node_offsets;
  sync target_node_offsets;
  sync create_candidate_offsets;

  nvtxRangeEnd(NVTX_PR12); // Parallel Region 1

  // At this point we know the number of candidates so we can allocate. All the syncs
  // needed to complete in create_meshes_gpu in order for us to do the allocation here.
  // This isn't really what we want, but is the best we can do at the moment.
  nvtxRangeId_t NVTX_PR22 = nvtxRangeStartA("Parallel Region 2");


  // compute centroids
  spawn cuda_centroids{
  nvtxRangeId_t NVTX_CEN2 = nvtxRangeStartA("Centroid on mesh defined with UVM");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument
  double *results = &source_coordinates_kit[2 * n_nodes];
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    &source_coordinates_kit[2 * n_nodes]);

  // check centroids
  if constexpr (LOG_LEVEL > 0)
    for (size_t i = 0; i < n_cells; ++i)
        printf("cell %3zu centroid: %.4f, %.4f\n", i, results[2 * i],
                results[2 * i + 1]);
  if constexpr (CHECK){
    printf("Graph 2 Centroids: ");
    check_equal(results, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN2);
  }

  sync cuda_centroids;
  nvtxRangeEnd(NVTX_PR22); // Parallel Region 2

  ///////////////////////////////
  // free application heap memory
  ///////////////////////////////

  nvtxRangeId_t NVTX_FREE2 = nvtxRangeStartA("Free Resources");
  dealloc(source_coordinates_kit);
  dealloc(source_cell_nodes_kit);
  dealloc(source_node_offsets_kit);

  dealloc(target_coordinates_kit);
  dealloc(target_cell_nodes_kit);
  dealloc(target_node_offsets_kit);

  dealloc(candidate_offsets_kit);
  dealloc(candidates_kit);
  nvtxRangeEnd(NVTX_FREE2);

  nvtxRangeEnd(NVTX_PMC2); // Parallel Mesh Creation 2 in UVM


  ///////////////////////////////
  // THIRD EXECUTION GRAPH
  ///////////////////////////////

  nvtxRangeId_t NVTX_PMC3 = nvtxRangeStartA("Parallel Mesh Creation 3 in UVM");

  nvtxRangeId_t NVTX_PR13 = nvtxRangeStartA("Parallel Region 1");

  spawn source_coordinates
  {
  nvtxRangeId_t NVTX_SC3 = nvtxRangeStartA("Source Coords");
  create_coordinates_gpu(source_coordinates_kit, nx, ny, x_max, y_max, 0., 0.,
                     "source_coordinates gpu",
                     shuffle ? &shuffle_source_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_SC3);
  }

  spawn source_cell_nodes
  {
  nvtxRangeId_t NVTX_SC2N3 = nvtxRangeStartA("Source C2N");
  create_cell_nodes_gpu(source_cell_nodes_kit, nx, ny, "source_cell_nodes",
                    shuffle ? &shuffle_source_nodes : nullptr,
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_SC2N3);
  }

  source_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "source_node_offsets");
  spawn source_node_offsets
  {
  nvtxRangeId_t NVTX_SNO3 = nvtxRangeStartA("Source Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    source_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_SNO3);
  }

  spawn target_coordinates
  {
  nvtxRangeId_t NVTX_TC3 = nvtxRangeStartA("Target Coords");
  create_coordinates_gpu(target_coordinates_kit, nx, ny, x_max, y_max, shift_x,
                     shift_y, "target_coordinates gpu",
                     shuffle ? &shuffle_target_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_TC3);
  }

  spawn target_cell_nodes
  {
  nvtxRangeId_t NVTX_TC2N3 = nvtxRangeStartA("Target C2N");
  // no need to shuffle the target cells
  create_cell_nodes_gpu(target_cell_nodes_kit, nx, ny, "target_cell_nodes",
                    shuffle ? &shuffle_target_nodes : nullptr, nullptr);
  nvtxRangeEnd(NVTX_TC2N3);
  }

  target_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "target_node_offsets");
  spawn target_node_offsets
  {
  nvtxRangeId_t NVTX_TNO3 = nvtxRangeStartA("Target Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    target_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_TNO3);
  }

  spawn create_candidate_offsets
  {
  nvtxRangeId_t NVTX_CANDO3 = nvtxRangeStartA("Candidates and Offsets");
  nvtxRangeId_t NVTX_CO3 = nvtxRangeStartA("Candidate Offsets");
  create_candidate_offsets(candidate_offsets_kit, nx, ny);
  nvtxRangeEnd(NVTX_CO3);
  nvtxRangeId_t NVTX_CAN3 = nvtxRangeStartA("Candidates");
  create_candidates_gpu(candidates_kit, candidate_offsets_kit[n_cells], nx, ny, candidate_offsets_kit,
                    "create candidates ",
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_CAN3);
  nvtxRangeEnd(NVTX_CANDO3); // Candidates and offsets
  }

  sync source_coordinates;
  sync source_cell_nodes;
  sync source_node_offsets;

  // compute centroids
  spawn cuda_centroids{
  nvtxRangeId_t NVTX_CEN3 = nvtxRangeStartA("Centroid on mesh defined with UVM");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument
  double *results = &source_coordinates_kit[2 * n_nodes];
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    &source_coordinates_kit[2 * n_nodes]);

  // check centroids
  if constexpr (LOG_LEVEL > 0)
    for (size_t i = 0; i < n_cells; ++i)
        printf("cell %3zu centroid: %.4f, %.4f\n", i, results[2 * i],
                results[2 * i + 1]);
  if constexpr (CHECK){
    printf("Graph 3 Centroids: ");
    check_equal(results, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN3);
  }

  sync create_candidate_offsets;

  sync target_coordinates;
  sync target_cell_nodes;
  sync target_node_offsets;

  sync cuda_centroids;


  nvtxRangeEnd(NVTX_PR13); // Parallel Region 1

  ///////////////////////////////
  // free application heap memory
  ///////////////////////////////

  nvtxRangeId_t NVTX_FREE3 = nvtxRangeStartA("Free Resources");
  dealloc(source_coordinates_kit);
  dealloc(source_cell_nodes_kit);
  dealloc(source_node_offsets_kit);

  dealloc(target_coordinates_kit);
  dealloc(target_cell_nodes_kit);
  dealloc(target_node_offsets_kit);

  dealloc(candidate_offsets_kit);
  dealloc(candidates_kit);
  nvtxRangeEnd(NVTX_FREE3);

  nvtxRangeEnd(NVTX_PMC3); // Parallel Mesh Creation 2 in UVM


  ///////////////////////////////
  // FOURTH EXECUTION GRAPH
  ///////////////////////////////

  nvtxRangeId_t NVTX_PMC4 = nvtxRangeStartA("Parallel Mesh Creation 4 in UVM");

  nvtxRangeId_t NVTX_PR14 = nvtxRangeStartA("Parallel Region 1");

  spawn source_coordinates1
  {
  nvtxRangeId_t NVTX_SC4 = nvtxRangeStartA("Source Coords");
  create_coordinates_gpu(source_coordinates_kit, nx, ny, x_max, y_max, 0., 0.,
                     "source_coordinates gpu",
                     shuffle ? &shuffle_source_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_SC4);
  }

  spawn source_cell_nodes1
  {
  nvtxRangeId_t NVTX_SC2N4= nvtxRangeStartA("Source C2N");
  create_cell_nodes_gpu(source_cell_nodes_kit, nx, ny, "source_cell_nodes",
                    shuffle ? &shuffle_source_nodes : nullptr,
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_SC2N4);
  }

  source_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "source_node_offsets");
  spawn source_node_offsets1
  {
  nvtxRangeId_t NVTX_SNO4 = nvtxRangeStartA("Source Node Offsets");
  // [[tapir::target("cuda")]]
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    source_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_SNO4);
  }
  sync source_coordinates1;
  sync source_cell_nodes1;
  sync source_node_offsets1;

  // compute centroids
  spawn cuda_centroids41
  {
  nvtxRangeId_t NVTX_CEN4 = nvtxRangeStartA("Centroid on mesh defined with UVM");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument
  double *results = &source_coordinates_kit[2 * n_nodes];
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    &source_coordinates_kit[2 * n_nodes]);

  // check centroids
  if constexpr (LOG_LEVEL > 0)
    for (size_t i = 0; i < n_cells; ++i)
        printf("cell %3zu centroid: %.4f, %.4f\n", i, results[2 * i],
                results[2 * i + 1]);
  if constexpr (CHECK){
    printf("Graph 4 Centroids 1: ");
    check_equal(results, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN4);
  }
  sync cuda_centroids41;

  spawn target_coordinates
  {
  nvtxRangeId_t NVTX_TC4 = nvtxRangeStartA("Target Coords");
  create_coordinates_gpu(target_coordinates_kit, nx, ny, x_max, y_max, shift_x,
                     shift_y, "target_coordinates gpu",
                     shuffle ? &shuffle_target_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_TC4);
  }

  spawn target_cell_nodes
  {
  nvtxRangeId_t NVTX_TC2N4 = nvtxRangeStartA("Target C2N");
  // no need to shuffle the target cells
  create_cell_nodes_gpu(target_cell_nodes_kit, nx, ny, "target_cell_nodes",
                    shuffle ? &shuffle_target_nodes : nullptr, nullptr);
  nvtxRangeEnd(NVTX_TC2N4);
  }

  target_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "target_node_offsets");
  spawn target_node_offsets
  {
  nvtxRangeId_t NVTX_TNO4 = nvtxRangeStartA("Target Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    target_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_TNO4);
  }

  spawn create_candidate_offsets
  {
  nvtxRangeId_t NVTX_CANDO4 = nvtxRangeStartA("Candidates and Offsets");
  nvtxRangeId_t NVTX_CO4 = nvtxRangeStartA("Candidate Offsets");
  create_candidate_offsets(candidate_offsets_kit, nx, ny);
  nvtxRangeEnd(NVTX_CO4);
  nvtxRangeId_t NVTX_CAN4 = nvtxRangeStartA("Candidates");
  create_candidates_gpu(candidates_kit, candidate_offsets_kit[n_cells], nx, ny, candidate_offsets_kit,
                    "create candidates ",
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_CAN4);
  nvtxRangeEnd(NVTX_CANDO4); // Candidates and offsets
  }

  // compute centroids
  spawn cuda_centroids42{
  nvtxRangeId_t NVTX_CEN24 = nvtxRangeStartA("Concurrent Centroid");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument
  centroids_kit1 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit1);

  // check centroids
  if constexpr (LOG_LEVEL > 0)
    for (size_t i = 0; i < n_cells; ++i)
        printf("cell %3zu centroid: %.4f, %.4f\n", i, centroids_kit1[2 * i],
                centroids_kit1[2 * i + 1]);
  if constexpr (CHECK){
    printf("Graph 4 Centroids 2: ");
    check_equal(centroids_kit1, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN24);
  }


  sync target_coordinates;
  sync target_cell_nodes;
  sync target_node_offsets;
  sync create_candidate_offsets;
  sync cuda_centroids42;


  nvtxRangeEnd(NVTX_PR14); // Parallel Region 1

  ///////////////////////////////
  // free application heap memory
  ///////////////////////////////

  nvtxRangeId_t NVTX_FREE4 = nvtxRangeStartA("Free Resources");
  dealloc(source_coordinates_kit);
  dealloc(source_cell_nodes_kit);
  dealloc(source_node_offsets_kit);

  dealloc(target_coordinates_kit);
  dealloc(target_cell_nodes_kit);
  dealloc(target_node_offsets_kit);

  dealloc(candidate_offsets_kit);
  dealloc(candidates_kit);
  dealloc(centroids_kit1);
  nvtxRangeEnd(NVTX_FREE4);

  nvtxRangeEnd(NVTX_PMC4); // Parallel Mesh Creation 2 in UVM


  ///////////////////////////////
  // FIFTH EXECUTION GRAPH
  ///////////////////////////////

  nvtxRangeId_t NVTX_PMC5= nvtxRangeStartA("Parallel Mesh Creation 5 in UVM");

  nvtxRangeId_t NVTX_PR15 = nvtxRangeStartA("Parallel Region 1");

  nvtxRangeId_t NVTX_SS5 = nvtxRangeStartA("Source Stuff");

  spawn source_coordinates5
  {
  nvtxRangeId_t NVTX_SC5 = nvtxRangeStartA("Source Coords");
  create_coordinates_gpu(source_coordinates_kit, nx, ny, x_max, y_max, 0., 0.,
                     "source_coordinates gpu",
                     shuffle ? &shuffle_source_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_SC5);
  }

  spawn source_cell_nodes5
  {
  nvtxRangeId_t NVTX_SC2N5= nvtxRangeStartA("Source C2N");
  create_cell_nodes_gpu(source_cell_nodes_kit, nx, ny, "source_cell_nodes",
                    shuffle ? &shuffle_source_nodes : nullptr,
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_SC2N5);
  }

  source_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "source_node_offsets");
  spawn source_node_offsets5
  {
  nvtxRangeId_t NVTX_SNO5 = nvtxRangeStartA("Source Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    source_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_SNO5);
  }
  sync source_coordinates5;
  sync source_cell_nodes5;
  sync source_node_offsets5;

  nvtxRangeId_t NVTX_AC5 = nvtxRangeStartA("All Centroids");
  //compute centroids
  spawn cuda_centroids51
  {
  nvtxRangeId_t NVTX_CEN5 = nvtxRangeStartA("Centroid on mesh defined with UVM");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument
  double *results = &source_coordinates_kit[2 * n_nodes];
  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    &source_coordinates_kit[2 * n_nodes]);

  // check centroids
  if constexpr (LOG_LEVEL > 0)
    for (size_t i = 0; i < n_cells; ++i)
        printf("cell %3zu centroid: %.4f, %.4f\n", i, results[2 * i],
                results[2 * i + 1]);
  if constexpr (CHECK){
    printf("Graph 5 Centroids 1: ");
    check_equal(results, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN5);
  }

  // compute centroids
  centroids_kit1 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids1x{
  nvtxRangeId_t NVTX_CEN15 = nvtxRangeStartA("Concurrent Centroid1");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit1);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 2: ");
    check_equal(centroids_kit1, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN15);
  }

  // compute centroids
  centroids_kit2 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids2x{
  nvtxRangeId_t NVTX_CEN25 = nvtxRangeStartA("Concurrent Centroid2");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit2);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 3: ");
    check_equal(centroids_kit2, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN25);
  }

  // compute centroids
  centroids_kit3 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids3x{
  nvtxRangeId_t NVTX_CEN35 = nvtxRangeStartA("Concurrent Centroid3");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit3);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 4: ");
    check_equal(centroids_kit3, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN35);
  }

  // compute centroids
  centroids_kit4 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids4x{
  nvtxRangeId_t NVTX_CEN45 = nvtxRangeStartA("Concurrent Centroid4");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit4);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 5: ");
    check_equal(centroids_kit4, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN45);
  }

  // compute centroids
  centroids_kit5 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids5x{
  nvtxRangeId_t NVTX_CEN55 = nvtxRangeStartA("Concurrent Centroid5");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit5);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 6: ");
    check_equal(centroids_kit5, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN55);
  }

  // compute centroids
  centroids_kit6 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids6x{
  nvtxRangeId_t NVTX_CEN65 = nvtxRangeStartA("Concurrent Centroid6");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit6);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 7: ");
    check_equal(centroids_kit6, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN65);
  }

  // compute centroids
  centroids_kit7 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids7x{
  nvtxRangeId_t NVTX_CEN75 = nvtxRangeStartA("Concurrent Centroid7");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit7);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 8: ");
    check_equal(centroids_kit7, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN75);
  }

  // compute centroids
  centroids_kit8 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids8x{
  nvtxRangeId_t NVTX_CEN85 = nvtxRangeStartA("Concurrent Centroid8");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit8);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 9: ");
    check_equal(centroids_kit8, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN85);
  }

  // compute centroids
  centroids_kit9 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids9x{
  nvtxRangeId_t NVTX_CEN95 = nvtxRangeStartA("Concurrent Centroid9");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit9);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 9: ");
    check_equal(centroids_kit9, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN95);
  }

  // compute centroids
  centroids_kit10 = allocate<double>(KITSUNE, 2 * n_cells, "explicit centroids");
  spawn cuda_centroids10x{
  nvtxRangeId_t NVTX_CEN105 = nvtxRangeStartA("Concurrent Centroid10");

  if constexpr (LOG_LEVEL > 0)
    printf("\nComputing parallel centroids with Cuda managed memory...\n");
  // raises a runtime KITSUNE WARNING if we use in the centroid kernel as the
  // final argument

  [[tapir::target("cuda")]]
  forall(size_t i = 0; i < n_cells; ++i) serial::centroid(
    i, source_node_offsets_kit, source_cell_nodes_kit, source_coordinates_kit,
    centroids_kit10);

  if constexpr (CHECK){
    printf("Graph 5 Centroids 10: ");
    check_equal(centroids_kit10, source_centroids, 2 * n_cells);
  }
  nvtxRangeEnd(NVTX_CEN105);
  }

  sync cuda_centroids51;
  sync cuda_centroids1x;
  sync cuda_centroids2x;
  sync cuda_centroids3x;
  sync cuda_centroids4x;
  sync cuda_centroids5x;
  sync cuda_centroids6x;
  sync cuda_centroids7x;
  sync cuda_centroids8x;
  sync cuda_centroids9x;
  sync cuda_centroids10x;

  nvtxRangeEnd(NVTX_AC5);

  nvtxRangeEnd(NVTX_SS5);

  spawn target_coordinates
  {
  nvtxRangeId_t NVTX_TC5 = nvtxRangeStartA("Target Coords");
  create_coordinates_gpu(target_coordinates_kit, nx, ny, x_max, y_max, shift_x,
                     shift_y, "target_coordinates gpu",
                     shuffle ? &shuffle_target_nodes : nullptr,
                     2 * nx * ny);
  nvtxRangeEnd(NVTX_TC5);
  }

  spawn target_cell_nodes
  {
  nvtxRangeId_t NVTX_TC2N5 = nvtxRangeStartA("Target C2N");
  // no need to shuffle the target cells
  create_cell_nodes_gpu(target_cell_nodes_kit, nx, ny, "target_cell_nodes",
                    shuffle ? &shuffle_target_nodes : nullptr, nullptr);
  nvtxRangeEnd(NVTX_TC2N5);
  }

  target_node_offsets_kit = allocate<size_t>(KITSUNE, nx * ny + 1, "target_node_offsets");
  spawn target_node_offsets
  {
  nvtxRangeId_t NVTX_TNO5 = nvtxRangeStartA("Target Node Offsets");
  forall (size_t i = 0; i < nx * ny + 1; ++i)
    target_node_offsets_kit[i] = 4 * i;
  nvtxRangeEnd(NVTX_TNO5);
  }

  spawn create_candidate_offsets
  {
  nvtxRangeId_t NVTX_CANDO5 = nvtxRangeStartA("Candidates and Offsets");
  nvtxRangeId_t NVTX_CO5 = nvtxRangeStartA("Candidate Offsets");
  create_candidate_offsets(candidate_offsets_kit, nx, ny);
  nvtxRangeEnd(NVTX_CO5);
  nvtxRangeId_t NVTX_CAN5 = nvtxRangeStartA("Candidates");
  create_candidates_gpu(candidates_kit, candidate_offsets_kit[n_cells], nx, ny, candidate_offsets_kit,
                    "create candidates ",
                    shuffle ? &shuffle_source_cells : nullptr);
  nvtxRangeEnd(NVTX_CAN5);
  nvtxRangeEnd(NVTX_CANDO5); // Candidates and offsets
  }

  sync create_candidate_offsets;

  sync target_coordinates;
  sync target_cell_nodes;
  sync target_node_offsets;



  nvtxRangeEnd(NVTX_PR15); // Parallel Region 1

  // At this point we know the number of candidates so we can allocate. All the syncs
  // needed to complete in create_meshes_gpu in order for us to do the allocation here.
  // This isn't really what we want, but is the best we can do at the moment.
  nvtxRangeId_t NVTX_PR23 = nvtxRangeStartA("Parallel Region 2");
  nvtxRangeEnd(NVTX_PR23); // Parallel Region 2

  /////////////////////////////
  // free application heap memory
  /////////////////////////////

  nvtxRangeId_t NVTX_FREE5 = nvtxRangeStartA("Free Resources");
  dealloc(source_coordinates_kit);
  dealloc(source_cell_nodes_kit);
  dealloc(source_node_offsets_kit);

  dealloc(target_coordinates_kit);
  dealloc(target_cell_nodes_kit);
  dealloc(target_node_offsets_kit);

  dealloc(candidate_offsets_kit);
  dealloc(candidates_kit);
  nvtxRangeEnd(NVTX_FREE5);

  nvtxRangeEnd(NVTX_PMC5); // Parallel Mesh Creation 2 in UVM

  // free heap memory
  nvtxRangeId_t NVTX_SERIALFREE2 = nvtxRangeStartA("Free Serial Resources");
  free(source_coordinates);
  free(source_cell_nodes);
  free(source_node_offsets);
  free(target_coordinates);
  free(target_cell_nodes);
  free(target_node_offsets);
  free(candidates);
  free(candidate_offsets);
  free(source_centroids);
  nvtxRangeEnd(NVTX_SERIALFREE2);


  return 0;
}
