// Declarations for getopt. We could just include unistd.h, but we want to keep
// the main test code simple and not include the header there. We could include
// it here, but since including a file such as this is already rather ugly, I
// don't see why it should be made even worse.
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int getopt(int argc, char *argv[], const char *optstring);

extern char *optarg;
extern int optind, opterr, optopt;

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#include "common.h"

static const char *usage = "euler3d [OPTIONS] <domain> [iters]\n"
                           "Try `euler3d -h` for more information";

// clang-format off
static const char *help =
  "<<<<Short description of the euler3d benchmark here>>>>\n"
  "\n"
  "    euler3d [OPTIONS] <domain> [iters]\n"
  "\n"
  "OPTIONS\n"
  "\n"
  "    -c <FILE>  The path to the reference output file on a CPU. If srad was\n"
  "               built with a GPU tapir target, this option is ignored\n"
  "    -g <FILE>  The path to the reference output file on a GPU. If srad was\n"
  "               built with a CPU tapir target, this option is ignored\n"
  "    -h         Print help and exit\n"
  "    -j         Print the recorded times in JSON\n"
  "    -t <N>     Number of threads per block on the GPU. This is only used\n"
  "               by the .cu and .hip implementations of this benchmark [256]\n"
  "\n"
  "ARGUMENTS\n"
  "\n"
  "    domain     File containing the domain geometry\n"
  "\n"
  "    iters      Number of iterations [4000]\n";
// clang-format on

// Show the times in JSON. Yes, it's global. Too bad.
static bool showJSON = false;

static void parseCommandLineInto(int argc, char *argv[],
                                 std::string &domainFile, int &iterations,
                                 std::string &outFile, std::string &cpuRefFile,
                                 std::string &gpuRefFile,
                                 unsigned *tpb = nullptr) {
  iterations = 4000;
  cpuRefFile = "";
  gpuRefFile = "";
  if (tpb) {
    // This is loosely for consistency with the launch parameters from kitsune.
    *tpb = 256;
  }
  outFile = std::filesystem::path(argv[0]).filename().string() + ".dat";

  int flag;
  while ((flag = getopt(argc, argv, "c:g:hjt:")) != -1) {
    switch (flag) {
    case 'c':
      cpuRefFile = optarg;
      break;
    case 'g':
      gpuRefFile = optarg;
      break;
    case 'h':
      std::cout << help << std::endl;
      exit(0);
    case 'j':
      showJSON = true;
      break;
    case 't':
      if (tpb) {
        *tpb = std::stoi(optarg);
      } else {
        std::cout << "-------------------------------------------------\n";
        std::cout << " WARNING: Ignoring threads-per-block option (-t) \n";
        std::cout << "-------------------------------------------------\n";
      }
      break;
    default:
      std::cout << "ERROR: Unknown option -'" << (char)optopt << "'\n";
      std::cout << "\n";
      std::cout << usage << "\n";
      exit(1);
    }
  }

  char **args = &argv[optind];
  int argn = argc - optind;
  if (argn == 0) {
    std::cout << usage << std::endl;
    exit(1);
  }
  if (argn > 0)
    domainFile = args[0];
  if (argn > 1)
    iterations = std::stoi(args[1]);
  if (argn > 2) {
    std::cout << usage << std::endl;
    exit(1);
  }
}

template <typename FArr, typename IArr>
static void alloc(FArr &ff_variable, FArr &areas,
                  IArr &elements_surrounding_elements, FArr &normals,
                  FArr &variables, FArr &old_variables, FArr &fluxes,
                  FArr &step_factors, int nelr) {
#if defined(__KOKKOS__)
  areas = FArr("areas", nelr);
  elements_surrounding_elements =
      IArr("elements_surrounding_elements", nelr * NNB);
  normals = FArr("normals", NDIM * NNB * nelr);
  variables = FArr("variables", nelr * NVAR);
  old_variables = FArr("old_variables", nelr * NVAR);
  fluxes = FArr("fluxes", nelr * NVAR);
  step_factors = FArr("step_factors", nelr);
#elif defined(__CUDACC__)
  cudaMallocManaged(&areas, nelr * sizeof(float));
  cudaMallocManaged(&elements_surrounding_elements, nelr * NNB * sizeof(int));
  cudaMallocManaged(&normals, nelr * NDIM * NNB * sizeof(float));
  cudaMallocManaged(&variables, nelr * NVAR * sizeof(float));
  cudaMallocManaged(&old_variables, nelr * NVAR * sizeof(float));
  cudaMallocManaged(&fluxes, nelr * NVAR * sizeof(float));
  cudaMallocManaged(&step_factors, nelr * sizeof(float));
#elif defined(__HIP__)
  (void)hipMallocManaged(&areas, nelr * sizeof(float));
  (void)hipMallocManaged(&elements_surrounding_elements,
                         nelr * NNB * sizeof(int));
  (void)hipMallocManaged(&normals, nelr * NDIM * NNB * sizeof(float));
  (void)hipMallocManaged(&variables, nelr * NVAR * sizeof(float));
  (void)hipMallocManaged(&old_variables, nelr * NVAR * sizeof(float));
  (void)hipMallocManaged(&fluxes, nelr * NVAR * sizeof(float));
  (void)hipMallocManaged(&step_factors, nelr * sizeof(float));
#elif defined(_OPENMP)
  areas = (float*)malloc(nelr * sizeof(float));
  elements_surrounding_elements = (int*)malloc(nelr * NNB * sizeof(int));
  normals = (float*)malloc(nelr * NDIM * NNB * sizeof(float));
  variables = (float*)malloc(nelr * NVAR * sizeof(float));
  old_variables = (float*)malloc(nelr * NVAR * sizeof(float));
  fluxes = (float*)malloc(nelr * NVAR * sizeof(float));
  step_factors = (float*)malloc(nelr * sizeof(float));
#else
  areas.alloc(nelr);
  elements_surrounding_elements.alloc(nelr * NNB);
  normals.alloc(NDIM * NNB * nelr);
  variables.alloc(nelr * NVAR);
  old_variables.alloc(nelr * NVAR);
  fluxes.alloc(nelr * NVAR);
  step_factors.alloc(nelr);
#endif
}

template <typename FArr, typename IArr>
static void dealloc(FArr &ff_variable, FArr &areas,
                    IArr &elements_surrounding_elements, FArr &normals,
                    FArr &variables, FArr &old_variables, FArr &fluxes,
                    FArr &step_factors) {
#if defined(__KOKKOS__)
#elif defined(__CUDACC__)
  cudaFree(ff_variable);
  cudaFree(areas);
  cudaFree(elements_surrounding_elements);
  cudaFree(normals);
  cudaFree(variables);
  cudaFree(old_variables);
  cudaFree(fluxes);
  cudaFree(step_factors);
#elif defined(__HIP__)
  (void)hipFree(ff_variable);
  (void)hipFree(areas);
  (void)hipFree(elements_surrounding_elements);
  (void)hipFree(normals);
  (void)hipFree(variables);
  (void)hipFree(old_variables);
  (void)hipFree(fluxes);
  (void)hipFree(step_factors);
#elif defined(_OPENMP)
  free(ff_variable);
  free(areas);
  free(elements_surrounding_elements);
  free(normals);
  free(variables);
  free(old_variables);
  free(fluxes);
  free(step_factors);
#else
  ff_variable.free();
  areas.free();
  elements_surrounding_elements.free();
  normals.free();
  variables.free();
  old_variables.free();
  fluxes.free();
  step_factors.free();
#endif
}

static void header(const std::string &label, const std::string &domainFile,
                   int iterations) {
  std::cout << "\n";
  std::cout << "---- euler3d benchmark (" << label << ") ----\n\n"
            << "  Input file : " << domainFile << "\n"
            << "  Iterations : " << iterations << ".\n\n";

  std::cout
      << "  Reading input data, allocating arrays, initializing data, etc..."
      << std::flush;
}

template <typename FArr, typename IArr>
static void
read_domain(FArr &ff_variable, FArr &areas, IArr &elements_surrounding_elements,
            FArr &normals, FArr &variables, FArr &old_variables, FArr &fluxes,
            FArr &step_factors, Float3 &ff_flux_contribution_momentum_x,
            Float3 &ff_flux_contribution_momentum_y,
            Float3 &ff_flux_contribution_momentum_z,
            Float3 &ff_flux_contribution_density_energy, int &nel, int &nelr,
            const std::string &domainFile) {
// these need to be computed the first time in order to compute time step
#if defined(__CUDACC__)
  cudaMallocManaged(&ff_variable, NVAR * sizeof(float));
#elif defined(__HIP__)
  (void)hipMallocManaged(&ff_variable, NVAR * sizeof(float));
#elif defined(_OPENMP)
  ff_variable = (float*)malloc(NVAR * sizeof(float));
#else
  ff_variable.alloc(NVAR);
#endif

  // set far field conditions
  const float angle_of_attack =
      float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);

  ff_variable[VAR_DENSITY] = float(1.4);

  float ff_pressure = float(1.0f);
  float ff_speed_of_sound =
      sqrtf(GAMMA * ff_pressure / ff_variable[VAR_DENSITY]);
  float ff_speed = float(ff_mach) * ff_speed_of_sound;

  Float3 ff_velocity;
  ff_velocity.x = ff_speed * float(cos((float)angle_of_attack));
  ff_velocity.y = ff_speed * float(sin((float)angle_of_attack));
  ff_velocity.z = 0.0f;

  ff_variable[VAR_MOMENTUM + 0] = ff_variable[VAR_DENSITY] * ff_velocity.x;
  ff_variable[VAR_MOMENTUM + 1] = ff_variable[VAR_DENSITY] * ff_velocity.y;
  ff_variable[VAR_MOMENTUM + 2] = ff_variable[VAR_DENSITY] * ff_velocity.z;

  ff_variable[VAR_DENSITY_ENERGY] =
      ff_variable[VAR_DENSITY] * (float(0.5f) * (ff_speed * ff_speed)) +
      (ff_pressure / float(GAMMA - 1.0f));

  Float3 ff_momentum;
  ff_momentum.x = ff_variable[VAR_MOMENTUM + 0];
  ff_momentum.y = ff_variable[VAR_MOMENTUM + 1];
  ff_momentum.z = ff_variable[VAR_MOMENTUM + 2];
  compute_flux_contribution(
      ff_variable[VAR_DENSITY], ff_momentum, ff_variable[VAR_DENSITY_ENERGY],
      ff_pressure, ff_velocity, ff_flux_contribution_momentum_x,
      ff_flux_contribution_momentum_y, ff_flux_contribution_momentum_z,
      ff_flux_contribution_density_energy);

  // read in domain geometry
  std::ifstream file(domainFile);
  file >> nel;
  nelr =
      block_length * ((nel / block_length) + std::min(1, nel % block_length));

  alloc(ff_variable, areas, elements_surrounding_elements, normals, variables,
        old_variables, fluxes, step_factors, nelr);

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

  std::cout << "  done.\n\n";
  std::cout << "  Starting benchmark ... " << std::flush;
}

// FIXME: We may be able to lower the tolerance once Kitsune correctly handles
// the fp-contract option.
static constexpr float epsilon = 1e-5;

/// Check that the contents of the output file matches the expected values.
static bool check(const std::string &out_file, const std::string &check_file) {
  bool matches = true;
  char ec, ac;

  FILE *fa = fopen(out_file.c_str(), "rb");
  FILE *fe = fopen(check_file.c_str(), "rb");
  float *av = nullptr;
  float *ev = nullptr;

  int enel, anel;
  int enelr, anelr;
  fread(&enel, sizeof(int), 1, fe);
  fread(&enelr, sizeof(int), 1, fe);
  fread(&anel, sizeof(int), 1, fa);
  fread(&anelr, sizeof(int), 1, fa);
  if (anel != enel or anelr != enelr) {
    matches = false;
    goto cleanup;
  }

  ev = (float *)malloc(sizeof(float) * enel);
  av = (float *)malloc(sizeof(float) * anel);

  fread(ev, sizeof(float), enel, fe);
  fread(av, sizeof(float), anel, fa);
  if (checkRelErr(av, ev, enel, epsilon)) {
    matches = false;
    goto cleanup;
  }

  for (int j = 0; j < NDIM; ++j) {
    fread(ev, sizeof(float), enel, fe);
    fread(av, sizeof(float), anel, fa);
    if (checkRelErr(av, ev, enel, epsilon)) {
      matches = false;
      goto cleanup;
    }
  }

  fread(ev, sizeof(float), enel, fe);
  fread(av, sizeof(float), anel, fa);
  if (checkRelErr(av, ev, enel, epsilon)) {
    matches = false;
    goto cleanup;
  }

  if (fgetc(fe) != EOF or fgetc(fa) != EOF) {
    matches = false;
    goto cleanup;
  }

cleanup:
  free(ev);
  free(av);
  fclose(fe);
  fclose(fa);

  return true;
}

template <typename FArr>
static void save(const std::string &out_file, FArr &variables, int nel,
                 int nelr) {
  FILE *fp = fopen(out_file.c_str(), "wb");
  char type;

  fwrite(&nel, sizeof(int), 1, fp);
  fwrite(&nelr, sizeof(int), 1, fp);
  fwrite(&variables[VAR_DENSITY * nelr], sizeof(float), nel, fp);
  for (int j = 0; j < NDIM; j++)
    fwrite(&variables[(VAR_MOMENTUM + j) * nelr], sizeof(float), nel, fp);
  fwrite(&variables[VAR_DENSITY_ENERGY * nelr], sizeof(float), nel, fp);

  fclose(fp);
}

template <typename FArr, typename IArr>
static bool footer(const TimerGroup &tg, FArr &ff_variable, FArr &areas,
                   IArr &elements_surrounding_elements, FArr &normals,
                   FArr &variables, FArr &old_variables, FArr &fluxes,
                   FArr &step_factors, int nel, int nelr,
                   const std::string &outFile, const std::string &cpuRefFile,
                   const std::string &gpuRefFile) {
  std::cout << "done\n\n";
  tg.prettyTimes(std::cout, 4);
  std::cout << "----\n\n";

  save(outFile, variables, nel, nelr);

  std::string checkFile = cpuRefFile;
#if defined(__CUDACC__) || defined(__HIP__) || defined(__KOKKOS__)
  checkFile = gpuRefFile;
#elif defined(_OPENMP)
#else
  std::string tt = __kitsune_tt__;
  if (tt == "cuda" or tt == "hip")
    checkFile = gpuRefFile;
#endif // !__CUDACC__ && !__HIP__

  bool ok = true;
  if (!checkFile.size()) {
    std::cout << "--------------------------------------------------\n";
    std::cout << " WARNING: Not checking output (no reference file) \n";
    std::cout << "--------------------------------------------------\n";
  } else {
    char buf[16];
    snprintf(buf, 16, "%.6g", epsilon);
    std::cout << "\n  Checking with epsilon = " << buf << " ..." << std::flush;
    ok = check(outFile, checkFile);
    if (!ok)
      std::cout << "  FAIL!\n\n";
    else
      std::cout << "  pass\n\n";
  }

  dealloc(ff_variable, areas, elements_surrounding_elements, normals, variables,
          old_variables, fluxes, step_factors);
  if (showJSON)
    tg.json(std::cout);

  return ok;
}
