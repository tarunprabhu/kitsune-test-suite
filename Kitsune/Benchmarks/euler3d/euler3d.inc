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
                                 std::string &cpuRefFile,
                                 std::string &gpuRefFile,
                                 unsigned *tpb = nullptr) {
  iterations = 4000;
  cpuRefFile = "";
  gpuRefFile = "";
  if (tpb) {
    // This is loosely for consistency with the launch parameters from kitsune.
    *tpb = 256;
  }

  char flag;
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
static void alloc(FArr &I, FArr &J, FArr &c, IArr &iN, IArr &iS, IArr &jE,
                  IArr &jW, FArr &dN, FArr &dS, FArr &dW, FArr &dE, int rows,
                  int cols) {
  int size_I = rows * cols;
#if defined(IS_CUDA)
  cudaMallocManaged(&I, size_I * sizeof(float));
  cudaMallocManaged(&J, size_I * sizeof(float));
  cudaMallocManaged(&c, size_I * sizeof(float));
  cudaMallocManaged(&iN, rows * sizeof(int));
  cudaMallocManaged(&iS, rows * sizeof(int));
  cudaMallocManaged(&jW, cols * sizeof(int));
  cudaMallocManaged(&jE, cols * sizeof(int));
  cudaMallocManaged(&dN, size_I * sizeof(float));
  cudaMallocManaged(&dS, size_I * sizeof(float));
  cudaMallocManaged(&dW, size_I * sizeof(float));
  cudaMallocManaged(&dE, size_I * sizeof(float));
#elif defined(IS_HIP)
  hipMallocManaged(&I, size_I * sizeof(float));
  hipMallocManaged(&J, size_I * sizeof(float));
  hipMallocManaged(&c, size_I * sizeof(float));
  hipMallocManaged(&iN, rows * sizeof(int));
  hipMallocManaged(&iS, rows * sizeof(int));
  hipMallocManaged(&jW, cols * sizeof(int));
  hipMallocManaged(&jE, cols * sizeof(int));
  hipMallocManaged(&dN, size_I * sizeof(float));
  hipMallocManaged(&dS, size_I * sizeof(float));
  hipMallocManaged(&dW, size_I * sizeof(float));
  hipMallocManaged(&dE, size_I * sizeof(float));
#else
  I.alloc(size_I);
  J.alloc(size_I);
  c.alloc(size_I);
  iN.alloc(rows);
  iS.alloc(rows);
  jW.alloc(cols);
  jE.alloc(cols);
  dN.alloc(size_I);
  dS.alloc(size_I);
  dW.alloc(size_I);
  dE.alloc(size_I);
#endif // CPU
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
#if defined(IS_CUDA)
  cudaMallocManaged(&ff_variable, NVAR * sizeof(float));
#elif defined(IS_HIP)
  hipMallocManaged(&ff_variable, NVAR * sizeof(float));
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

#if defined(IS_CUDA)
  cudaMallocManaged(&areas, nelr * sizeof(float));
  cudaMallocManaged(&elements_surrounding_elements, nelr * NNB * sizeof(int));
  cudaMallocManaged(&normals, nelr * NDIM * NNB * sizeof(float));
  cudaMallocManaged(&variables, nelr * NVAR * sizeof(float));
  cudaMallocManaged(&old_variables, nelr * NVAR * sizeof(float));
  cudaMallocManaged(&fluxes, nelr * NVAR * sizeof(float));
  cudaMallocManaged(&step_factors, nelr * sizeof(float));
#elif defined(IS_HIP)
  hipMallocManaged(&areas, nelr * sizeof(float));
  hipMallocManaged(&elements_surrounding_elements, nelr * NNB * sizeof(int));
  hipMallocManaged(&normals, nelr * NDIM * NNB * sizeof(float));
  hipMallocManaged(&variables, nelr * NVAR * sizeof(float));
  hipMallocManaged(&old_variables, nelr * NVAR * sizeof(float));
  hipMallocManaged(&fluxes, nelr * NVAR * sizeof(float));
  hipMallocManaged(&step_factors, nelr * sizeof(float));
#else
  areas.alloc(nelr);
  elements_surrounding_elements.alloc(nelr * NNB);
  normals.alloc(NDIM * NNB * nelr);
  variables.alloc(nelr * NVAR);
  old_variables.alloc(nelr * NVAR);
  fluxes.alloc(nelr * NVAR);
  step_factors.alloc(nelr);
#endif

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

static bool check(const std::string &out_file, const std::string &check_file) {
  float epsilon = 1e-6;
  char ec, ac;

  FILE *fa = fopen(out_file.c_str(), "rb");
  FILE *fe = fopen(check_file.c_str(), "rb");

  int enel, anel;
  int enelr, anelr;
  fread(&enel, sizeof(int), 1, fe);
  fread(&enelr, sizeof(int), 1, fe);
  fread(&anel, sizeof(int), 1, fa);
  fread(&anelr, sizeof(int), 1, fa);
  if (anel != enel or anelr != enelr)
    return false;

  float *ev = (float *)malloc(sizeof(float) * enel);
  float *av = (float *)malloc(sizeof(float) * anel);

  fread(ev, sizeof(float), enel, fe);
  fread(av, sizeof(float), anel, fa);
  for (int i = 0; i < enel; ++i)
    if (fabs(ev[i] - av[i]) >= epsilon)
      return false;

  for (int j = 0; j < NDIM; ++j) {
    fread(ev, sizeof(float), enel, fe);
    fread(av, sizeof(float), anel, fa);
    for (int i = 0; i < enel; ++i)
      if (fabs(ev[i] - av[i]) >= epsilon)
        return false;
  }

  fread(ev, sizeof(float), enel, fe);
  fread(av, sizeof(float), anel, fa);
  for (int i = 0; i < enel; ++i)
    if (fabs(ev[i] - av[i]) >= epsilon)
      return false;

  if (fgetc(fe) != EOF or fgetc(fa) != EOF)
    return false;

  free(ev);
  free(av);
  fclose(fe);
  fclose(fa);

  return true;
}

template <typename FArr>
static void save(const std::string &out_file, FArr& variables,
                 int nel, int nelr) {
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
static void dealloc(FArr &ff_variable, FArr &areas,
                    IArr &elements_surrounding_elements, FArr &normals,
                    FArr &variables, FArr &old_variables, FArr &fluxes,
                    FArr &step_factors) {
#if defined(IS_CUDA)
  cudaFree(ff_variable);
  cudaFree(areas);
  cudaFree(elements_surrounding_elements);
  cudaFree(normals);
  cudaFree(variables);
  cudaFree(old_variables);
  cudaFree(fluxes);
  cudaFree(step_factors);
#elif defined(IS_HIP)
  hipFree(ff_variable);
  hipFree(areas);
  hipFree(elements_surrounding_elements);
  hipFree(normals);
  hipFree(variables);
  hipFree(old_variables);
  hipFree(fluxes);
  hipFree(step_factors);
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
#if defined(IS_CUDA) || defined(IS_HIP)
  checkFile = gpuRefFile;
#else
  std::string tt = __kitsune_tt__;
  if (tt == "cuda" or tt == "hip")
    checkFile = gpuRefFile;
#endif // !IS_CUDA && !IS_HIP

  bool ok = true;
  if (!checkFile.size()) {
    std::cout << "--------------------------------------------------\n";
    std::cout << " WARNING: Not checking output (no reference file) \n";
    std::cout << "--------------------------------------------------\n";
  } else {
    std::cout << "\n  Checking final result..." << std::flush;
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
