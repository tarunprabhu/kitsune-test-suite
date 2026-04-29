
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
  "    -c <FILE>  Path to the reference output file\n"
  "    -h         Print help and exit\n"
  "\n"
  "ARGUMENTS\n"
  "\n"
  "    domain     File containing the domain geometry\n"
  "\n"
  "    iters      Number of iterations [20]\n";
// clang-format on

[[clang::noinline]]
static std::string getOutFile(char *argv[]) {
  return std::filesystem::path(argv[0]).filename().string() + ".dat";
}

[[clang::noinline]]
static void setup(float *[[kitsune::mobile]] & ff_variable,
                  float *[[kitsune::mobile]] & areas,
                  int *[[kitsune::mobile]] & elements_surrounding_elements,
                  float *[[kitsune::mobile]] & normals,
                  float *[[kitsune::mobile]] & variables,
                  float *[[kitsune::mobile]] & old_variables,
                  float *[[kitsune::mobile]] & fluxes,
                  float *[[kitsune::mobile]] & step_factors,
                  Float3 &ff_flux_contribution_momentum_x,
                  Float3 &ff_flux_contribution_momentum_y,
                  Float3 &ff_flux_contribution_momentum_z,
                  Float3 &ff_flux_contribution_density_energy, int &nel,
                  int &nelr, const std::string &domainFile) {
  using FloatMobilePtr = float *[[kitsune::mobile]];
  using IntMobilePtr = int *[[kitsune::mobile]];

  // these need to be computed the first time in order to compute time step
  ff_variable = (FloatMobilePtr)kitsune_mobile_alloc(NVAR * sizeof(float));

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

  areas = (FloatMobilePtr)kitsune_mobile_alloc(nelr * sizeof(float));
  elements_surrounding_elements =
      (IntMobilePtr)kitsune_mobile_alloc(nelr * NNB * sizeof(int));
  normals =
      (FloatMobilePtr)kitsune_mobile_alloc(nelr * NDIM * NNB * sizeof(float));
  variables = (FloatMobilePtr)kitsune_mobile_alloc(nelr * NVAR * sizeof(float));
  old_variables =
      (FloatMobilePtr)kitsune_mobile_alloc(nelr * NVAR * sizeof(float));
  fluxes = (FloatMobilePtr)kitsune_mobile_alloc(nelr * NVAR * sizeof(float));
  step_factors = (FloatMobilePtr)kitsune_mobile_alloc(nelr * sizeof(float));

  // read in data
  for (int i = 0; i < nel; i++) {
    file >> (float &)areas[i];
    for (int j = 0; j < NNB; j++) {
      file >> (int &)elements_surrounding_elements[i + j * nelr];
      if (elements_surrounding_elements[i + j * nelr] < 0)
        elements_surrounding_elements[i + j * nelr] = -1;
      // it's coming in with Fortran numbering
      elements_surrounding_elements[i + j * nelr]--;

      for (int k = 0; k < NDIM; k++) {
        file >> (float &)normals[i + (j + k * NNB) * nelr];
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
}

[[clang::noinline]]
static void teardown(float *[[kitsune::mobile]] ff_variable,
                     float *[[kitsune::mobile]] areas,
                     int *[[kitsune::mobile]] elements_surrounding_elements,
                     float *[[kitsune::mobile]] normals,
                     float *[[kitsune::mobile]] variables,
                     float *[[kitsune::mobile]] old_variables,
                     float *[[kitsune::mobile]] fluxes,
                     float *[[kitsune::mobile]] step_factors) {
  kitsune_mobile_free(ff_variable);
  kitsune_mobile_free(areas);
  kitsune_mobile_free(elements_surrounding_elements);
  kitsune_mobile_free(normals);
  kitsune_mobile_free(variables);
  kitsune_mobile_free(old_variables);
  kitsune_mobile_free(fluxes);
  kitsune_mobile_free(step_factors);
}

[[clang::noinline]]
static void save(const std::string &outFile, const float *variables, int nel,
                 int nelr) {
  if (FILE *fp = fopen(outFile.c_str(), "wb")) {
    fwrite(&nel, sizeof(int), 1, fp);
    fwrite(&nelr, sizeof(int), 1, fp);
    fwrite((void *)&variables[VAR_DENSITY * nelr], sizeof(float), nel, fp);
    for (int j = 0; j < NDIM; j++)
      fwrite(&variables[(VAR_MOMENTUM + j) * nelr], sizeof(float), nel, fp);
    fwrite(&variables[VAR_DENSITY_ENERGY * nelr], sizeof(float), nel, fp);

    fclose(fp);
  }
}

static constexpr float epsilon = 1e-6;

static float relErr(float actual, float expected) {
  return std::abs(actual - expected) / (std::abs(expected) + 1);
}

static bool checkRelErr(float actual, float expected, float epsilon) {
  return relErr(actual, expected) > epsilon;
}

static size_t checkRelErr(float *actual, float *expected, size_t n,
                          float epsilon) {
  size_t errors = 0;
  for (size_t i = 0; i < n; ++i)
    if (checkRelErr(actual[i], expected[i], epsilon)) {
      if (errors < 10)
        printf("%ld: %.6f, %.6f [%.6f]\n", i, actual[i], expected[i],
               relErr(actual[i], expected[i]));
      ++errors;
    }
  return errors;
}

/// Check that the contents of the output file matches the expected values.
[[clang::noinline]]
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

  return matches;
}

[[clang::noinline]]
static int report(bool ok) {
  if (!ok)
    printf("FAIL\n");
  else
    printf("PASS\n");

  // ok will be true (non-zero) on success. But the OS needs 0 to indicate
  // success.
  return !ok;
}

[[clang::noinline]]
static void parseCommandLineArgs(int argc, char *argv[], int &iterations,
                                 std::string &domainFile,
                                 std::string &checkFile, std::string &outFile) {
  iterations = 20;
  checkFile = "";
  outFile = getOutFile(argv);

  int flag;
  while ((flag = getopt(argc, argv, "c:h")) != -1) {
    switch (flag) {
    case 'c':
      checkFile = optarg;
      break;
    case 'h':
      printf("%s\n", help);
      exit(0);
    default:
      printf("ERROR: Unknown option '-%c'\n\n", optopt);
      printf("%s\n", usage);
      exit(1);
    }
  }

  char **args = &argv[optind];
  int argn = argc - optind;
  if (argn == 0) {
    printf("%s\n", usage);
    exit(1);
  }
  if (argn > 0)
    domainFile = args[0];
  if (argn > 1)
    iterations = atoi(args[1]);
  if (argn > 2) {
    printf("%s\n", usage);
    exit(1);
  }
}
