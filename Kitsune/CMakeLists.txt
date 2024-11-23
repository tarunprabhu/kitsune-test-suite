# The Kitsune tests. For performance comparisons, this contains .cu and .hip
# sources as well. Those languages are not set here because the actual compiler
# used would depend on the CUDA (respectively HIP) installations used to build
# Kitsune's cuda (respectively hip) tapir targets (if they are built at all).
# Those languages will, therefore, only be enabled conditionally. Fortran
# support will likely always be optional, so that will also only be enabled
# after querying the kitsune installation using the kit-config utility.
project(kitsune-test-suite
  C CXX)

option(KITSUNE_TEST_TAPIR_TARGETS
  "A space-separated list of the tapir targets to test. If this is not set, \
all Tapir targets enabled in Kitsune will be tested"
  "")

option(KITSUNE_SKIP_TAPIR_TARGETS
  "A string-separated list of tapir targets to skip"
  "")

option(KITSUNE_TEST_KOKKOS
  "Test Kitsune's Kokkos support if it has been enabled in Kitsune"
  ON)

option(KITSUNE_TEST_CUDA_LANG
  "Run the cuda language (.cu) tests when the cuda tapir target is tested"
  OFF)

option(KITSUNE_TEST_HIP_LANG
  "Run the hip language (.hip) tests when the hip tapir target is tested"
  OFF)

# Ideally, the kitsune tests should only be run using Kitsune as the compiler.
# Right now, this is not the easiest thing to do because Kitsune reports its
# own ID as "clang". We could change that, but that may require teaching cmake
# about Kitsune. At some point, we might add something to Kitsune that can be
# used to ensure that it is Kitsune that we are dealing with.

# In principle, Kitsune "could" be built without C/C++ support. But that is
# highly unlikely, so use the installation directory of the compiler to
# determine the location of the kit-config utility.
get_filename_component(COMPILER_BINDIR "${CMAKE_C_COMPILER}" DIRECTORY)

find_program(KIT_CONFIG kit-config REQUIRED
  PATHS ${COMPILER_BINDIR}
  NO_DEFAULT_PATH)

message(STATUS "Kitsune configuration tool: ${KIT_CONFIG}")

set(TEST_CUDA_LANG ON)
set(TEST_HIP_LANG ON)
set(TEST_KOKKOS_MODE ON)
set(TEST_CUDA_TARGET OFF)
set(TEST_HIP_TARGET OFF)
set(TEST_LAMBDA_TARGET OFF)
set(TEST_OMPTASK_TARGET OFF)
set(TEST_OPENCILK_TARGET OFF)
set(TEST_OPENMP_TARGET OFF)
set(TEST_QTHREADS_TARGET OFF)
set(TEST_REALM_TARGET OFF)
set(TEST_SERIAL_TARGET OFF)

execute_process(COMMAND "${KIT_CONFIG}" "--kokkos-mode"
  OUTPUT_VARIABLE KOKKOS_MODE_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# If Kokkos mode has not been enabled in Kitsune, don't test Kokkos at all.
# NOTE: This will also not disable testing of vanilla Kokkos.
if (KOKKOS_MODE_STR STREQUAL "OFF" OR NOT KITSUNE_TEST_KOKKOS)
  set(TEST_KOKKOS_MODE OFF)
endif ()

execute_process(COMMAND "${KIT_CONFIG}" "--tapir-targets"
  OUTPUT_VARIABLE TAPIR_TARGETS_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REPLACE " " ";" ENABLED_TAPIR_TARGETS "${TAPIR_TARGETS_STR}")
string(REPLACE " " ";" SKIP_TAPIR_TARGETS "${KITSUNE_SKIP_TAPIR_TARGETS}")

# If an explicit list of tapir targets to test has not been provided, test all
# the enabled tapir targets. Otherwise, test the explicit list. All tapir
# targets that were explicitly specified must have been enabled.
if (${KITSUNE_TEST_TAPIR_TARGETS} STREQUAL "")
  set(TEST_TAPIR_TARGETS "${ENABLED_TAPIR_TARGETS}")
else ()
  string(REPLACE " " ";" TEST_TAPIR_TARGETS "${KITSUNE_TEST_TAPIR_TARGETS}")
  foreach (tapir_target IN ${TEST_TAPIR_TARGETS})
    if (NOT ${tapir_target} IN_LIST ${ENABLED_TAPIR_TARGETS})
      message(FATAL_ERROR "'${tapir_target}' has not been enabled")
    endif ()
  endforeach ()
endif ()

# Filter out any tapir targets to skip.
foreach (tapir_target IN ${SKIP_TAPIR_TARGETS})
  list(REMOVE_ITEM ${TEST_TAPIR_TARGETS} ${tapir_target})
endforeach ()

foreach (tapir_target IN ${KITSUNE_TEST_TAPIR_TARGETS})
  if (${tapir_target} STREQUAL "cuda")
    set(TEST_CUDA_TARGET ON)
  elseif (${tapir_target} STREQUAL "hip")
    set(TEST_HIP_TARGET ON)
  elseif (${tapir_target} STREQUAL "lambda")
    set(TEST_LAMBDA_TARGET ON)
  elseif (${tapir_target} STREQUAL "omptask")
    set(TEST_OMPTASK_TARGET ON)
  elseif (${tapir_target} STREQUAL "opencilk")
    set(TEST_OPENCILK_TARGET ON)
  elseif (${tapir_target} STREQUAL "openmp")
    set(TEST_OPENMP_TARGET ON)
  elseif (${tapir_target} STREQUAL "qthreads")
    set(TEST_QTHREADS_TARGET ON)
  elseif (${tapir_target} STREQUAL "realm")
    set(TEST_REALM_TARGET ON)
  elseif (${tapir_target} STREQUAL "serial")
    set(TEST_SERIAL_TARGET ON)
  else ()
    message(FATAL_ERROR "Unknown tapir target: ${tapir_target}")
  endif ()
  message("Testing tapir target: ${tapir_target}")
endforeach ()

# If the cuda tapir target is to be tested, we may also want to test the cuda
# language (.cu) examples. This is typically if we want performance comparisons.
# If we are only interested in Kitsune's correctness, this is not needed.
set(TEST_CUDA_LANG ON)
if (NOT ${KITSUNE_TEST_CUDA_LANG} OR NOT ${TEST_CUDA_TARGET})
  set(TEST_CUDA_LANG OFF)
endif ()

if (${TEST_CUDA_LANG})
  # We could use clang to compile the .cu files, but we use nvcc since we will
  # only be doing this when we are interested in running performance tests.
  execute_process(COMMAND "${KIT_CONFIG}" "--cuda-prefix"
    OUTPUT_VARIABLE CUDA_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  find_program(NVCC nvcc REQUIRED
    PATHS ${CUDA_PREFIX}/bin
    NO_DEFAULT_PATH)
endif ()

# See the comments for the cuda language tests above. Same goes for hip.
set(TEST_HIP_LANG ${KITSUNE_TEST_HIP_LANG})
if (NOT ${KITSUNE_TEST_HIP_LANG} OR NOT ${TEST_HIP_TARGET})
  set(TEST_HIP_LANG OFF)
endif ()

if (${TEST_HIP_LANG})
  # We could use clang to compile the .hip files, but we use hipcc since we will
  # only be doing this when we are interested in running performance tests.
  execute_process(COMMAND ${KIT_CONFIG} "--hip-prefix"
    OUTPUT_VARIABLE HIP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  find_program(HIPCC hipcc REQUIRED
    PATHS ${HIP_PREFIX}/bin
    NO_DEFAULT_PATH)
endif ()
