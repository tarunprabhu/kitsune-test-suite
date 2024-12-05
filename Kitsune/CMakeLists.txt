# The Kitsune tests. For performance comparisons, this contains .cu and .hip
# sources as well. Those languages are not set here because the actual compiler
# used would depend on the CUDA (respectively HIP) installations used to build
# Kitsune's cuda (respectively hip) tapir targets (if they are built at all).
# Those languages will, therefore, only be enabled conditionally. Fortran
# support will likely always be optional, so that will also only be enabled
# after querying the kitsune installation using the kit-config utility.
project(kitsune-test-suite)

set(KITSUNE_C "kitcc")
set(KITSUNE_CXX "kit++")
set(KITSUNE_Fortran "kitfc")
set(KITSUNE_ALL_FRONTENDS
  "${KITSUNE_C};${KITSUNE_CXX};${KITSUNE_Fortran}")
set(KITSUNE_ALL_TAPIR_TARGETS "cuda;hip;opencilk;serial")

set(KITSUNE_TEST_FRONTENDS "" CACHE STRING
  "A space separated list of the Kitsune frontends to test. If this is not \
set, all the available Kitsune frontends will be tested. The allowed values \
in the list are: ${KITSUNE_ALL_FRONTENDS}")

set(KITSUNE_SKIP_FRONTENDS "" CACHE STRING
  "A space-separated list of the Kitsune frontends to skip. The allowed values \
in the list are: ${KITSUNE_ALL_FRONTENDS}")

set(KITSUNE_TEST_TAPIR_TARGETS "" CACHE STRING
  "A space-separated list of the tapir targets to test. If this is not set, \
all Tapir targets enabled in Kitsune will be tested. The allowed values in the \
list are: ${KITSUNE_ALL_TAPIR_TARGETS}")

set(KITSUNE_SKIP_TAPIR_TARGETS "" CACHE STRING
  "A string-separated list of tapir targets to skip. The allowed values in the \
list are: ${KITSUNE_ALL_TAPIR_TARGETS}")

option(KITSUNE_TEST_KOKKOS_MODE
  "Check Kitsune's -fkokkos-mode if it has been enabled"
  ON)

option(KITSUNE_TEST_KOKKOS_LANG
  "Build the Kokkos (.kokkos.cpp) tests with a regular C++ compiler and run \
them when the Kitsune's kokkos mode is tested"
  ON)

option(KITSUNE_TEST_CUDA_LANG
  "Build and run the cuda (.cu) tests when the cuda tapir target is tested"
  OFF)

option(KITSUNE_TEST_HIP_LANG
  "Build and run the hip (.hip) tests when the hip tapir target is tested"
  OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
message(STATUS "module path: ${CMAKE_MODULE_PATH}")

set(TEST_CUDA_LANG ON)
set(TEST_HIP_LANG ON)
set(TEST_KOKKOS_LANG ON)

set(TEST_C ON)
set(TEST_CXX ON)
set(TEST_Fortran ON)

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

string(REPLACE " " ";" TEST_FRONTENDS "${KITSUNE_TEST_FRONTENDS}")
string(REPLACE " " ";" SKIP_FRONTENDS "${KITSUNE_SKIP_FRONTENDS}")

# The top-level CMakeLists.txt will have set the C and CXX languages which means
# the C and C++ compilers will already have been set. If the compilers are
# proper Kitsune frontends (which are the only ones that should be used with
# this test suite), the kit-config utility should be present in the same
# directory.
get_filename_component(COMPILER_BINDIR ${CMAKE_CXX_COMPILER} DIRECTORY)
find_program(KIT_CONFIG kit-config REQUIRED
  PATHS ${COMPILER_BINDIR}
  NO_DEFAULT_PATH)

message(STATUS "Kitsune configuration tool: ${KIT_CONFIG}")

# Check that the C, C++ and Fortran compilers are the same as those that were
# found since this should only be setup with Kitsune.
execute_process(COMMAND ${KIT_CONFIG} "--c-frontend"
  OUTPUT_VARIABLE KITCC
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if (KITCC STREQUAL "")
  set(TEST_C OFF)
elseif (NOT KITCC STREQUAL CMAKE_C_COMPILER)
  message(FATAL_ERROR "C compiler is not Kitsune")
elseif (NOT KITSUNE_TEST_FRONTENDS STREQUAL "")
  if (NOT KITSUNE_C IN_LIST KITSUNE_TEST_FRONTENDS)
    set(TEST_C OFF)
  endif ()
elseif (KITSUNE_C IN_LIST KITSUNE_SKIP_FRONTENDS)
  set(TEST_C OFF)
endif ()
if (TEST_C)
  # Find clang because we may need to compile vanilla C code in the test suite.
  # This should be the one on which Kitsune is based.
  find_program(CLANG clang REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)

  message(STATUS "Testing frontend: ${KITSUNE_C}")
endif ()

execute_process(COMMAND ${KIT_CONFIG} "--c++-frontend"
  OUTPUT_VARIABLE KITXX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if (KITXX STREQUAL "")
  set(TEST_CXX OFF)
elseif (NOT KITXX STREQUAL CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler is not Kitsune")
elseif (NOT KITSUNE_TEST_FRONTENDS STREQUAL "")
  if (NOT KITSUNE_CXX IN_LIST KITSUNE_TEST_FRONTENDS)
    set(TEST_CXX OFF)
  endif ()
elseif (KITSUNE_CXX IN_LIST KITSUNE_SKIP_FRONTENDS)
  set(TEST_CXX OFF)
endif ()
if (TEST_CXX)
  # Find clang++ because we may need to compile vanilla C++ code in the test
  # suite. This should be the one on which Kitsune is based.
  find_program(CLANGXX clang++ REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)

  message(STATUS "Testing frontend: ${KITSUNE_CXX}")
endif ()

execute_process(COMMAND ${KIT_CONFIG} "--fortran-frontend"
  OUTPUT_VARIABLE KITFC
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
# The Fortran language may not have been enabled if TEST_SUITE_FORTRAN was not
# set.
if (NOT KITFC STREQUAL "" AND CMAKE_Fortran_COMPILER STREQUAL "")
  set(CMAKE_Fortran_COMPILER ${KITFC})
  enable_language(Fortran)
endif ()
if (KITFC STREQUAL "")
  set(TEST_Fortran OFF)
elseif (NOT KITFC STREQUAL CMAKE_Fortran_COMPILER)
  message(FATAL_ERROR "Fortran compiler is not Kitsune")
elseif (NOT KITSUNE_TEST_FRONTENDS STREQUAL "")
  if (NOT KITSUNE_Fortran IN_LIST KITSUNE_TEST_FRONTENDS)
    set(TEST_Fortran OFF)
  endif ()
elseif (KITSUNE_Fortran IN_LIST KITSUNE_SKIP_FRONTENDS)
  set(TEST_Fortran OFF)
endif ()
if (TEST_Fortran)
  # Find flang because we may need to compile vanilla Fortran code in the test
  # suite. This should be the one on which Kitsune is based.
  find_program(FLANG flang REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)

  message(STATUS "Testing frontend: ${KITSUNE_Fortran}")
endif ()

execute_process(COMMAND "${KIT_CONFIG}" "--tapir-targets"
  OUTPUT_VARIABLE TAPIR_TARGETS_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

string(REPLACE " " ";" ENABLED_TAPIR_TARGETS "${TAPIR_TARGETS_STR}")
string(REPLACE " " ";" SKIP_TAPIR_TARGETS "${KITSUNE_SKIP_TAPIR_TARGETS}")

# If an explicit list of tapir targets to test has not been provided, test all
# the enabled tapir targets. Otherwise, test the explicit list. All tapir
# targets that were explicitly specified must have been enabled.
if (KITSUNE_TEST_TAPIR_TARGETS STREQUAL "")
  set(TEST_TAPIR_TARGETS "${ENABLED_TAPIR_TARGETS}")
else ()
  string(REPLACE " " ";" TEST_TAPIR_TARGETS "${KITSUNE_TEST_TAPIR_TARGETS}")
  foreach (tapir_target IN TEST_TAPIR_TARGETS)
    if (NOT tapir_target IN_LIST ENABLED_TAPIR_TARGETS)
      message(FATAL_ERROR "'${tapir_target}' has not been enabled")
    endif ()
  endforeach ()
endif ()

# Filter out any tapir targets to skip.
foreach (tapir_target ${SKIP_TAPIR_TARGETS})
  message(STATUS "Skipping tapir target: ${tapir_target}")
  list(REMOVE_ITEM TEST_TAPIR_TARGETS ${tapir_target})
endforeach ()

foreach (tapir_target ${TEST_TAPIR_TARGETS})
  if (tapir_target STREQUAL "cuda")
    set(TEST_CUDA_TARGET ON)
  elseif (tapir_target STREQUAL "hip")
    set(TEST_HIP_TARGET ON)
  elseif (tapir_target STREQUAL "lambda")
    set(TEST_LAMBDA_TARGET ON)
  elseif (tapir_target STREQUAL "omptask")
    set(TEST_OMPTASK_TARGET ON)
  elseif (tapir_target STREQUAL "opencilk")
    set(TEST_OPENCILK_TARGET ON)
  elseif (tapir_target STREQUAL "openmp")
    set(TEST_OPENMP_TARGET ON)
  elseif (tapir_target STREQUAL "qthreads")
    set(TEST_QTHREADS_TARGET ON)
  elseif (tapir_target STREQUAL "realm")
    set(TEST_REALM_TARGET ON)
  elseif (tapir_target STREQUAL "serial")
    set(TEST_SERIAL_TARGET ON)
  else ()
    message(FATAL_ERROR "Unknown tapir target: ${tapir_target}")
  endif ()
  message(STATUS "Testing tapir target: ${tapir_target}")
endforeach ()

# Testing Kitsune's -fkokkos mode
execute_process(COMMAND "${KIT_CONFIG}" "--kokkos-mode"
  OUTPUT_VARIABLE KOKKOS_MODE_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# If Kokkos mode has not been enabled in Kitsune, don't test Kokkos at all.
if (KOKKOS_MODE_STR STREQUAL "OFF" OR NOT KITSUNE_TEST_KOKKOS_MODE)
  set(TEST_KOKKOS_MODE OFF)
endif ()

# If the cuda tapir target is to be tested, we may also want to test the cuda
# language (.cu) examples. This is typically if we want performance comparisons.
# If we are only interested in Kitsune's correctness, this is not needed.
if (NOT KITSUNE_TEST_CUDA_LANG OR NOT TEST_CUDA_TARGET)
  set(TEST_CUDA_LANG OFF)
endif ()

if (TEST_CUDA_LANG)
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
if (NOT KITSUNE_TEST_HIP_LANG OR NOT TEST_HIP_TARGET)
  set(TEST_HIP_LANG OFF)
endif ()
if (TEST_HIP_LANG)
  execute_process(COMMAND ${KIT_CONFIG} "--hip-prefix"
    OUTPUT_VARIABLE HIP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  find_program(HIPCC hipcc REQUIRED
    PATHS ${HIP_PREFIX}/bin
    NO_DEFAULT_PATH)
endif ()

# Similar to Cuda and Hip, we may also want to check the .kokkos.cpp files with
# a regular C++ compiler for performance checks. These are generally only useful
# when doing performance comparisons.
if (NOT KITSUNE_TEST_KOKKOS_LANG OR NOT TEST_KOKKOS_MODE)
  set(TEST_KOKKOS_LANG OFF)
endif ()

add_subdirectory(vecadd)
