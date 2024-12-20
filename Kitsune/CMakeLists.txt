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

set(KITSUNE_TEST_FRONTENDS "" CACHE STRING
  "The Kitsune frontends to test. If this is not set, all the built Kitsune \
frontends will be tested")

set(KITSUNE_SKIP_FRONTENDS "" CACHE STRING
  "The Kitsune frontends to skip")

set(KITSUNE_TEST_TAPIR_TARGETS "" CACHE STRING
  "The tapir targets to test. If this is not set, all Tapir targets enabled in \
Kitsune will be tested. Each element of this list must be a value that can be \
passed to the -ftapir= option of the frontend")

set(KITSUNE_SKIP_TAPIR_TARGETS "" CACHE STRING
  "The list of tapir targets to skip. Each element of this list must be a \
value that can be passed to the -ftapir= option of the frontend")

set(KITSUNE_TEST_INPUT_SIZE "default" CACHE STRING
  "The size of the input in the tests that support it. Must be one of \
\"default\", or \"small\". Defaults to \"default\"")

option(KITSUNE_TEST_KOKKOS_MODE
  "Check Kitsune's -fkokkos-mode if it has been enabled"
  ON)

option(KITSUNE_RUN_ALL
  "Build and run the performance tests. The tests in the Benchmarks directory \
are intended to be used for both correctness testing and performance. Setting \
this option will result in the non-kitsune single-source tests (.cu, .hip \
etc.) being built and run."
  OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Build and run the vanilla cuda (.cu) tests
set(TEST_CUDA_LANG OFF)

# Build and run the vanilla hip (.hip) tests
set(TEST_HIP_LANG OFF)

# Build and run the Kokkos tests with a vanilla (non-Kitsune) compiler
set(TEST_KOKKOS_LANG OFF)

# Build and run the Fortran tests with a vanilla (non-Kitsune) compiler
set(TEST_Fortran_LANG OFF)

set(TEST_C OFF)
set(TEST_CXX OFF)
set(TEST_Fortran OFF)

set(TEST_KOKKOS_MODE OFF)
set(TEST_CUDA_TARGET OFF)
set(TEST_HIP_TARGET OFF)
set(TEST_LAMBDA_TARGET OFF)
set(TEST_OMPTASK_TARGET OFF)
set(TEST_OPENCILK_TARGET OFF)
set(TEST_OPENMP_TARGET OFF)
set(TEST_QTHREADS_TARGET OFF)
set(TEST_REALM_TARGET OFF)
set(TEST_SERIAL_TARGET OFF)

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

# Check that ${CMAKE_C_COMPILER} is the same as the C frontend returned by
# kit-config
execute_process(COMMAND "${KIT_CONFIG}" --c-frontend
  OUTPUT_VARIABLE KITCC
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if (KITCC STREQUAL "")
  set(TEST_C OFF)
elseif (NOT KITCC STREQUAL CMAKE_C_COMPILER)
  message(FATAL_ERROR "C compiler is not Kitsune [${CMAKE_C_COMPILER}]")
elseif (NOT KITSUNE_TEST_FRONTENDS STREQUAL "")
  if (NOT KITSUNE_C IN_LIST KITSUNE_TEST_FRONTENDS)
    set(TEST_C OFF)
  endif ()
elseif (KITSUNE_C IN_LIST KITSUNE_SKIP_FRONTENDS)
  set(TEST_C OFF)
else ()
  set(TEST_C ON)
endif ()
if (TEST_C)
  # Find clang because we may need to compile vanilla C code in the test suite.
  # This should be the one on which Kitsune is based.
  find_program(CLANG clang REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)

  message(STATUS "Testing frontend: ${KITSUNE_C}")
endif ()

# Check that ${CMAKE_CXX_COMPILER} is the same as the C++ frontend returned by
# kit-config
execute_process(COMMAND "${KIT_CONFIG}" --c++-frontend
  OUTPUT_VARIABLE KITXX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if (KITXX STREQUAL "")
  set(TEST_CXX OFF)
elseif (NOT KITXX STREQUAL CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler is not Kitsune [${CMAKE_CXX_COMPILER}]")
elseif (NOT KITSUNE_TEST_FRONTENDS STREQUAL "")
  if (NOT KITSUNE_CXX IN_LIST KITSUNE_TEST_FRONTENDS)
    set(TEST_CXX OFF)
  endif ()
elseif (KITSUNE_CXX IN_LIST KITSUNE_SKIP_FRONTENDS)
  set(TEST_CXX OFF)
else ()
  set(TEST_CXX ON)
endif ()
if (TEST_CXX)
  # Find clang++ because we may need to compile vanilla C++ code in the test
  # suite. This should be the one on which Kitsune is based.
  find_program(CLANGXX clang++ REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)

  message(STATUS "Testing frontend: ${KITSUNE_CXX}")
endif ()

# Check that ${CMAKE_Fortran_COMPILER} is the same as the Fortran frontend
# returned by kit-config
execute_process(COMMAND "${KIT_CONFIG}" --fortran-frontend
  OUTPUT_VARIABLE KITFC
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if (KITFC STREQUAL "")
  set(TEST_Fortran OFF)
elseif (NOT KITFC STREQUAL CMAKE_Fortran_COMPILER)
  message(FATAL_ERROR "Fortran compiler is not Kitsune [${CMAKE_Fortran_COMPILER}]")
elseif (NOT KITSUNE_TEST_FRONTENDS STREQUAL "")
  if (NOT KITSUNE_Fortran IN_LIST KITSUNE_TEST_FRONTENDS)
    set(TEST_Fortran OFF)
  endif ()
elseif (KITSUNE_Fortran IN_LIST KITSUNE_SKIP_FRONTENDS)
  set(TEST_Fortran OFF)
else ()
  set(TEST_Fortran ON)
endif ()
if (TEST_Fortran)
  # The Fortran language may not have been enabled if TEST_SUITE_FORTRAN was not
  # set.
  enable_language(Fortran)

  # Find flang because we may need to compile vanilla Fortran code in the test
  # suite. This should be the one on which Kitsune is based.
  find_program(FLANG flang REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)

  message(STATUS "Testing frontend: ${KITSUNE_Fortran}")
endif ()

# Sanity check the inputs to KITSUNE_TEST_FRONTENDS and KITSUNE_SKIP_FRONTENDS
execute_process(COMMAND "${KIT_CONFIG}" --all-frontends
  OUTPUT_VARIABLE ALL_FRONTENDS_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE " " ";" ALL_FRONTENDS "${ALL_FRONTENDS_STR}")

foreach (FRONTEND ${KITSUNE_TEST_FRONTENDS})
  if (NOT FRONTEND IN_LIST ALL_FRONTENDS)
    message(FATAL_ERROR "Unknown frontend: ${FRONTEND}")
  endif ()
endforeach ()

foreach (FRONTEND ${KITSUNE_SKIP_FRONTENDS})
  if (NOT FRONTEND IN_LIST ALL_FRONTENDS)
    message(FATAL_ERROR "Unknown frontend: ${FRONTEND}")
  endif ()
endforeach ()

# Sanity check the inputs to KITSUNE_TEST_TAPIR_TARGETS and
# KITSUNE_SKIP_TAPIR_TARGETS
execute_process(COMMAND "${KIT_CONFIG}" --all-tapir-targets
  OUTPUT_VARIABLE ALL_TAPIR_TARGETS_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE " " ";" ALL_TAPIR_TARGETS "${ALL_TAPIR_TARGETS_STR}")

foreach (TAPIR_TARGET ${KITSUNE_TEST_TAPIR_TARGETS})
  if (NOT ${TAPIR_TARGET} IN_LIST ALL_TAPIR_TARGETS)
    message(FATAL_ERROR "Unknown tapir target: ${TAPIR_TARGET}")
  endif ()
endforeach ()

foreach (TAPIR_TARGET ${KITSUNE_SKIP_TAPIR_TARGETS})
  if (NOT ${TAPIR_TARGET} IN_LIST ALL_TAPIR_TARGETS)
    message(FATAL_ERROR "Unknown tapir target: ${TAPIR_TARGET}")
  endif ()
endforeach ()

# Sanity check the test input size
if (NOT KITSUNE_TEST_INPUT_SIZE STREQUAL "default" AND
    NOT KITSUNE_TEST_INPUT_SIZE STREQUAL "small")
  message(FATAL_ERROR "Unknown test input size: ${KITSUNE_TEST_INPUT_SIZE}")
endif ()

execute_process(COMMAND "${KIT_CONFIG}" --tapir-targets
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
    # It is possible that the cuda target has been built, but the device on
    # which the tests are built (and run) does not have an NVIDIA GPU. In this
    # case, skip the cuda target
    execute_process(COMMAND "${KIT_CONFIG}" --cuda-prefix
      OUTPUT_VARIABLE CUDA_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)

    find_program(NVIDIA_SMI nvidia-smi REQUIRED)

    execute_process(COMMAND "${NVIDIA_SMI}"
      RESULT_VARIABLE NVIDIA_SMI_RESULT
      OUTPUT_QUIET
      ERROR_QUIET)

    if (NOT NVIDIA_SMI_RESULT STREQUAL "0")
      set(TEST_CUDA_TARGET OFF)
      message(STATUS "No NVIDIA GPU's found")
      message(STATUS "Skipping tapir target: cuda")
    else ()
      set(TEST_CUDA_TARGET ON)
      message(STATUS "Testing tapir target: cuda")
    endif ()
  elseif (tapir_target STREQUAL "hip")
    # It is possible that the hip target has been built, but the device on which
    # the tests are built (and run) does not have an AMD GPU. In this case, skip
    # the hip target
    execute_process(COMMAND "${KIT_CONFIG}" --hip-prefix
      OUTPUT_VARIABLE HIP_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)

    find_program(ROCM_SMI rocm-smi REQUIRED)

    execute_process(COMMAND "${ROCM_SMI}"
      RESULT_VARIABLE ROCM_SMI_RESULT
      OUTPUT_VARIABLE ROCM_SMI_OUT
      ERROR_VARIABLE ROCM_SMI_ERR)

    # Of course, rocm-smi will not return an error code if a GPU is not found.
    # Yes, yes, one can make an argument for why the absence of a GPU is
    # "normal", but it is not unreasonable to expect "found GPU" to be success
    # and "GPU not found" to be a failure in a tool like this. Plus, there is
    # guarantee that this string will not change. (*#&$(@*@!))
    string(FIND "${ROCM_SMI_OUT}" "Driver not initialized" ERR_OUT)
    string(FIND "${ROCM_SMI_ERR}" "Driver not initialized" ERR_ERR)

    if (NOT ROCM_SMI_RESULT STREQUAL 0
        OR NOT ERR_OUT STREQUAL -1
        OR NOT ERR_ERR STREQUAL -1)
      set(TEST_HIP_TARGET OFF)
      message(STATUS "No AMD GPU's found")
      message(STATUS "Skipping tapir target: hip")
    else ()
      set(TEST_HIP_TARGET ON)
      message(STATUS "Testing tapir target: hip")
    endif ()
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
execute_process(COMMAND "${KIT_CONFIG}" --kokkos-mode
  OUTPUT_VARIABLE KOKKOS_MODE_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

# If Kokkos mode has not been enabled in Kitsune, don't test Kokkos at all.
if (KOKKOS_MODE_STR STREQUAL "ON" AND KITSUNE_TEST_KOKKOS_MODE)
  set(TEST_KOKKOS_MODE ON)
else ()
  set(TEST_KOKKOS_MODE OFF)
endif ()

if (TEST_CUDA_TARGET AND KITSUNE_RUN_ALL)
  set(TEST_CUDA_LANG ON)

  # We could use clang to compile the .cu files, but since this is only useful
  # for performance comparisons, use nvcc since that is expected to be the best
  # compiler for cuda code.
  execute_process(COMMAND "${KIT_CONFIG}" --cuda-prefix
    OUTPUT_VARIABLE CUDA_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

  find_program(NVCC nvcc REQUIRED
    PATHS ${CUDA_PREFIX}/bin
    NO_DEFAULT_PATH)

  set(CMAKE_CUDA_COMPILER ${NVCC})
  enable_language(CUDA)
endif ()

# See the comments for the cuda language tests above. Same goes for hip.
if (TEST_HIP_TARGET AND KITSUNE_RUN_ALL)
  set(TEST_HIP_LANG ON)

  # We could use clang to compile the .hip files, but since this is only useful
  # for performance comparisons, use hipcc since that is expected to be the best
  # compiler for hip code.
  execute_process(COMMAND "${KIT_CONFIG}" --hip-prefix
    OUTPUT_VARIABLE HIP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

  find_program(HIPCC hipcc REQUIRED
    PATHS ${HIP_PREFIX}/bin
    NO_DEFAULT_PATH)

  set(CMAKE_HIP_COMPILER ${HIPCC})
  enable_language(HIP)
endif ()

# Similar to Cuda and Hip, we may also want to check the .kokkos.cpp files with
# a regular C++ compiler for performance checks. These are generally only useful
# when doing performance comparisons.
if (TEST_KOKKOS_MODE AND KITSUNE_RUN_ALL)
  set(TEST_KOKKOS_LANG ON)
endif ()

add_subdirectory(Benchmarks)

# FIXME: The SingleSource directory contains tests that themselves have not been
# checked for "sanity". Until then, we don't descend into it, but once we check
# that those tests are actually sound, this should be enabled.
# add_subdirectory(SingleSource)

# FIXME: Not all tests in the MultiSource directory have valid CMakeLists.txt or
# Makefile's yet. Until those are setup, don't descend into the directory at
# all.
# add_subdirectory(MultiSource)

file(COPY lit.local.cfg DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
