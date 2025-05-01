# The Kitsune tests. For performance comparisons, this contains .cu and .hip
# sources as well. Those languages are not set here because the actual compiler
# used would depend on the CUDA (respectively HIP) installations used to build
# Kitsune's cuda (respectively hip) tapir targets (if they are built at all).
# Those languages will, therefore, only be enabled conditionally. Fortran
# support will likely always be optional, so that will also only be enabled
# after querying the kitsune installation using the kit-config utility.
project(kitsune-test-suite)

# The list of languages here must be kept in sync with that known to Kitsune
set(KITSUNE_TEST_LANGS "" CACHE STRING
  "A semicolon-separated list of languages to test. If this is not set, all \
the languages for which frontends have been built will be tested. The \
currently known languages are: c cxx fortran")

set(KITSUNE_SKIP_LANGS "" CACHE STRING
  "A semicolon-separated list of languages to skip")

set(KITSUNE_TEST_TAPIR_TARGETS "" CACHE STRING
  "A semicolon-separated list of tapir targets to test. If this is not set, \
all tapir targets enabled in Kitsune will be tested. Each element of this list \
must be a value that can be passed to the --tapir= option of the frontend")

set(KITSUNE_SKIP_TAPIR_TARGETS "" CACHE STRING
  "A semicolon-separated list of tapir targets to skip. Each element of this \
list must be a value that can be passed to the --tapir= option of the frontend")

set(KITSUNE_FLAGS_EXTRA "" CACHE STRING
  "Additional options to pass to all Kitsune frontends")

set(KITSUNE_C_FLAGS_EXTRA "" CACHE STRING
  "Additional options to pass to Kitsune's C frontend")

set(KITSUNE_CXX_FLAGS_EXTRA "" CACHE STRING
  "Additional options to pass to Kitsune's C++ frontend")

set(KITSUNE_Fortran_FLAGS_EXTRA "" CACHE STRING
  "Additional options to pass to Kitsune's Fortran frontend")

set(KITSUNE_LINKER_FLAGS_EXTRA "" CACHE STRING
  "Additional options to pass to the linker when building executables")

set(KOKKOS_CUDA_PREFIX "" CACHE STRING
  "Path to the Kokkos (cuda \"backend\") install prefix")

set(KOKKOS_HIP_PREFIX "" CACHE STRING
  "Path to the Kokkos (hip \"backend\") install prefix")

option(KITSUNE_TEST_KOKKOS_MODE
  "Check Kitsune's -fkokkos-mode if it has been enabled"
  ON)

option(KITSUNE_BENCHMARK
  "Build and run the performance tests. The tests in the Benchmarks directory \
are intended to be used for both correctness testing and performance. Setting \
this option will result in the non-kitsune single-source tests (.cu, .hip \
etc.) being built and run. This will only run a subset of the tests that are \
intended as benchmarks. NOTE: This may skip some correctness checks"
  OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

# Build and run the vanilla cuda (.cu) tests
set(TEST_VANILLA_CUDA OFF)

# Build and run the vanilla hip (.hip) tests
set(TEST_VANILLA_HIP OFF)

# Build and run the Kokkos tests with a vanilla (non-Kitsune) compiler where
# Kokkos has been built with the cuda backend.
set(TEST_VANILLA_KOKKOS_CUDA OFF)

# Build and run the Kokkos tests with a vanilla (non-Kitsune) compile where
# Kokkos has been built with the hip backend.
set(TEST_VANILLA_KOKKOS_HIP OFF)

# Build and run the Fortran tests with a vanilla (non-Kitsune) compiler
set(TEST_VANILLA_Fortran OFF)

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

# The compiler options to use with the various frontends. These will be a
# combination of any default options required by Kitsune, those required to
# build the tests for the given configuration (Debug, Release etc.) and any
# extra options specified by the user.
set(KITSUNE_C_FLAGS "")
set(KITSUNE_CXX_FLAGS "")
set(KITSUNE_Fortran_FLAGS "")

# There are several known issues with Kitsune:
#
#   - Some tapir targets require optimizations to be enabled in order to work
#     correctly.
#
#   - Some tapir targets do not support debug symbols.
#
# These will eventually get fixed, but for now, we need to work around them.
# We do this by requiring that this test suite be built in Release mode. In
# principle, we should also check that the CMAKE_${LANG}_FLAGS do not contain
# -g and do contain -On (n > 0), but we take the quick route for now.
if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  message(FATAL_ERROR "Kitsune test suite must be built with CMAKE_BUILD_TYPE=Release")
endif ()
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_CONFIG)

# Grep is needed to check if a test passed or failed. The other utilities are
# needed to determine the unique GPU on the machine. These should be present on
# platforms that we care about, but just in case, make sure that we find them.
find_program(GREP grep REQUIRED)
find_program(SORT sort REQUIRED)
find_program(UNIQ uniq REQUIRED)

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
elseif (NOT KITSUNE_TEST_LANGS STREQUAL "")
  if (NOT "c" IN_LIST KITSUNE_TEST_LANGS)
    set(TEST_C OFF)
  endif ()
elseif ("c" IN_LIST KITSUNE_SKIP_LANGS)
  set(TEST_C OFF)
else ()
  set(TEST_C ON)
  set(KITSUNE_C_FLAGS
    ${CMAKE_C_FLAGS}
    ${CMAKE_C_FLAGS_${BUILD_CONFIG}}
    ${KITSUNE_FLAGS_EXTRA}
    ${KITSUNE_C_FLAGS_EXTRA})
  string(STRIP "${KITSUNE_C_FLAGS}" KITSUNE_C_FLAGS)
  string(REPLACE " " ";" KITSUNE_C_FLAGS "${KITSUNE_C_FLAGS}")
  message(STATUS "C compiler options: ${KITSUNE_C_FLAGS}")
endif ()
if (TEST_C)
  # Find clang because we may need to compile vanilla C code in the test suite.
  # This should be the one on which Kitsune is based.
  find_program(CLANG clang REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)
  message(STATUS "Testing language: C")
endif ()

# Check that ${CMAKE_CXX_COMPILER} is the same as the C++ frontend returned by
# kit-config
execute_process(COMMAND "${KIT_CONFIG}" --cxx-frontend
  OUTPUT_VARIABLE KITXX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
if (KITXX STREQUAL "")
  set(TEST_CXX OFF)
elseif (NOT KITXX STREQUAL CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler is not Kitsune [${CMAKE_CXX_COMPILER}]")
elseif (NOT KITSUNE_TEST_LANGS STREQUAL "")
  if (NOT "cxx" IN_LIST KITSUNE_TEST_LANGS)
    set(TEST_CXX OFF)
  endif ()
elseif ("cxx" IN_LIST KITSUNE_SKIP_LANGS)
  set(TEST_CXX OFF)
else ()
  set(TEST_CXX ON)
  set(KITSUNE_CXX_FLAGS
    ${CMAKE_CXX_FLAGS}
    ${CMAKE_CXX_FLAGS_${BUILD_CONFIG}}
    ${KITSUNE_FLAGS_EXTRA}
    ${KITSUNE_CXX_FLAGS_EXTRA})
  string(STRIP "${KITSUNE_CXX_FLAGS}" KITSUNE_CXX_FLAGS)
  string(REPLACE " " ";" KITSUNE_CXX_FLAGS "${KITSUNE_CXX_FLAGS}")
  message(STATUS "C++ compiler options: ${KITSUNE_CXX_FLAGS}")
endif ()
if (TEST_CXX)
  # Find clang++ because we may need to compile vanilla C++ code in the test
  # suite. This should be the one on which Kitsune is based.
  find_program(CLANGXX clang++ REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)
  message(STATUS "Testing langauge: C++")
endif ()

set(KITSUNE_LINKER_FLAGS
  ${CMAKE_EXE_LINKER_FLAGS}
  ${KITSUNE_LINKER_FLAGS_EXTRA})
string(STRIP "${KITSUNE_LINKER_FLAGS}" KITSUNE_LINKER_FLAGS)
string(REPLACE " " ";" KITSUNE_LINKER_FLAGS "${KITSUNE_LINKER_FLAGS}")
message(STATUS "Linker options: ${KITSUNE_LINKER_FLAGS}")

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
elseif (NOT KITSUNE_TEST_LANGS STREQUAL "")
  if (NOT "fortran" IN_LIST KITSUNE_TEST_LANGS)
    set(TEST_Fortran OFF)
  endif ()
elseif ("fortran" IN_LIST KITSUNE_SKIP_LANGS)
  set(TEST_Fortran OFF)
else ()
  set(TEST_Fortran ON)
  # The Fortran language may not have been enabled if TEST_SUITE_FORTRAN was not
  # set.
  enable_language(Fortran)
  set(KITSUNE_Fortran_FLAGS
    ${CMAKE_Fortran_FLAGS}
    ${CMAKE_Fortran_FLAGS_${BUILD_CONFIG}}
    ${KITSUNE_FLAGS_EXTRA}
    ${KITSUNE_Fortran_FLAGS_EXTRA})
  string(STRIP "${KITSUNE_Fortran_FLAGS}" KITSUNE_Fortran_FLAGS)
  string(REPLACE " " ";" KITSUNE_Fortran_FLAGS "${KITSUNE_Fortran_FLAGS}")
  message(STATUS "Fortran compiler options: ${KITSUNE_Fortran_FLAGS}")
endif ()
if (TEST_Fortran)
  # Find flang because we may need to compile vanilla Fortran code in the test
  # suite. This should be the one on which Kitsune is based.
  find_program(FLANG flang flang-new REQUIRED
    PATHS ${COMPILER_BINDIR}
    NO_DEFAULT_PATH)

  message(STATUS "Testing language: Fortran")
endif ()

# Sanity check the inputs to KITSUNE_TEST_LANGS and KITSUNE_SKIP_LANGS
execute_process(COMMAND "${KIT_CONFIG}" --known-langs
  OUTPUT_VARIABLE KNOWN_LANGS_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE " " ";" KNOWN_LANGS "${KNOWN_LANGS_STR}")

foreach (LANG ${KITSUNE_TEST_LANGS})
  if (NOT LANG IN_LIST KNOWN_LANGS)
    message(FATAL_ERROR "Unsupported language: ${LANG}")
  endif ()
endforeach ()

foreach (LANG ${KITSUNE_SKIP_LANGS})
  if (NOT LANG IN_LIST KNOWN_LANGS)
    message(FATAL_ERROR "Unknown language: ${LANG}")
  endif ()
endforeach ()

# Sanity check the inputs to KITSUNE_TEST_TAPIR_TARGETS and
# KITSUNE_SKIP_TAPIR_TARGETS
execute_process(COMMAND "${KIT_CONFIG}" --known-tapir-targets
  OUTPUT_VARIABLE KNOWN_TAPIR_TARGETS_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE " " ";" KNOWN_TAPIR_TARGETS "${KNOWN_TAPIR_TARGETS_STR}")

foreach (TAPIR_TARGET ${KITSUNE_TEST_TAPIR_TARGETS})
  if (NOT ${TAPIR_TARGET} IN_LIST KNOWN_TAPIR_TARGETS)
    message(FATAL_ERROR "Unknown tapir target: ${TAPIR_TARGET}")
  endif ()
endforeach ()

foreach (TAPIR_TARGET ${KITSUNE_SKIP_TAPIR_TARGETS})
  if (NOT ${TAPIR_TARGET} IN_LIST KNOWN_TAPIR_TARGETS)
    message(FATAL_ERROR "Unknown tapir target: ${TAPIR_TARGET}")
  endif ()
endforeach ()

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
    # If the cuda target has been built, but the device on which this is run
    # does not have an NVIDIA GPU, skip the cuda target.
    #
    # nvptx-arch is guaranteed to be built when Kitsune is built, so use that
    # to determine if an NVIDIA GPU is present.
    find_program(NVPTX_ARCH nvptx-arch REQUIRED
      PATHS ${COMPILER_BINDIR}
      NO_DEFAULT_PATH)

    execute_process(COMMAND "${NVPTX_ARCH}"
      COMMAND ${SORT}
      COMMAND ${UNIQ}
      OUTPUT_VARIABLE NVPTX_ARCHS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    string(REPLACE "\n" ";" NVPTX_ARCHS "${NVPTX_ARCHS}")
    list(LENGTH NVPTX_ARCHS NUM_NVIDIA_GPUS)

    if (NUM_NVIDIA_GPUS EQUAL 1)
      list(GET NVPTX_ARCHS 0 NVIDIA_ARCH)
      message(STATUS "NVIDIA GPU found: ${NVIDIA_ARCH}")
      message(STATUS "Testing tapir target: cuda")
      set(TEST_CUDA_TARGET ON)
    elseif (NUM_NVIDIA_GPUS EQUAL 0)
      message(WARNING "No NVIDIA GPU's found! Skipping tapir target: cuda")
      set(TEST_CUDA_TARGET OFF)
    else ()
      message(WARNING "NVIDIA GPU's with different architectures found: \
'${NVPTX_ARCHS}'! Skipping tapir target: cuda")
      set(TEST_CUDA_TARGET OFF)
    endif ()
  elseif (tapir_target STREQUAL "hip")
    # If the cuda target has been built, but the device on which this is run
    # does not have an AMD GPU, skip the hip target.
    #
    # amdgpu-arch is guaranteed to be built when Kitsune is built, so use that
    # to determine if an AMD GPU is present.
    find_program(AMDGPU_ARCH amdgpu-arch REQUIRED
      PATHS ${COMPILER_BINDIR}
      NO_DEFAULT_PATH)

    execute_process(COMMAND "${AMDGPU_ARCH}"
      COMMAND ${SORT}
      COMMAND ${UNIQ}
      OUTPUT_VARIABLE AMDGPU_ARCHS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    string(REPLACE "\n" ";" AMDGPU_ARCHS "${AMDGPU_ARCHS}")
    list(LENGTH AMDGPU_ARCHS NUM_AMD_GPUS)

    if (NUM_AMD_GPUS EQUAL 1)
      list(GET AMDGPU_ARCHS 0 AMD_ARCH)
      message(STATUS "AMD GPU found: ${AMD_ARCH}")
      message(STATUS "Testing tapir target: hip")
      set(TEST_HIP_TARGET ON)
    elseif (NUM_AMD_GPUS EQUAL 0)
      message(WARNING "No AMD GPU's found! Skipping tapir target: hip")
      set(TEST_HIP_TARGET OFF)
    else ()
      message(WARNING "AMD GPU's with different architectures found: \
'${AMDGPU_ARCHS}'! Skipping tapir target: hip")
      set(TEST_HIP_TARGET OFF)
    endif ()
  elseif (tapir_target STREQUAL "lambda")
    set(TEST_LAMBDA_TARGET ON)
    message(STATUS "Testing tapir target: ${tapir_target}")
  elseif (tapir_target STREQUAL "omptask")
    set(TEST_OMPTASK_TARGET ON)
    message(STATUS "Testing tapir target: ${tapir_target}")
  elseif (tapir_target STREQUAL "opencilk")
    set(TEST_OPENCILK_TARGET ON)
    message(STATUS "Testing tapir target: ${tapir_target}")
  elseif (tapir_target STREQUAL "openmp")
    set(TEST_OPENMP_TARGET ON)
    message(STATUS "Testing tapir target: ${tapir_target}")
  elseif (tapir_target STREQUAL "qthreads")
    set(TEST_QTHREADS_TARGET ON)
    message(STATUS "Testing tapir target: ${tapir_target}")
  elseif (tapir_target STREQUAL "realm")
    set(TEST_REALM_TARGET ON)
    message(STATUS "Testing tapir target: ${tapir_target}")
  elseif (tapir_target STREQUAL "serial")
    # We never want to test the serial tapir target when in benchmarking mode.
    # The benchmarks are all built to test problem sizes that benefit from
    # parallel execution. The code produced by the serial tapir target is far,
    # far too slow in such cases.
    if (KITSUNE_BENCHMARK)
      set(TEST_SERIAL_TARGET OFF)
      message(STATUS "Skipping tapir target when benchmarking: serial")
    else ()
      set(TEST_SERIAL_TARGET ON)
      message(STATUS "Testing tapir target: ${tapir_target}")
    endif ()
  else ()
    message(FATAL_ERROR "Unknown tapir target: ${tapir_target}")
  endif ()
endforeach ()

# Testing Kitsune's -fkokkos mode
execute_process(COMMAND "${KIT_CONFIG}" --kokkos
  OUTPUT_VARIABLE KOKKOS_MODE_STR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

# If Kokkos mode has not been enabled in Kitsune, don't test Kokkos at all.
if (KOKKOS_MODE_STR STREQUAL "ON" AND KITSUNE_TEST_KOKKOS_MODE)
  set(TEST_KOKKOS_MODE ON)
else ()
  set(TEST_KOKKOS_MODE OFF)
endif ()

if (TEST_CUDA_TARGET AND KITSUNE_BENCHMARK)
  set(TEST_VANILLA_CUDA ON)

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

  find_library(LIBCUDART cudart_static REQUIRED
    PATHS ${CUDA_PREFIX}/lib ${CUDA_PREFIX}/lib64
    NO_DEFAULT_PATH)

  find_library(LIBCUDA cuda REQUIRED)

  set(CMAKE_CUDA_COMPILER ${NVCC})
  enable_language(CUDA)

  # Ensure that we collect the compile and link times. This is conditionally
  # enabled for consistency with what is done with C and C++.
  if(TEST_SUITE_COLLECT_COMPILE_TIME)
    set(CMAKE_CUDA_COMPILE_OBJECT
      "${CMAKE_BINARY_DIR}/tools/timeit --summary <OBJECT>.time ${CMAKE_CUDA_COMPILE_OBJECT}")
    set(CMAKE_CUDA_LINK_EXECUTABLE
      "${CMAKE_BINARY_DIR}/tools/timeit --summary <TARGET>.link.time ${CMAKE_CUDA_LINK_EXECUTABLE}")
  endif()

  if (TEST_KOKKOS_MODE)
    # In principle, we could fetch and install the right Kokkos version instead
    # of requiring the user to provide it, but that would require determining
    # the GPU architecture on this machine and mapping that to Kokkos'
    # configuration flags and also keeping up with any changes to Kokkos'
    # configuaration-time parameters. All that is really not worth it since this
    # benchmarking mode is only really for the core developers anyway.
    if (NOT KOKKOS_CUDA_PREFIX)
      message(FATAL_ERROR "KOKKOS_CUDA_PREFIX must be set in order to run Kokkos (cuda)")
    endif ()

    find_library(LIBKOKKOSCORE_CUDA kokkoscore REQUIRED
      PATHS ${KOKKOS_CUDA_PREFIX}/lib ${KOKKOS_CUDA_PREFIX}/lib64
      NO_DEFAULT_PATH)

    set(TEST_VANILLA_KOKKOS_CUDA ON)
  endif ()
endif ()

# See the comments for the cuda language tests above. Same goes for hip.
if (TEST_HIP_TARGET AND KITSUNE_BENCHMARK)
  set(TEST_VANILLA_HIP ON)

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

  # This is utterly ridiculous! (I could think of several stronger, and perhaps
  # more appropriate words, but that would be unprofessional)
  #
  # We cannot set CMAKE_HIP_COMPILER to the hipcc that we want to use (found
  # above) for ... reasons. Instead, we are supposed to either use clang
  # directly - which we don't really want to do because clang here is kitsune -
  # or let cmake find hipcc. But cmake looks for hipcc in $PATH. It doesn't
  # seem to care if you set ROCM_PATH or not. So the only thing we can do is
  # to set $PATH to the directory containing hipcc just to let cmake find the
  # right one, then restore $PATH.
  set(PATH $ENV{PATH})
  set(ENV{PATH} "${HIP_PREFIX}/bin:${PATH}")
  enable_language(HIP)
  set(ENV{PATH} "${PATH}")

  # Ensure that we collect the compile and link times. This is conditionally
  # enabled for consistency with what is done with C and C++.
  if(TEST_SUITE_COLLECT_COMPILE_TIME)
    set(CMAKE_HIP_COMPILE_OBJECT
      "${CMAKE_BINARY_DIR}/tools/timeit --summary <OBJECT>.time ${CMAKE_HIP_COMPILE_OBJECT}")
    set(CMAKE_HIP_LINK_EXECUTABLE
      "${CMAKE_BINARY_DIR}/tools/timeit --summary <TARGET>.link.time ${CMAKE_HIP_LINK_EXECUTABLE}")
  endif()

  if (TEST_KOKKOS_MODE)
    # In principle, we could fetch and install the right Kokkos version instead
    # of requiring the user to provide it, but that would require determining
    # the GPU architecture on this machine and mapping that to Kokkos'
    # configuration flags and also keeping up with any changes to Kokkos'
    # configuaration-time parameters. All that is really not worth it since this
    # benchmarking mode is only really for the core developers anyway.
    if (NOT KOKKOS_HIP_PREFIX)
      message(FATAL_ERROR "KOKKOS_HIP_PREFIX must be set in order to run Kokkos (hip)")
    endif ()

    message(STATUS "KOKKOS_HIP_PREFIX: ${KOKKOS_HIP_PREFIX}")
    find_library(LIBKOKKOSCORE_HIP kokkoscore REQUIRED
      PATHS ${KOKKOS_HIP_PREFIX}/lib ${KOKKOS_HIP_PREFIX}/lib64
      NO_DEFAULT_PATH)

    set(TEST_VANILLA_KOKKOS_HIP ON)
  endif ()
endif ()

# CMake adds -Werror=date-time in order to get reproducible builds. This is a
# good idea in general, but we use nvcc in "benchmark" mode which does not
# support this. So just remove it since we are without bitwise reproducibliity
# (for now). And yes, this really should not have been a "definition" in the
# first place since this very much isn't one.
remove_definitions(-Werror=date-time)

# The tests in the Benchmarks directory can be used both for performance
# comparisons and correctness checks.
add_subdirectory(Benchmarks)

# The single-source tests in particular are only intended for correctness. They
# are generally very small and there is nothing to be gained when running them
# during benchmarking. It is possible that we could do some benchmarking with
# the multisource tests at some point, but for now, we don't.
if (NOT KITSUNE_BENCHMARK)
  add_subdirectory(SingleSource)
  add_subdirectory(MultiSource)
endif ()

add_subdirectory(utils)

file(COPY lit.local.cfg DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

# Use configure_file instead of file(COPY) because that will ensure that the
# file will be copied when it is changed. Unlike lit.local.cfg, this will change
# during development.
configure_file(run-tests ${CMAKE_CURRENT_BINARY_DIR}/run-tests COPYONLY)
