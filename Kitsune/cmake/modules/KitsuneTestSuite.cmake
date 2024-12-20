##===- KitsuneTestSuite.cmake ---------------------------------------------===##
#
# This defines functions that are used by the Kitsune-specific tests in this
# suite. They generally mimic the corresponding llvm_* functions in this suite.
#
##===----------------------------------------------------------------------===##

include(SingleMultiSource)

# Look at the file name to determine the source language. This is not really a
# source language in the strict sense of the term, but something for internal
# use that will help determine what to do with the file. The "not widely
# recognized as languages" languages that this will return are:
#
#    kitc       C files with kitsune-specific extensions
#    kitc++     C++ files with kitsune-specific extensions
#    kokkos     C++ files with Kokkos. These may or may not contain any
#               Kitsune-specific extensions
#
#    c          C files without kitsune-specific extensions
#    c++        C++ files without kitsune-specific extensions
#    fortran    Fortran files
#    cuda       Cuda files (those with a .cu extension)
#    hip        Hip files (those with a .hip extension)
#
function (source_language source lang)
  # We treat Kokkos as its own language.
  if (source MATCHES ".+[.]kokkos[.]cpp$" OR source MATCHES ".+[.]kokkos[.]c$")
    set(${lang} "kokkos" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]kit[.]c$" OR source MATCHES ".+[.]kit[.]c$")
    set(${lang} "kitc" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]kit[.]cpp$" OR source MATCHES ".+[.]kit[.]cc$")
    set(${lang} "kitc++" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]c$")
    set(${lang} "c" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]cpp$" OR source MATCHES ".+[.]cc$")
    set(${lang} "c++" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]cu$")
    set(${lang} "cuda" PARENT_SCOPE)
  elseif (source MATCHES ".+[.]hip$")
    set(${lang} "hip" PARENT_SCOPE)
  elseif (source MATCHES ".+[.][Ff]$" OR
      source MATCHES ".+[.][Ff]90$" OR
      source MATCHES ".+[.][Ff]95$" OR
      source MATCHES ".+[.][Ff]03$" OR
      source MATCHES ".+[.][Ff]08$")
    set(${lang} "fortran" PARENT_SCOPE)
  else ()
    message(FATAL_ERROR "Cannot determine source language: ${source}")
  endif ()
endfunction ()

# Setup a single-source test for the given tapir target.
#
#     source         The absolute path to the source file
#     lang           The source language
#     tapir_target   The tapir target. A value of "none" is a special case. It
#                    will be treated as "not to be built with any tapir target"
#     cmdargs        A list of command line arguments to be passed when running
#                    the test
#     data       A list of files containing that will be used by the test. These
#                will be copied into the build directory.
#
function (kit_singlesource_test source lang tapir_target cmdargs data)
  get_filename_component(base "${source}" NAME_WLE)
  string(REPLACE "." "-" base "${base}")
  if (tapir_target STREQUAL "none")
    set(target "${base}-nokit")
  else ()
    set(target "${base}-${tapir_target}")
  endif ()

  message(STATUS "Setting up test: ${target}")
  llvm_test_executable_no_test(${target} ${source})
  llvm_test_run(WORKDIR "%S" "${cmdargs}")
  llvm_test_data(${target} ${data})
  llvm_add_test_for_target(${target})

  set(tapir_flags "-ftapir=${tapir_target}")
  set(kokkos_flags "-fkokkos -fkokkos-no-init")
  if (NOT tapir_target STREQUAL "none")
    target_compile_options(${target} BEFORE PUBLIC "${tapir_flags}")
    target_link_options(${target} BEFORE PUBLIC "${tapir_flags}")
  endif ()

  if (lang STREQUAL "kitc++" OR lang STREQUAL "kokkos")
    target_compile_options(${target} BEFORE PUBLIC "-fno-exceptions")
  endif ()


  if (lang STREQUAL "kokkos")
    target_compile_options(${target} BEFORE PUBLIC "${kokkos_flags}")
    target_link_options(${target} BEFORE PUBLIC "${kokkos_flags}")
  endif ()
endfunction()

# Add tests for all tapir targets being tested. This should not be used by
# consumers. They should call kitsune_singlesource() instead.
#
# ARGUMENTS
#
#     source     The absolute path to the source file
#     lang       The source language
#     cmdargs    A list of command line arguments to be passed when running the
#                test
#     data       A list of files containing that will be used by the test. These
#                will be copied into the build directory.
#
function(kit_singlesource_all_targets source lang cmdargs data)
  if (TEST_CUDA_TARGET)
    kit_singlesource_test(${source} ${lang} "cuda" "${cmdargs}" "${data}")
  endif ()
  if (TEST_HIP_TARGET)
    kit_singlesource_test(${source} ${lang} "hip" "${cmdargs}" "${data}")
  endif()
  if (TEST_LAMBDA_TARGET)
    kit_singlesource_test(${source} ${lang} "lambda" "${cmdargs}" "${data}")
  endif ()
  if (TEST_OMPTASK_TARGET)
    kit_singlesource_test(${source} ${lang} "omptask" "${cmdargs}" "${data}")
  endif ()
  if (TEST_OPENCILK_TARGET)
    kit_singlesource_test(${source} ${lang} "opencilk" "${cmdargs}" "${data}")
  endif()
  if (TEST_OPENMP_TARGET)
    kit_singlesource_test(${source} ${lang} "openmp" "${cmdargs}" "${data}")
  endif ()
  if (TEST_QTHREADS_TARGET)
    kit_singlesource_test(${source} ${lang} "qthreads" "${cmdargs}" "${data}")
  endif ()
  if (TEST_REALM_TARGET)
    kit_singlesource_test(${source} ${lang} "realm" "${cmdargs}" "${data}")
  endif ()
  if (TEST_SERIAL_TARGET)
    kit_singlesource_test(${source} ${lang} "serial" "${cmdargs}" "${data}")
  endif ()
endfunction()

# Configure the current directory as a SingleSource subdirectory - i.e. every
# C/C++/Fortran file is treated as its own test.
#
# ARGUMENTS
#
#     <none>
#
# KEYWORD ARGUMENTS
#
#     CMDARGS    The list of command line arguments to be passed when running
#                the test
#
#     DATA       A list of files containing data that will be used by the
#                test. These will be copied to the build directory
#
function(kitsune_singlesource)
  cmake_parse_arguments(KIT "" "" "CMDARGS;DATA" ${ARGN})
  set(cmdargs "${KIT_CMDARGS}")
  set(data "${KIT_DATA}")

  file(GLOB sources
    *.c
    *.cpp *.cc
    *.cu
    *.f *.F *.f90 *.F90 *.f03 *.F03 *.f08 *.F08
    *.hip)
  foreach(source ${sources})
    set(lang)
    source_language(${source} lang)

    if (lang STREQUAL "cuda")
      if (TEST_CUDA_LANG)
        kit_singlesource_test(${source} ${lang} "none" "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "hip")
      if (TEST_HIP_LANG)
        kit_singlesource_test(${source} ${lang} "none" "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "kokkos")
      if (TEST_KOKKOS_LANG)
        kit_singlesource_test(${source} ${lang} "none" "${cmdargs}" "${data}")
      endif ()
      if (TEST_KOKKOS_MODE)
        # We only care about testing Kokkos with the GPU-centric targets
        if (TEST_CUDA_TARGET)
          kit_singlesource_test(${source} ${lang} "cuda" "${cmdargs}" "${data}")
        endif ()
        if (TEST_HIP_TARGET)
          kit_singlesource_test(${source} ${lang} "hip" "${cmdargs}" "${data}")
        endif ()
      endif ()
    elseif (lang STREQUAL "kitc")
      if (TEST_C)
        kit_singlesource_all_targets(${source} ${lang} "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "kitc++")
      if (TEST_CXX)
        kit_singlesource_all_targets(${source} ${lang} "${cmdargs}" "${data}")
      endif ()
    elseif (lang STREQUAL "fortran")
      # Fortran is not yet supported. Complain loudly so we know to change this
      # and take a closer look at everything when Fortran is supported.
      message(FATAL_ERROR "Kitsune does not yet support Fortran: ${source}")
      if (TEST_Fortran)
        kit_singlesource_all_targets(${source} ${lang} "${cmdargs}" "${data}")
      endif ()
    else ()
      message(FATAL_ERROR "Testing of file not supported: ${source} [${lang}]")
    endif ()
  endforeach()
endfunction()
